import os
import json
import re
import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Conv1D, Dropout, Dense, Concatenate, Layer, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

# --- NLTK Downloads ---
# Ensure necessary NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# --- Global Configuration & Constants ---
DATA_BASE_PATH = "." # Base directory for the dataset.
ATTENTION_LOSS_WEIGHT = 0.5 # Hyperparameter: How much the explanation loss contributes to the total loss.
HUMAN_FEEDBACK_SAMPLE_SIZE = 5 # Number of samples to show a human for feedback per fold.
N_FOLDS = 5 # Number of folds for cross-validation.
PRETRAIN_EPOCHS = 5 # Epochs for the initial training phase.
FINETUNE_EPOCHS = 10 # Max epochs for the feedback-driven fine-tuning phase.
FINETUNE_PATIENCE = 3 # Patience for early stopping during fine-tuning.
EXPLAINABILITY_TOP_K_PERCENTAGE = 0.2 # Percentage of tokens to perturb for faithfulness metrics.

# ───── Feedback Abstractions ─────
# Defines a base class for feedback providers for structural consistency.
class FeedbackProvider:
    def get_feedback(self, items: Sequence[Any], **kw) -> Sequence[Any]:
        raise NotImplementedError

# ───── Data I/O and Preprocessing ─────
# Loads a JSONL file (one JSON object per line).
def load_jsonl(path: str) -> list[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Data file not found: {path}")
        return []
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return []

# Reads the raw text of a movie review given its annotation ID.
def review_text(aid: str, base_path: str = DATA_BASE_PATH) -> str:
    fp = os.path.join(base_path, "movies", "docs", aid)
    if os.path.exists(fp):
        try:
            with open(fp, encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"Error reading file {fp}: {e}")
            return ""
    return ""

# Converts string labels ('POS'/'NEG') to binary (1/0).
def label_of(entry: dict) -> int:
    return 1 if entry.get("classification", "POS") == "POS" else 0

# Cleans and tokenizes a string of text.
def preprocess(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s']", "", text.lower()) # Lowercase and remove special characters except apostrophes.
    return word_tokenize(text)

# A pre-run check to ensure data files are accessible before starting the main process.
def check_validation_data_loading(val_file_path: str, docs_base_path: str) -> bool:
    logging.info(f"--- Pre-check: Validating data loading for {val_file_path} ---")
    val_entries = load_jsonl(val_file_path)
    if not val_entries:
        logging.error(f"Pre-check FAILED: Could not load or empty file: {val_file_path}")
        return False
    test_texts = [txt for e in val_entries if "annotation_id" in e and (txt := review_text(e["annotation_id"], base_path=docs_base_path))]
    if not test_texts:
        logging.error("Pre-check FAILED: No usable text data loaded from validation docs.")
        return False
    logging.info("Pre-check SUCCESS: Found valid text data for validation set.")
    return True

# ───── Model & Attention Mechanism ─────
# Custom Keras layer to compute attention scores and a context vector.
class AttentionLayer(Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True # Important for handling padded sequences.

    def build(self, input_shape):
        # Create trainable weights for the layer.
        self.W = self.add_weight(name="attention_W", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_b", shape=(1,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        # Forward pass logic for the attention mechanism.
        e = K.tanh(K.dot(x, self.W) + self.b) # Compute alignment scores.
        e = K.squeeze(e, axis=-1)
        
        # Apply the mask to ignore padded parts of the sequence.
        if mask is not None:
            adder = (1.0 - K.cast(mask, K.floatx())) * -1e9
            e += adder
            
        a = K.softmax(e, axis=1) # Normalize scores to a probability distribution (attention weights).
        a_expanded = K.expand_dims(a, axis=-1)
        ctx = K.sum(x * a_expanded, axis=1) # Compute the context vector (weighted sum).
        return ctx, a

# Function to build the complete TextCNN model architecture.
def build_cnn(vocab: int, maxlen: int) -> tf.keras.Model:
    inp = tf.keras.Input((maxlen,), dtype="int32", name="input_layer")
    # Embedding layer with mask_zero=True to handle padding.
    emb = Embedding(vocab, 100, mask_zero=True, name="embedding_layer")(inp)
    
    outs_ctx = []
    last_attention_weights = None

    # Create three parallel convolutional paths for different n-gram sizes.
    for k_size in (3, 4, 5):
        conv_name = f"conv1d_k{k_size}"
        attn_name = f"attention_k{k_size}"
        c = Conv1D(100, k_size, activation="relu", padding="same", name=conv_name)(emb)
        ctx, att_weights = AttentionLayer(name=attn_name)(c)
        outs_ctx.append(ctx)
        # Capture the attention weights from the k=5 path as the model's explanation output.
        if k_size == 5:
            last_attention_weights = Lambda(lambda x: tf.identity(x), name=attn_name + "_output")(att_weights)

    x = Concatenate(name="concatenate_contexts")(outs_ctx) # Combine the results from the parallel paths.
    x = Dropout(0.5, name="dropout_layer")(x) # Apply dropout for regularization.
    out_sentiment = Dense(1, activation="sigmoid", name="output_dense")(x) # Final prediction layer.

    if last_attention_weights is None:
        raise ValueError("Attention weights from k=5 layer were not captured.")

    # Define the model with two outputs: the prediction and the explanation.
    model = tf.keras.Model(inputs=inp, outputs=[out_sentiment, last_attention_weights], name="AttentionCNN_DualOutput")
    return model

# ───── Custom Loss Function ─────
# Computes the loss for the "Attention as Loss" mechanism.
def cosine_similarity_loss(y_true_attention, y_pred_attention):
    y_true_attention = tf.cast(y_true_attention, tf.float32)
    y_pred_attention = tf.cast(y_pred_attention, tf.float32)
    
    # Create a mask to only compute loss for samples with human feedback.
    mask = K.cast(K.any(K.not_equal(y_true_attention, 0.0), axis=-1), K.floatx())
    
    # Normalize vectors to calculate cosine similarity.
    y_true_norm = K.l2_normalize(y_true_attention, axis=-1)
    y_pred_norm = K.l2_normalize(y_pred_attention, axis=-1)
    
    cosine_similarity = K.sum(y_true_norm * y_pred_norm, axis=-1)
    loss = (1.0 - cosine_similarity) * mask
    
    # Average the loss only over the active (masked) samples.
    return K.sum(loss) / (K.sum(mask) + K.epsilon())

# ───── Feedback Mechanism Classes ─────
# Helper function to get attention weights from a trained model for a given text.
def get_attention(model: tf.keras.Model, tok: Tokenizer, text: str, maxlen: int) -> Tuple[list[str], np.ndarray]:
    processed_toks = preprocess(text)
    if not processed_toks: return [], np.array([])
    seq = tok.texts_to_sequences([" ".join(processed_toks)])
    pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post", dtype="int32")
    try:
        prediction_output = model.predict(pad, verbose=0)
        # The model has two outputs, the second one is the attention.
        if isinstance(prediction_output, list) and len(prediction_output) > 1:
            raw_att = prediction_output[1]
            att_weights = raw_att[0][:len(processed_toks)] # Trim to the original sentence length.
            sum_w = np.sum(att_weights)
            if sum_w > 1e-6:
                att_weights = att_weights / sum_w # Normalize to sum to 1.
            return processed_toks, att_weights
        else:
            # Fallback if the model doesn't return two outputs.
            return processed_toks, np.full(len(processed_toks), 1.0 / len(processed_toks) if processed_toks else 0)
    except Exception:
        # Fallback in case of prediction error.
        return processed_toks, np.full(len(processed_toks), 1.0 / len(processed_toks) if processed_toks else 0)

# Calculates the automated AI feedback score.
class AttentionFocusScorer(FeedbackProvider):
    def __init__(self, model, tok, maxlen, th=0.05):
        self.model, self.tok, self.maxlen, self.th = model, tok, maxlen, th

    def get_feedback(self, items: Sequence[str], **kw) -> List[float]:
        scores = []
        for txt in items:
            _, w = get_attention(self.model, self.tok, txt, self.maxlen)
            # The score is the proportion of tokens with attention below a small threshold.
            scores.append(np.sum(w < self.th) / w.size if w.size > 0 else 0.0)
        return scores

# Manages the interactive process of getting feedback from a human.
class InteractiveHumanValidator(FeedbackProvider):
    def __init__(self, model: tf.keras.Model, tok: Tokenizer, maxlen: int, top_k_sentences: int = 2):
        self.model, self.tok, self.maxlen = model, tok, maxlen
        self.top_k_sentences = top_k_sentences
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def get_feedback(self, items: Sequence[str], **kw) -> List[Tuple[float, Optional[List[int]]]]:
        feedback_results = []
        BOLD = '\033[1m'; RESET = '\033[0m'

        if HUMAN_FEEDBACK_SAMPLE_SIZE == 0:
            return [(0.0, None) for _ in items]

        for item_idx, original_text in enumerate(items):
            print("\n" + "="*80)
            print(f"--- Human Validation: Item {item_idx+1}/{len(items)} ---")
            
            model_processed_tokens, token_attention_weights = get_attention(self.model, self.tok, original_text, self.maxlen)
            if not model_processed_tokens or token_attention_weights.size == 0:
                feedback_results.append((0.0, None))
                continue

            # Get the model's prediction and confidence score.
            seq_for_pred = self.tok.texts_to_sequences([" ".join(model_processed_tokens)])
            pad_seq_for_pred = pad_sequences(seq_for_pred, maxlen=self.maxlen, padding="post", dtype="int32")
            try:
                prediction_output = self.model.predict(pad_seq_for_pred, verbose=0)
                conf = float(prediction_output[0][0][0]) if isinstance(prediction_output, list) else float(prediction_output[0][0])
                pred_label = "Positive" if conf > 0.5 else "Negative"
            except Exception:
                conf, pred_label = 0.5, "Unknown"

            # Heuristic to score sentences based on the average attention of their tokens.
            sentences = self.sent_tokenizer.tokenize(original_text)
            sentence_scores = []
            sentence_token_indices_map = []
            current_token_offset = 0
            temp_model_tokens_lower = [t.lower() for t in model_processed_tokens]

            for sent_text in sentences:
                sent_tokens_raw = word_tokenize(sent_text)
                sent_tokens_processed_style = [re.sub(r"[^a-z0-9\s']", "", t.lower()) for t in sent_tokens_raw if t]
                current_sentence_token_indices = []
                current_sentence_attention_sum = 0.0
                found_tokens_for_sent = 0
                
                temp_offset_search = current_token_offset
                for s_tok in sent_tokens_processed_style:
                    try:
                        idx_in_model_tokens = temp_model_tokens_lower.index(s_tok, temp_offset_search)
                        if idx_in_model_tokens < len(token_attention_weights):
                           current_sentence_token_indices.append(idx_in_model_tokens)
                           current_sentence_attention_sum += token_attention_weights[idx_in_model_tokens]
                           temp_offset_search = idx_in_model_tokens + 1
                           found_tokens_for_sent +=1
                    except ValueError:
                        pass
                
                avg_sent_attention = current_sentence_attention_sum / found_tokens_for_sent if found_tokens_for_sent > 0 else 0.0
                sentence_scores.append(avg_sent_attention)
                sentence_token_indices_map.append(current_sentence_token_indices)
                if temp_offset_search > current_token_offset:
                    current_token_offset = temp_offset_search

            if not sentence_scores:
                feedback_results.append((0.0, None))
                continue

            # Select the top-scoring sentences to present to the human.
            sorted_sentence_indices = np.argsort(sentence_scores)[::-1]
            actual_top_k = min(self.top_k_sentences, len(sentences))
            selected_indices = sorted_sentence_indices[:actual_top_k]

            # Display the context and rationale to the human annotator.
            print(f"\nMODEL PREDICTION: {pred_label} (Confidence: {conf:.2f})")
            print("-" * 25)
            print("\n--- FULL REVIEW (RATIONALE HIGHLIGHTED) ---")
            for i, sent_text in enumerate(sentences):
                if i in selected_indices:
                    print(f"{BOLD}{sent_text}{RESET}")
                else:
                    print(sent_text)
            print("-" * 25)
            print("\n--- MODEL'S PROPOSED RATIONALE SENTENCES ---")
            for sent_idx in selected_indices:
                print(f"- {sentences[sent_idx]}")
            
            # Get the human's binary 'yes'/'no' feedback.
            ans = ""
            while ans not in ['y', 'n']:
                try:
                    ans = input(f"\nDo these sentences make a sensible rationale for the '{pred_label}' prediction? (y/n): ").strip().lower()
                except EOFError:
                    ans = 'n'
            
            score = 1.0 if ans == "y" else 0.0
            approved_token_indices = None
            if ans == 'y':
                # If approved, collect the token indices for the attention target.
                approved_token_indices = []
                for sent_idx in selected_indices:
                    approved_token_indices.extend(sentence_token_indices_map[sent_idx])
                approved_token_indices = sorted(list(set(approved_token_indices)))
            
            feedback_results.append((score, approved_token_indices))
        return feedback_results

# Orchestrates the feedback loop, combining human and AI signals.
@dataclass
class HybridFeedback:
    human_module: FeedbackProvider
    artificial_module: FeedbackProvider
    maxlen: int
    alpha: float = 0.3

    def combine_feedback(self, items: Sequence[Any], **kw) -> Tuple[np.ndarray, np.ndarray]:
        ai_scores = np.array(self.artificial_module.get_feedback(items), dtype=float)
        num_items = len(items)
        human_scores = np.full(num_items, -1.0, dtype=float) # -1.0 indicates no human feedback.
        human_approved_indices = [None] * num_items

        # Select a subset of items for human review based on the AI score.
        query_indices = list(range(num_items))
        if HUMAN_FEEDBACK_SAMPLE_SIZE > 0 and HUMAN_FEEDBACK_SAMPLE_SIZE < num_items:
             sorted_by_ai_score_indices = np.argsort(ai_scores) # Prioritize low scores (diffuse attention).
             query_indices = sorted_by_ai_score_indices[:HUMAN_FEEDBACK_SAMPLE_SIZE]
        
        items_to_query = [items[i] for i in query_indices] if HUMAN_FEEDBACK_SAMPLE_SIZE > 0 else []

        if items_to_query:
             human_feedback_results = self.human_module.get_feedback(items_to_query)
             for i, (score, approved_idxs) in enumerate(human_feedback_results):
                 original_index = query_indices[i]
                 human_scores[original_index] = float(score)
                 if approved_idxs is not None:
                     human_approved_indices[original_index] = approved_idxs
        
        # Blend human and AI scores using the alpha parameter to create sample weights.
        final_human_component = np.where(human_scores != -1.0, human_scores, ai_scores)
        blended_weights = self.alpha * final_human_component + (1 - self.alpha) * ai_scores
        blended_weights = np.clip(blended_weights, 0.0, 1.0)

        # Create the target attention vectors for the 'Attention as Loss' mechanism.
        target_attentions = np.zeros((num_items, self.maxlen), dtype=float)
        for i in range(num_items):
            approved_idxs = human_approved_indices[i]
            if approved_idxs is not None and human_scores[i] == 1.0:
                valid_idxs = [idx for idx in approved_idxs if idx < self.maxlen]
                if valid_idxs:
                     binary_vec = np.zeros(self.maxlen)
                     binary_vec[valid_idxs] = 1.0
                     target_attentions[i] = binary_vec / np.sum(binary_vec) # Normalize to a distribution.
        return blended_weights.astype("float32"), target_attentions.astype("float32")

# ───── Explainability Metrics ─────
# Helper function to get the model's confidence score for the correct label.
def get_probability_for_label(predictions: Union[np.ndarray, List[np.ndarray]], label: int) -> float:
    pred_output = predictions[0] if isinstance(predictions, list) else predictions
    prob_positive = pred_output[0][0]
    return prob_positive if label == 1 else 1.0 - prob_positive

# Calculates faithfulness metrics by perturbing the top-K sentences.
def calculate_sentence_based_faithfulness(
    model_explain: tf.keras.Model, text: str, true_label: int, tokenizer: Tokenizer,
    maxlen: int, top_k_sentences: int = 2, padding_token_id: int = 0
) -> Tuple[Optional[float], Optional[float]]:
    # Get the model's attention for the full text.
    original_tokens, attention_weights = get_attention(model_explain, tokenizer, text, maxlen)
    if not original_tokens or attention_weights.size == 0: return None, None

    # Identify the top-K sentences based on average attention.
    try:
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(text)
    except Exception: return None, None
        
    sentence_scores, sentence_token_indices_map = [], []
    current_token_offset = 0
    temp_model_tokens_lower = [t.lower() for t in original_tokens]

    for sent_text in sentences:
        sent_tokens_raw = word_tokenize(sent_text)
        sent_tokens_processed_style = [re.sub(r"[^a-z0-9\s']", "", t.lower()) for t in sent_tokens_raw if t]
        current_indices, current_sum, found_tokens = [], 0.0, 0
        temp_offset_search = current_token_offset
        for s_tok in sent_tokens_processed_style:
            try:
                idx = temp_model_tokens_lower.index(s_tok, temp_offset_search)
                current_indices.append(idx)
                current_sum += attention_weights[idx]
                temp_offset_search = idx + 1
                found_tokens +=1
            except (ValueError, IndexError): pass
        
        sentence_scores.append(current_sum / found_tokens if found_tokens > 0 else 0.0)
        sentence_token_indices_map.append(current_indices)
        if temp_offset_search > current_token_offset:
            current_token_offset = temp_offset_search
    
    if not sentence_scores: return None, None

    sorted_indices = np.argsort(sentence_scores)[::-1]
    actual_top_k = min(top_k_sentences, len(sentences))
    
    rationale_token_indices = []
    for i in range(actual_top_k):
        rationale_token_indices.extend(sentence_token_indices_map[sorted_indices[i]])
    rationale_token_indices = sorted(list(set(rationale_token_indices)))
    
    if not rationale_token_indices: return None, None

    # Get the model's prediction on the original, unperturbed text.
    original_seq_ids = tokenizer.texts_to_sequences([" ".join(original_tokens)])[0]
    original_pad = pad_sequences([original_seq_ids], maxlen=maxlen, padding="post", dtype="int32")
    prob_original = get_probability_for_label(model_explain.predict(original_pad, verbose=0), true_label)

    # Deletion Test (Comprehensiveness): Remove the rationale sentences.
    deleted_seq = list(original_seq_ids)
    for idx in rationale_token_indices:
        if idx < len(deleted_seq): deleted_seq[idx] = padding_token_id
    pad_deleted = pad_sequences([deleted_seq], maxlen=maxlen, padding="post", dtype="int32")
    prob_deleted = get_probability_for_label(model_explain.predict(pad_deleted, verbose=0), true_label)
    deletion_score = prob_original - prob_deleted

    # Sufficiency Test: Keep only the rationale sentences.
    kept_seq = [padding_token_id] * len(original_seq_ids)
    for idx in rationale_token_indices:
        if idx < len(kept_seq): kept_seq[idx] = original_seq_ids[idx]
    pad_kept = pad_sequences([kept_seq], maxlen=maxlen, padding="post", dtype="int32")
    prob_kept = get_probability_for_label(model_explain.predict(pad_kept, verbose=0), true_label)
    sufficiency_score = prob_original - prob_kept

    return deletion_score, sufficiency_score

# ───── Main Execution Pipeline ─────
def main():
    # ------------------- CHOOSE YOUR EXPERIMENT MODE HERE -------------------
    EXPERIMENT_MODE = 'FULL_HYBRID' 
    ALPHA_SETTING = 1.0 
    # --------------------------------------------------------------------------

    logging.info(f"==========================================================")
    logging.info(f"--- STARTING RUN FOR EXPERIMENT: {EXPERIMENT_MODE} (Alpha: {ALPHA_SETTING if EXPERIMENT_MODE not in ['BASELINE'] else 'N/A'}) ---")
    logging.info(f"==========================================================")

    # --- Data Loading and Preprocessing ---
    val_file_to_check = os.path.join(DATA_BASE_PATH, "movies", "val.jsonl")
    if not check_validation_data_loading(val_file_to_check, DATA_BASE_PATH):
        return
    logging.info("--- Loading Training Data ---")
    train_file = os.path.join(DATA_BASE_PATH, "movies", "train.jsonl")
    train = load_jsonl(train_file)
    if not train: return
    texts, labels = [], []; missing_docs = 0
    for e in train:
        if "annotation_id" not in e: continue
        txt = review_text(e["annotation_id"], base_path=DATA_BASE_PATH)
        if txt: texts.append(txt); labels.append(label_of(e))
        else: missing_docs += 1
    if missing_docs > 0: logging.warning(f"{missing_docs} train docs missing/empty.")
    if not texts: logging.error("No valid training texts found."); return
    
    logging.info("--- Processing Training Data ---")
    tok = Tokenizer(oov_token="<UNK>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    maxlen = max(len(s) for s in seqs if s)
    vocab_size = len(tok.word_index) + 1
    X = pad_sequences(seqs, maxlen=maxlen, padding="post", dtype="int32")
    y_sentiment_full = np.array(labels, dtype="float32")
    logging.info(f"Data processed: Vocab size={vocab_size}, Maxlen={maxlen}")

    # --- Setup for K-Fold Cross-Validation ---
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    final_model_state = None

    # --- Main K-Fold Loop ---
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        logging.info(f"\n--- Starting Fold {fold}/{N_FOLDS} for {EXPERIMENT_MODE} ---")
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr_sentiment, yva_sentiment = y_sentiment_full[tr_idx], y_sentiment_full[va_idx]
        train_items = [texts[i] for i in tr_idx]

        # --- STAGE 1: Pre-training to get a common starting point for all models ---
        logging.info(f"Fold {fold}: Pre-training base model...")
        pretrain_model_arch = build_cnn(vocab=vocab_size, maxlen=maxlen)
        pretrain_model_for_training = tf.keras.Model(
            inputs=pretrain_model_arch.input, 
            outputs=pretrain_model_arch.get_layer('output_dense').output
        )
        pretrain_model_for_training.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        es_pretrain = EarlyStopping(monitor="val_loss", patience=FINETUNE_PATIENCE, restore_best_weights=True, verbose=0)
        
        pretrain_model_for_training.fit(
            Xtr.astype("int32"), ytr_sentiment.astype("float32"),
            epochs=PRETRAIN_EPOCHS, batch_size=32, verbose=0,
            validation_data=(Xva.astype("int32"), yva_sentiment.astype("float32")),
            callbacks=[es_pretrain]
        )
        pretrain_weights = pretrain_model_arch.get_weights()
        del pretrain_model_arch, pretrain_model_for_training

        # --- STAGE 2: Run the selected experiment ---
        final_fold_model_arch = build_cnn(vocab=vocab_size, maxlen=maxlen)
        acc = 0.0

        if EXPERIMENT_MODE == 'BASELINE':
            logging.info(f"Fold {fold}: Evaluating BASELINE model...")
            final_fold_model_arch.set_weights(pretrain_weights)
            model_to_evaluate = tf.keras.Model(inputs=final_fold_model_arch.input, outputs=final_fold_model_arch.get_layer('output_dense').output)
            model_to_evaluate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            loss, acc = model_to_evaluate.evaluate(Xva.astype("int32"), yva_sentiment.astype("float32"), verbose=0)
        
        else: # For all other experiments that involve fine-tuning
            temp_model_for_feedback = build_cnn(vocab=vocab_size, maxlen=maxlen)
            temp_model_for_feedback.set_weights(pretrain_weights)
            human_fb = InteractiveHumanValidator(temp_model_for_feedback, tok, maxlen)
            ai_fb    = AttentionFocusScorer(temp_model_for_feedback, tok, maxlen)
            hybrid   = HybridFeedback(human_fb, ai_fb, maxlen=maxlen, alpha=ALPHA_SETTING)
            
            wtr_scalar, ytr_attn_target = hybrid.combine_feedback(
                train_items, human_kwargs={"sample_size": HUMAN_FEEDBACK_SAMPLE_SIZE}
            )
            wtr_scalar_f32 = wtr_scalar.astype("float32")
            ytr_list_targets = [ytr_sentiment.astype("float32"), ytr_attn_target.astype("float32")]
            
            yva_attn_target_zeros = np.zeros((len(Xva), maxlen), dtype="float32")
            yva_list_targets = [yva_sentiment.astype("float32"), yva_attn_target_zeros]
            validation_data_dual = (Xva.astype("int32"), yva_list_targets)
            validation_data_single = (Xva.astype("int32"), yva_sentiment.astype("float32"))
            es_finetune = EarlyStopping(monitor="val_loss", patience=FINETUNE_PATIENCE, restore_best_weights=True, verbose=1)

            if EXPERIMENT_MODE == 'FULL_HYBRID':
                logging.info(f"Fold {fold}: Fine-tuning FULL_HYBRID model...")
                final_fold_model_arch.set_weights(pretrain_weights)
                final_fold_model_arch.compile(optimizer=tf.keras.optimizers.Adam(), loss=['binary_crossentropy', cosine_similarity_loss], loss_weights=[1.0, ATTENTION_LOSS_WEIGHT], metrics=[['accuracy'], []])
                final_fold_model_arch.fit(Xtr.astype("int32"), ytr_list_targets, sample_weight=wtr_scalar_f32, epochs=FINETUNE_EPOCHS, batch_size=32, validation_data=validation_data_dual, callbacks=[es_finetune], verbose=1)
                eval_results = final_fold_model_arch.evaluate(Xva.astype("int32"), yva_list_targets, verbose=0, return_dict=True)
                acc = eval_results.get('output_dense_accuracy', eval_results.get('accuracy', 0.0))

            elif EXPERIMENT_MODE == 'ATTENTION_LOSS_ONLY':
                logging.info(f"Fold {fold}: Fine-tuning ATTENTION_LOSS_ONLY model...")
                final_fold_model_arch.set_weights(pretrain_weights)
                final_fold_model_arch.compile(optimizer=tf.keras.optimizers.Adam(), loss=['binary_crossentropy', cosine_similarity_loss], loss_weights=[1.0, ATTENTION_LOSS_WEIGHT], metrics=[['accuracy'], []])
                final_fold_model_arch.fit(Xtr.astype("int32"), ytr_list_targets, epochs=FINETUNE_EPOCHS, batch_size=32, validation_data=validation_data_dual, callbacks=[es_finetune], verbose=1)
                eval_results = final_fold_model_arch.evaluate(Xva.astype("int32"), yva_list_targets, verbose=0, return_dict=True)
                acc = eval_results.get('output_dense_accuracy', eval_results.get('accuracy', 0.0))

            elif EXPERIMENT_MODE == 'SAMPLE_WEIGHTING_ONLY':
                logging.info(f"Fold {fold}: Fine-tuning SAMPLE_WEIGHTING_ONLY model...")
                model_to_train_weighted = tf.keras.Model(inputs=final_fold_model_arch.input, outputs=final_fold_model_arch.get_layer('output_dense').output)
                model_to_train_weighted.set_weights(pretrain_weights)
                model_to_train_weighted.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                model_to_train_weighted.fit(Xtr.astype("int32"), ytr_sentiment, sample_weight=wtr_scalar_f32, epochs=FINETUNE_EPOCHS, batch_size=32, validation_data=validation_data_single, callbacks=[es_finetune], verbose=1)
                loss, acc = model_to_train_weighted.evaluate(Xva.astype("int32"), yva_sentiment.astype("float32"), verbose=0)
                final_fold_model_arch.set_weights(model_to_train_weighted.get_weights())

        logging.info(f"Fold {fold} [{EXPERIMENT_MODE}] Validation Accuracy: {acc:.4f}")
        fold_accuracies.append(acc)

        if fold == N_FOLDS:
            logging.info(f"Final fold finished. Saving model weights from {EXPERIMENT_MODE} run.")
            final_model_state = final_fold_model_arch.get_weights()

    # --- Plotting and Final Evaluation ---
    if fold_accuracies:
        avg_acc = np.mean(fold_accuracies)
        logging.info(f"\n--- Experiment Complete: {EXPERIMENT_MODE} ---")
        logging.info(f"Average Validation Accuracy Across {N_FOLDS} Folds: {avg_acc:.4f}")
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, N_FOLDS + 1), fold_accuracies, marker="o", linestyle='-')
        plt.title(f"{EXPERIMENT_MODE} Model: Validation Accuracy per Fold (Alpha={ALPHA_SETTING if EXPERIMENT_MODE not in ['BASELINE'] else 'N/A'})")
        plt.xlabel("Fold Number"); plt.ylabel("Accuracy"); plt.xticks(range(1, N_FOLDS + 1))
        plt.grid(True); plt.show()

    if final_model_state:
        logging.info(f"\n--- Evaluating final model state of {EXPERIMENT_MODE} on Held-Out Set ---")
        final_model_dual_output = build_cnn(vocab_size, maxlen)
        final_model_dual_output.set_weights(final_model_state)
        final_model_single_output = tf.keras.Model(inputs=final_model_dual_output.input,
                                                   outputs=final_model_dual_output.get_layer('output_dense').output)
        final_model_single_output.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        val_file_path = os.path.join(DATA_BASE_PATH, "movies", "val.jsonl")
        val_data = load_jsonl(val_file_path)
        test_texts_final, test_labels_final = [], []
        if not val_data: logging.error(f"No data in {val_file_path}")
        else:
             for e in val_data:
                 if "annotation_id" not in e: continue
                 txt = review_text(e["annotation_id"], base_path=DATA_BASE_PATH)
                 if txt: test_texts_final.append(txt); test_labels_final.append(label_of(e))

        if not test_texts_final: logging.error("No text data loaded for final evaluation.")
        else:
            seqs_test_final = tok.texts_to_sequences(test_texts_final)
            X_test_final = pad_sequences(seqs_test_final, maxlen=maxlen, padding="post", truncating="post", dtype="int32")
            y_test_final = np.array(test_labels_final, dtype="float32")

            if X_test_final.shape[0] > 0:
                logging.info("Evaluating final model state (unweighted accuracy)...")
                loss, acc = final_model_single_output.evaluate(X_test_final, y_test_final, verbose=1)
                print(f"\nFINAL [{EXPERIMENT_MODE}] Held-out val.jsonl Accuracy: {acc:.4f}")

                logging.info(f"\n--- Calculating Explainability Metrics for [{EXPERIMENT_MODE}] ---")
                all_deletion_scores = []
                all_comprehensiveness_scores = []
                num_explain_samples = min(len(test_texts_final), 20)
                padding_id_for_explain = 0

                for i in range(num_explain_samples):
                    text_sample = test_texts_final[i]
                    label_sample = test_labels_final[i]
                    del_score, comp_score = calculate_sentence_based_faithfulness(
                        final_model_dual_output, text_sample, int(label_sample), tok, maxlen,
                        top_k_sentences=2, padding_token_id=padding_id_for_explain
                    )
                    if del_score is not None: all_deletion_scores.append(del_score)
                    if comp_score is not None: all_comprehensiveness_scores.append(comp_score)

                if all_deletion_scores:
                    avg_deletion = np.mean(all_deletion_scores)
                    print(f"Average Deletion Score: {avg_deletion:.4f} (Higher is better)")
                if all_comprehensiveness_scores:
                    avg_comprehensiveness = np.mean(all_comprehensiveness_scores)
                    print(f"Average Comprehensiveness Score (Drop): {avg_comprehensiveness:.4f} (Lower is better)")
            else: 
                logging.error("X_test_final for final evaluation is empty.")
    else:
        logging.warning(f"No final model state saved for {EXPERIMENT_MODE}. Cannot perform final evaluation.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(module)s:%(lineno)d:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    np.random.seed(42)
    tf.random.set_seed(42)
    main()

