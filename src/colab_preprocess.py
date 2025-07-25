from google.colab import files
import zipfile
import os
import shutil

# Upload the zipped folder
uploaded = files.upload()

# Get the uploaded zip file name
zip_file = next(iter(uploaded))

# Optional: specify destination directory
destination_dir = "movies"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(destination_dir)

print(f"Files extracted to: {destination_dir}")

# Fix nested directory structure if it exists
nested_dir = os.path.join(destination_dir, "movies copy")

# Check if the nested directory exists
if os.path.exists(nested_dir):
    # Move everything from "movies/movies copy" into "movies"
    for item in os.listdir(nested_dir):
        s = os.path.join(nested_dir, item)
        d = os.path.join(destination_dir, item)
        
        # Check if destination already exists to avoid errors
        if not os.path.exists(d):
            shutil.move(s, d)
        else:
            print(f"Warning: {item} already exists in {destination_dir}, skipping...")
    
    # Remove the now-empty "movies copy" directory
    os.rmdir(nested_dir)
    print("✓ Moved contents from 'movies copy' into 'movies'")
else:
    print("No nested 'movies copy' directory found. Structure is correct.")

# Optional: List the final contents to verify
print("\nFinal contents of movies directory:")
for item in os.listdir(destination_dir):
    print(f"  - {item}")
