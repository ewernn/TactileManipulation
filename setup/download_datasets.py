# Create download script (save as download_stack.py)
from huggingface_hub import hf_hub_download
import os

# Create directory if it doesn't exist
os.makedirs("datasets/mimicgen/core", exist_ok=True)

# Download with correct repository ID
file_path = hf_hub_download(
    repo_id="amandlek/mimicgen_datasets",  # Correct repo ID
    filename="core/stack_d0.hdf5",         # Correct path within repo
    repo_type="dataset",        # Specify it's a dataset repository
    local_dir="datasets"                   # Local directory
)

print(f"File downloaded to: {file_path}")