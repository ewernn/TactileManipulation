import h5py
import numpy as np
import argparse
import os

def explore_hdf5(file_path):
    """
    Explore the structure of an HDF5 file and print information about its contents.
    
    Args:
        file_path: Path to the HDF5 file
    """
    print(f"Exploring HDF5 file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        
        # Function to recursively explore groups
        def explore_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}, Keys: {list(obj.keys())}")
                for key in obj.keys():
                    if isinstance(obj[key], h5py.Group):
                        # Only print group info, don't recurse to avoid deep nesting
                        print(f"  Subgroup: {name}/{key}, Keys: {list(obj[key].keys())}")
                    else:
                        # For datasets, print shape and dtype
                        dataset = obj[key]
                        print(f"  Dataset: {name}/{key}, Shape: {dataset.shape}, Type: {dataset.dtype}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                
                # If it's a small dataset, print some sample values
                if len(obj.shape) == 0 or (len(obj.shape) == 1 and obj.shape[0] < 10):
                    try:
                        print(f"  Values: {obj[...]}")
                    except:
                        print("  Values: [Could not display]")
                elif len(obj.shape) > 0:
                    try:
                        # Print the first few values
                        if obj.shape[0] > 0:
                            print(f"  First value: {obj[0]}")
                    except:
                        print("  First value: [Could not display]")
        
        # Explore attributes
        print("Attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        
        # Explore each top-level group
        for key in f.keys():
            explore_group(key, f[key])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    
    args = parser.parse_args()
    
    explore_hdf5(args.file)
