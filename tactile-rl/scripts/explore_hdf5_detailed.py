import h5py
import numpy as np
import argparse
import os
import json
from datetime import datetime

def explore_hdf5_detailed(file_path, output_file=None):
    """
    Explore the structure of an HDF5 file in detail and save the output to a file.
    
    Args:
        file_path: Path to the HDF5 file
        output_file: Path to save the output (if None, will generate a filename)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hdf5_exploration_{os.path.basename(file_path).replace('.hdf5', '')}_{timestamp}.txt"
    
    print(f"Exploring HDF5 file: {file_path}")
    print(f"Results will be saved to: {output_file}")
    
    with open(output_file, 'w') as out_file:
        out_file.write(f"HDF5 File Exploration: {file_path}\n")
        out_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        with h5py.File(file_path, 'r') as f:
            out_file.write(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB\n")
            out_file.write(f"Root keys: {list(f.keys())}\n\n")
            
            # Explore attributes
            out_file.write("Attributes:\n")
            for attr_name, attr_value in f.attrs.items():
                out_file.write(f"  {attr_name}: {attr_value}\n")
            out_file.write("\n")
            
            # Function to recursively explore groups and datasets with proper handling
            def explore_group(group, prefix="", depth=0, max_depth=3):
                if depth > max_depth:
                    return  # Avoid going too deep in the hierarchy
                
                # Print group info
                if prefix:
                    out_file.write(f"{prefix} (Group)\n")
                    out_file.write(f"  Keys: {list(group.keys())}\n")
                
                # Explore each item in the group
                for key in group.keys():
                    item_path = f"{prefix}/{key}" if prefix else key
                    try:
                        item = group[key]
                        
                        if isinstance(item, h5py.Group):
                            # If it's a group, recursively explore
                            explore_group(item, item_path, depth + 1, max_depth)
                        elif isinstance(item, h5py.Dataset):
                            # If it's a dataset, print its details
                            out_file.write(f"{item_path} (Dataset)\n")
                            out_file.write(f"  Shape: {item.shape}\n")
                            out_file.write(f"  Dtype: {item.dtype}\n")
                            
                            # For small datasets, print sample values
                            if len(item.shape) == 0 or (len(item.shape) == 1 and item.shape[0] < 10):
                                try:
                                    out_file.write(f"  Values: {item[...]}\n")
                                except:
                                    out_file.write("  Values: [Could not display]\n")
                            elif len(item.shape) > 0 and item.shape[0] > 0:
                                try:
                                    # Print the first value
                                    if item.shape[0] > 0:
                                        if item.dtype.kind in ['S', 'U']:
                                            # Handle string types
                                            out_file.write(f"  First value: {item[0].decode() if item.dtype.kind == 'S' else item[0]}\n")
                                        else:
                                            out_file.write(f"  First value: {item[0]}\n")
                                except Exception as e:
                                    out_file.write(f"  Error accessing first value: {e}\n")
                    except Exception as e:
                        out_file.write(f"Error exploring {item_path}: {e}\n")
            
            # Start exploring from the root groups
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    explore_group(f[key], key)
                else:
                    # Handle top-level datasets
                    out_file.write(f"{key} (Dataset)\n")
                    item = f[key]
                    out_file.write(f"  Shape: {item.shape}\n")
                    out_file.write(f"  Dtype: {item.dtype}\n")
            
            # Special detailed analysis for demos if they exist
            if 'data' in f and isinstance(f['data'], h5py.Group):
                data_group = f['data']
                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                
                out_file.write(f"\nFound {len(demo_keys)} demos in 'data' group\n")
                
                # Sample a few demos for more detailed analysis
                sample_demos = demo_keys[:3] if len(demo_keys) > 3 else demo_keys
                
                for demo_key in sample_demos:
                    out_file.write(f"\nDetailed analysis of {demo_key}:\n")
                    demo_group = data_group[demo_key]
                    
                    for key in demo_group.keys():
                        item = demo_group[key]
                        
                        if isinstance(item, h5py.Dataset):
                            # It's a dataset
                            out_file.write(f"  {key} (Dataset): shape={item.shape}, dtype={item.dtype}\n")
                            
                            # Sample values based on the dataset type
                            try:
                                if key == 'actions' and len(item.shape) > 0:
                                    out_file.write(f"    First action: {item[0]}\n")
                                    out_file.write(f"    Last action: {item[-1]}\n")
                                    out_file.write(f"    Action dimensions: {len(item[0]) if len(item.shape) > 1 else 1}\n")
                                elif key == 'dones' and len(item.shape) > 0:
                                    done_count = np.sum(item[:])
                                    out_file.write(f"    Done count: {done_count} / {item.shape[0]}\n")
                                elif len(item.shape) > 0:
                                    out_file.write(f"    First value shape: {item[0].shape if len(item.shape) > 1 else 'scalar'}\n")
                                    out_file.write(f"    First value: {item[0]}\n")
                            except Exception as e:
                                out_file.write(f"    Error analyzing {key}: {e}\n")
                        else:
                            # It's a group, provide summary
                            out_file.write(f"  {key} (Group): {list(item.keys())}\n")
                            
                            # Examine first level of subgroups
                            for subkey in list(item.keys())[:5]:  # Look at first 5 keys
                                subitem = item[subkey]
                                if isinstance(subitem, h5py.Dataset):
                                    out_file.write(f"    {subkey} (Dataset): shape={subitem.shape}, dtype={subitem.dtype}\n")
                                    if len(subitem.shape) > 0 and subitem.shape[0] > 0:
                                        try:
                                            out_file.write(f"      First value: {subitem[0]}\n")
                                        except Exception as e:
                                            out_file.write(f"      Error accessing value: {e}\n")
                                elif isinstance(subitem, h5py.Group):
                                    out_file.write(f"    {subkey} (Group): {list(subitem.keys())}\n")
            
            # Count structure
            group_count = 0
            dataset_count = 0
            
            def count_items(name, obj):
                nonlocal group_count, dataset_count
                if isinstance(obj, h5py.Group):
                    group_count += 1
                elif isinstance(obj, h5py.Dataset):
                    dataset_count += 1
            
            f.visititems(count_items)
            
            out_file.write(f"\nTotal groups: {group_count}, Total datasets: {dataset_count}\n")
    
    print(f"Exploration completed. Results saved to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--output", type=str, help="Path to save the output (optional)")
    
    args = parser.parse_args()
    
    explore_hdf5_detailed(args.file, args.output) 