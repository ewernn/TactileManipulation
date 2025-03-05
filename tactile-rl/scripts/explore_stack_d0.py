import h5py
import numpy as np
import os

def explore_stack_d0(file_path):
    """
    Focused exploration of the stack_d0.hdf5 dataset.
    """
    print(f"Exploring: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"Root keys: {list(f.keys())}")
        
        if 'data' in f:
            data_group = f['data']
            demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
            print(f"Found {len(demo_keys)} demos in 'data' group")
            
            # Analyze the first demo
            first_demo_key = demo_keys[0]
            print(f"\nExploring demo: {first_demo_key}")
            demo = data_group[first_demo_key]
            print(f"Keys in demo: {list(demo.keys())}")
            
            # Explore each key in the demo
            for key in demo.keys():
                item = demo[key]
                if isinstance(item, h5py.Dataset):
                    print(f"\n{key} (Dataset):")
                    print(f"  Shape: {item.shape}")
                    print(f"  Dtype: {item.dtype}")
                    
                    # Sample values
                    if len(item.shape) > 0 and item.shape[0] > 0:
                        if key == 'actions':
                            print(f"  First action: {item[0]}")
                            print(f"  Last action: {item[-1]}")
                        elif key == 'dones':
                            done_count = np.sum(item[:])
                            print(f"  Done count: {done_count} / {item.shape[0]}")
                        else:
                            print(f"  First value shape: {item[0].shape if len(item.shape) > 1 else 'scalar'}")
                            print(f"  First value: {item[0]}")
                elif isinstance(item, h5py.Group):
                    print(f"\n{key} (Group):")
                    print(f"  Subkeys: {list(item.keys())}")
                    
                    # Look at a few subkeys
                    for subkey in list(item.keys())[:3]:
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(f"  {subkey} (Dataset):")
                            print(f"    Shape: {subitem.shape}")
                            print(f"    Dtype: {subitem.dtype}")
                            if len(subitem.shape) > 0 and subitem.shape[0] > 0:
                                print(f"    First value: {subitem[0]}")
            
            # Examine what we need for simulation
            print("\nIMPORTANT SIMULATION DETAILS:")
            print("-----------------------------")
            
            # Actions
            if 'actions' in demo:
                actions = demo['actions']
                print(f"Actions shape: {actions.shape}")
                print(f"Action dimensionality: {actions.shape[1]}")
                print(f"First action: {actions[0]}")
                print(f"Last action: {actions[-1]}")
            
            # States
            if 'states' in demo:
                states = demo['states']
                print(f"States shape: {states.shape}")
                print(f"First state: {states[0]}")
            
            # Check if obs contains relevant info
            if 'obs' in demo:
                obs = demo['obs']
                if isinstance(obs, h5py.Dataset):
                    print(f"Observations shape: {obs.shape}")
                    print(f"First observation: {obs[0]}")
                elif isinstance(obs, h5py.Group):
                    print(f"Observation keys: {list(obs.keys())}")
                    for key in list(obs.keys())[:3]:
                        print(f"  {key} shape: {obs[key].shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../datasets/core/stack_d0.hdf5", 
                        help="Path to stack_d0.hdf5 file")
    
    args = parser.parse_args()
    
    explore_stack_d0(args.file) 