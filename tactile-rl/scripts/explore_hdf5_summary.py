import h5py
import numpy as np
import argparse

def explore_hdf5_summary(file_path):
    """Explore the structure of an HDF5 file with summarized output"""
    
    # Store info about dataset structure
    structure = {}
    demo_count = 0
    obs_keys = set()
    
    def collect_info(name, obj):
        nonlocal demo_count, obs_keys
        
        # Track demo count
        if isinstance(obj, h5py.Group) and name.startswith('data/demo_'):
            demo_count += 1
            
        # Track observation types
        if isinstance(obj, h5py.Dataset) and '/obs/' in name:
            parts = name.split('/')
            if len(parts) >= 4:  # data/demo_X/obs/KEY
                obs_keys.add(parts[3])
            
        # Track dataset shapes
        if isinstance(obj, h5py.Dataset):
            # Get group path (parent folders)
            path_parts = name.split('/')
            if len(path_parts) > 1:
                group_path = '/'.join(path_parts[:-1])
                key = path_parts[-1]
            else:
                group_path = '/'
                key = name
                
            if group_path not in structure:
                structure[group_path] = {}
                
            # Store shape and type
            structure[group_path][key] = (obj.shape, obj.dtype)
    
    with h5py.File(file_path, 'r') as f:
        # Collect information
        f.visititems(collect_info)
        
        # Print summary
        print(f"=== Summary for {file_path} ===")
        print(f"Number of demonstrations: {demo_count}")
        print(f"Observation keys: {sorted(list(obs_keys))}")
        
        # Check first demo for structure details
        if 'data/demo_0' in structure:
            print("\nStructure of first demonstration:")
            # Print observation shapes
            if 'data/demo_0/obs' in structure:
                print("\n  Observations:")
                for key, (shape, dtype) in structure['data/demo_0/obs'].items():
                    print(f"    {key}: Shape {shape}, Type {dtype}")
            
            # Print action shape
            if 'actions' in structure.get('data/demo_0', {}):
                shape, dtype = structure['data/demo_0']['actions']
                print(f"\n  Actions: Shape {shape}, Type {dtype}")
                
                # Get sample length from actions shape
                if len(shape) > 0:
                    print(f"  Trajectory length: {shape[0]} timesteps")
            
            # Other important fields
            for field in ['rewards', 'dones', 'states']:
                if field in structure.get('data/demo_0', {}):
                    shape, dtype = structure['data/demo_0'][field]
                    print(f"  {field.capitalize()}: Shape {shape}, Type {dtype}")
        
        # Examine sample of first observation and action if available
        try:
            if demo_count > 0 and len(obs_keys) > 0:
                print("\nFirst observation and action sample:")
                first_obs_key = sorted(list(obs_keys))[0]
                first_obs = f['data/demo_0/obs'][first_obs_key][0]
                first_action = f['data/demo_0/actions'][0]
                
                print(f"  First {first_obs_key}: Shape {first_obs.shape}, Min {np.min(first_obs)}, Max {np.max(first_obs)}")
                print(f"  First action: {first_action}")
        except Exception as e:
            print(f"  Couldn't extract samples: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize HDF5 file structure")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 file")
    args = parser.parse_args()
    
    explore_hdf5_summary(args.dataset)
