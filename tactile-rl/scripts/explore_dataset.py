import h5py
import numpy as np
import matplotlib.pyplot as plt
from robomimic.utils.dataset import SequenceDataset

def explore_dataset(dataset_path):
    # First, open the file to check what keys are available
    with h5py.File(dataset_path, "r") as f:
        print("Available keys in dataset:")
        print(list(f.keys()))
        
        # Get observation keys if they exist
        obs_keys = []
        if "obs" in f:
            print("\nObservation keys:")
            obs_group = f["obs"]
            obs_keys = list(obs_group.keys())
            print(obs_keys)
    
    # Now load dataset with proper keys
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,  # Pass the list of keys instead of None
        dataset_keys=["actions", "rewards", "dones"],  # Use list instead of tuple
        load_next_obs=True,
    )
    
    # Print dataset info
    print(f"\nDataset: {dataset_path}")
    print(f"Number of demonstrations: {dataset.n_demos}")
    print(f"Observation keys: {dataset.obs_keys}")
    
    # Print shape of each observation
    if dataset.n_demos > 0:
        demo_idx = 0
        demo = dataset.get_demo(demo_idx)
        print("\nObservation shapes for demo 0:")
        for key in demo["obs"]:
            print(f"  {key}: {demo['obs'][key].shape}")
        
        # Print action shape
        print(f"Action shape: {demo['actions'].shape}")
        
        # Visualize a trajectory if there are images
        if 'agentview_image' in demo['obs']:
            plt.figure(figsize=(10, 10))
            plt.imshow(demo['obs']['agentview_image'][0])
            plt.title("First frame of demo 0")
            plt.savefig("first_frame.png")
            print("Saved first frame to first_frame.png")
    else:
        print("Dataset has no demonstrations!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()
    
    explore_dataset(args.dataset)
