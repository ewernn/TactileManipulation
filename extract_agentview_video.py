import argparse
import h5py
import os
import numpy as np
import imageio
from tqdm import tqdm

def extract_agentview_video(dataset_path, demo_idx=0, output_dir=None, fps=30):
    """
    Extract agentview_image frames from an HDF5 dataset and save as a video.
    
    Args:
        dataset_path: Path to the HDF5 dataset
        demo_idx: Index of the demonstration to extract
        output_dir: Directory to save the video
        fps: Frames per second for the output video
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine output path
    filename = os.path.basename(dataset_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{filename}_demo{demo_idx}_agentview.mp4") if output_dir else f"{filename}_demo{demo_idx}_agentview.mp4"
    
    # Load the dataset
    with h5py.File(dataset_path, 'r') as f:
        # Check if the dataset has the expected structure
        if 'data' not in f:
            raise ValueError(f"Dataset does not contain 'data' group")
        
        data_group = f['data']
        demo_key = f"demo_{demo_idx}"
        
        if demo_key not in data_group:
            available_demos = [k for k in data_group.keys() if k.startswith('demo_')]
            print(f"Available demos: {sorted(available_demos)[:10]}...")
            raise ValueError(f"Demo {demo_idx} not found in dataset")
        
        demo_group = data_group[demo_key]
        
        # Check if agentview_image exists
        if 'obs' not in demo_group or 'agentview_image' not in demo_group['obs']:
            raise ValueError(f"Demo {demo_idx} does not contain agentview_image")
        
        # Get agentview_image data
        agentview_images = demo_group['obs']['agentview_image'][:]
        
        print(f"Extracted {len(agentview_images)} frames with shape {agentview_images.shape}")
        
        # Create video writer
        with imageio.get_writer(output_path, fps=fps) as writer:
            # Add each frame to the video
            for i, frame in enumerate(tqdm(agentview_images, desc="Writing frames")):
                writer.append_data(frame)
        
        print(f"Saved video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0, help="Index of the demonstration to extract")
    parser.add_argument("--output-dir", type=str, help="Directory to save the video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    
    args = parser.parse_args()
    
    extract_agentview_video(
        dataset_path=args.dataset,
        demo_idx=args.demo,
        output_dir=args.output_dir,
        fps=args.fps
    ) 