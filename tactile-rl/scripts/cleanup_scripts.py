#!/usr/bin/env python3
"""
Script to clean up the scripts directory, keeping only essential files
"""

import os
import shutil
from datetime import datetime

# Scripts to KEEP - essential for the tactile manipulation pipeline
KEEP_SCRIPTS = {
    # Core training
    "train_policies.py",
    "train_simple.py",  # Alternative training approach
    
    # Data collection
    "collect_tactile_demos.py",
    "create_expert_demos.py",
    
    # Data processing
    "augment_dataset.py",
    
    # Visualization and analysis
    "visualize_grasp.py",
    "replay_demo.py",
    "generate_video.py",
    "explore_dataset.py",
    
    # Recent diagnostic tools (useful for development)
    "diagnose_expert_policy.py",
    "tune_expert_systematically.py",
    
    # This cleanup script
    "cleanup_scripts.py"
}

def cleanup_scripts(dry_run=True):
    """Clean up scripts directory, moving non-essential scripts to archive"""
    
    # Get current directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(scripts_dir, f"_archive_{timestamp}")
    
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)
    
    # Get all Python scripts
    all_scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
    
    # Categorize scripts
    to_keep = []
    to_archive = []
    
    for script in all_scripts:
        if script in KEEP_SCRIPTS:
            to_keep.append(script)
        else:
            to_archive.append(script)
    
    # Print summary
    print("📊 Script Cleanup Summary")
    print("=" * 60)
    print(f"Total scripts: {len(all_scripts)}")
    print(f"Scripts to keep: {len(to_keep)}")
    print(f"Scripts to archive: {len(to_archive)}")
    
    print("\n✅ Scripts to KEEP:")
    for script in sorted(to_keep):
        print(f"   - {script}")
    
    print(f"\n📦 Scripts to ARCHIVE ({len(to_archive)} files):")
    # Show first 10 as examples
    for i, script in enumerate(sorted(to_archive)[:10]):
        print(f"   - {script}")
    if len(to_archive) > 10:
        print(f"   ... and {len(to_archive) - 10} more")
    
    if dry_run:
        print("\n⚠️  DRY RUN - No files moved")
        print("Run with --execute to actually move files")
    else:
        # Move files
        print(f"\n🚀 Moving {len(to_archive)} files to {archive_dir}")
        moved = 0
        for script in to_archive:
            src = os.path.join(scripts_dir, script)
            dst = os.path.join(archive_dir, script)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                print(f"   ❌ Error moving {script}: {e}")
        
        print(f"\n✅ Successfully archived {moved} files")
        print(f"📁 Archive location: {archive_dir}")
    
    return len(to_keep), len(to_archive)

if __name__ == "__main__":
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    print("🧹 Cleaning up scripts directory...")
    print("This will keep only essential scripts for the tactile manipulation pipeline")
    print()
    
    kept, archived = cleanup_scripts(dry_run=dry_run)
    
    if dry_run:
        print("\n💡 To execute cleanup, run:")
        print("   python cleanup_scripts.py --execute")