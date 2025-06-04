#!/bin/bash
# Cleanup script for TactileManipulation repository

echo "🧹 Cleaning up TactileManipulation repository..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
find . -name "*.pyd" -delete 2>/dev/null

# Remove .DS_Store files
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null

# Remove generated videos from scripts
echo "Removing generated videos..."
rm -f tactile-rl/scripts/*.mp4

# Remove generated images from scripts
echo "Removing generated images..."
rm -f tactile-rl/scripts/grasp_test_*.png
rm -f tactile-rl/scripts/scene_*.png
rm -f tactile-rl/scripts/debug_*.png
rm -f tactile-rl/scripts/block_configurations.png
rm -f tactile-rl/scripts/robot_orientation_comparison.png

# Remove diagnostic directories
echo "Removing diagnostic directories..."
rm -rf tactile-rl/scripts/expert_policy_diagnostics/
rm -rf tactile-rl/scripts/expert_tuning_*/

# Remove egg-info
echo "Removing egg-info directories..."
rm -rf mimicgen.egg-info/

# Git remove cached files that should be ignored
echo "Removing cached files from git..."
git rm -r --cached **/__pycache__/ 2>/dev/null || true
git rm --cached **/*.pyc 2>/dev/null || true
git rm --cached **/.DS_Store 2>/dev/null || true
git rm -r --cached mimicgen.egg-info/ 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Review the changes with: git status"
echo "2. Stage the cleanup: git add -A"
echo "3. Commit: git commit -m 'Clean up repository and update .gitignore'"