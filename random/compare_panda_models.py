import os
import difflib

# Path to the standard robosuite model
standard_model_path = "/Users/ewern/opt/miniconda3/envs/rl/lib/python3.9/site-packages/robosuite/models/assets/robots/panda/robot.xml"

# Path to your custom model
your_model_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda.xml"

# Read both files
with open(standard_model_path, 'r') as f:
    standard_lines = f.readlines()
    
with open(your_model_path, 'r') as f:
    custom_lines = f.readlines()

# Compare the files and print differences
diff = difflib.unified_diff(standard_lines, custom_lines, 
                           fromfile='standard', tofile='custom')
print(''.join(diff))