import robosuite
import os

# Get the robosuite installation path
robosuite_path = os.path.dirname(robosuite.__file__)

# Path to the Panda model
panda_model_path = os.path.join(robosuite_path, "models/assets/robots/panda")

# Print the full path
print(f"Panda model directory: {panda_model_path}")

# List files in the directory
print("Files in Panda model directory:")
print(os.listdir(panda_model_path))

# Specific path to the main Panda XML file
panda_xml_path = os.path.join(panda_model_path, "robot.xml")
print(f"Main Panda XML file: {panda_xml_path}")