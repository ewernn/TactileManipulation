import robosuite
from robosuite.environments.manipulation.stack import Stack
import xml.etree.ElementTree as ET

# Create a stack environment
env = Stack(robots="Panda")
model_xml = env.model.get_xml()

# Write it to a file to inspect
with open("original_stack_environment.xml", "w") as f:
    f.write(model_xml)