import mujoco
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

# Step simulation
for _ in range(100):
    mujoco.mj_step(model, data)

# If this runs without errors, MuJoCo is working
print("MuJoCo is working!")
