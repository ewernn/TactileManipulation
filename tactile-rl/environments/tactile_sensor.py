import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mujoco

class TactileSensor:
    """
    Tactile sensor implementation for a two-finger gripper.
    Simulates a grid of taxels that can detect normal and tangential forces.
    """
    
    def __init__(self, model=None, data=None, n_taxels_x=10, n_taxels_y=10, 
                 finger_pad_size=(0.016, 0.016), noise_level=0.01):
        """
        Initialize the tactile sensor.
        
        Args:
            model: MuJoCo model (optional)
            data: MuJoCo data (optional)
            n_taxels_x: Number of taxels in the x direction
            n_taxels_y: Number of taxels in the y direction
            finger_pad_size: Size of the finger pad in meters (width, height)
            noise_level: Level of noise to add to the readings (fraction of reading)
        """
        self.n_taxels_x = n_taxels_x
        self.n_taxels_y = n_taxels_y
        self.pad_width, self.pad_height = finger_pad_size
        self.noise_level = noise_level
        
        # Finger pad center offsets from finger body center
        # These depend on the specific robot model being used
        self.finger_pad_offsets = {
            "left": np.array([0.0, 0.004, 0.0]),   # Offset from left finger body to pad center
            "right": np.array([0.0, -0.004, 0.0])  # Offset from right finger body to pad center
        }
        
        # Store model and data for later use
        self.model = model
        self.data = data
        
        # Find the finger body IDs if model is provided
        if model is not None:
            self.finger_ids = self._find_finger_ids(model)
        else:
            self.finger_ids = {"left": None, "right": None}
        
        # Initialize taxel positions and readings
        # Each taxel reading has 3 components: [normal_force, tangential_force_x, tangential_force_z]
        self.left_taxel_positions = self._compute_taxel_positions("left")
        self.right_taxel_positions = self._compute_taxel_positions("right")
        
        # Initialize empty readings arrays
        self.left_readings = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
        self.right_readings = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
    
    def _find_finger_ids(self, model):
        """Find the body IDs for the left and right fingers in the MuJoCo model."""
        finger_ids = {"left": None, "right": None}
        
        # Search for finger bodies in the model
        for i in range(model.nbody):
            body_name = model.body(i).name
            
            if "finger" in body_name.lower() or "pad" in body_name.lower():
                if "left" in body_name.lower() or "finger1" in body_name.lower() or "panda_leftfinger" in body_name.lower():
                    finger_ids["left"] = i
                elif "right" in body_name.lower() or "finger2" in body_name.lower() or "panda_rightfinger" in body_name.lower():
                    finger_ids["right"] = i
        
        print(f"Found finger IDs: {finger_ids}")
        return finger_ids
    
    def _compute_taxel_positions(self, finger):
        """
        Compute the positions of taxels in the local coordinate frame of the finger.
        
        Args:
            finger: "left" or "right" to specify which finger
            
        Returns:
            Array of shape (n_taxels_x, n_taxels_y, 3) containing the 3D positions of each taxel
        """
        # Create a grid of taxel positions
        x_positions = np.linspace(-self.pad_width/2, self.pad_width/2, self.n_taxels_x)
        z_positions = np.linspace(-self.pad_height/2, self.pad_height/2, self.n_taxels_y)
        
        # Create a meshgrid of positions
        x_grid, z_grid = np.meshgrid(x_positions, z_positions)
        
        # Y position is fixed (the surface of the pad)
        y_offset = self.finger_pad_offsets[finger][1]
        
        # Create the array of taxel positions
        taxel_positions = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
        taxel_positions[:, :, 0] = x_grid.T  # x position
        taxel_positions[:, :, 1] = y_offset  # y position is fixed at the finger pad surface
        taxel_positions[:, :, 2] = z_grid.T  # z position
        
        return taxel_positions
    
    def process_contacts(self, model, data):
        """
        Process contacts from MuJoCo simulation data.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            
        Returns:
            List of dictionaries containing contact information
        """
        # Reset readings
        self.left_readings.fill(0.0)
        self.right_readings.fill(0.0)
        
        contact_info = []
        
        # Process contacts if we have finger IDs
        if self.finger_ids["left"] is None or self.finger_ids["right"] is None:
            print("Warning: Finger IDs not found. Cannot process contacts.")
            return contact_info
        
        # Iterate through contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Skip inactive contacts
            if model.geom(contact.geom1).contype == 0 or model.geom(contact.geom2).contype == 0:
                continue
            
            # Get contact force in world frame
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force)
            force_norm = np.linalg.norm(force[:3])
            
            # Skip low forces
            if force_norm < 0.01:
                continue
            
            # Check which bodies are involved in the contact
            body1 = model.geom(contact.geom1).bodyid
            body2 = model.geom(contact.geom2).bodyid
            
            # Find if contact involves left or right finger
            finger = None
            if body1 == self.finger_ids["left"] or body2 == self.finger_ids["left"]:
                finger = "left"
            elif body1 == self.finger_ids["right"] or body2 == self.finger_ids["right"]:
                finger = "right"
            
            if finger is None:
                continue
            
            # Get contact position in world frame
            pos_world = contact.pos.copy()
            
            # Get contact force in world frame
            force_world = force[:3].copy()
            
            # Transform to local frame
            pos_local = self._world_to_local(pos_world, model, data, finger)
            force_local = self._world_to_local_force(force_world, model, data, finger)
            
            # Distribute force to nearby taxels
            self._distribute_force(pos_local, force_local, finger)
            
            # Store contact information
            contact_info.append({
                "finger": finger,
                "pos": pos_world,
                "pos_local": pos_local,
                "force": force_world,
                "force_local": force_local,
                "force_norm": force_norm,
                "body_name": model.body(self.finger_ids[finger]).name
            })
        
        # Add noise to readings
        self._add_noise()
        
        return contact_info
    
    def _world_to_local(self, pos_world, model, data, finger):
        """
        Transform a position from world frame to finger's local frame.
        
        Args:
            pos_world: Position in world frame (x,y,z)
            model: MuJoCo model
            data: MuJoCo data
            finger: "left" or "right" to specify which finger
            
        Returns:
            Position in finger's local frame
        """
        # Get the finger body ID
        finger_id = self.finger_ids[finger]
        
        # Get the body's position and orientation
        body_pos = data.xpos[finger_id]
        body_mat = data.xmat[finger_id].reshape(3, 3)
        
        # Transform position to local frame
        pos_local = np.matmul(body_mat.T, pos_world - body_pos)
        
        # Adjust for the finger pad offset
        pos_local = pos_local - self.finger_pad_offsets[finger]
        
        return pos_local
    
    def _world_to_local_force(self, force_world, model, data, finger):
        """
        Transform a force vector from world frame to finger's local frame.
        
        Args:
            force_world: Force in world frame (x,y,z)
            model: MuJoCo model
            data: MuJoCo data
            finger: "left" or "right" to specify which finger
            
        Returns:
            Force in finger's local frame
        """
        # Get the finger body ID
        finger_id = self.finger_ids[finger]
        
        # Get the body's orientation
        body_mat = data.xmat[finger_id].reshape(3, 3)
        
        # Transform force to local frame (only rotation, no translation)
        force_local = np.matmul(body_mat.T, force_world)
        
        return force_local
    
    def _distribute_force(self, contact_pos, contact_force, finger):
        """
        Distribute a contact force to nearby taxels.
        
        Args:
            contact_pos: Contact position in local frame
            contact_force: Contact force in local frame
            finger: "left" or "right" to specify which finger
        """
        # Get the appropriate array of taxel positions and readings
        if finger == "left":
            taxel_positions = self.left_taxel_positions
            readings = self.left_readings
        else:
            taxel_positions = self.right_taxel_positions
            readings = self.right_readings
        
        # Calculate distance from contact point to each taxel
        distances = np.zeros((self.n_taxels_x, self.n_taxels_y))
        for i in range(self.n_taxels_x):
            for j in range(self.n_taxels_y):
                taxel_pos = taxel_positions[i, j]
                # Only consider distance in x-z plane (along the pad surface)
                dx = taxel_pos[0] - contact_pos[0]
                dz = taxel_pos[2] - contact_pos[2]
                distances[i, j] = np.sqrt(dx**2 + dz**2)
        
        # Force distribution function (Gaussian)
        sigma = 0.003  # Standard deviation in meters
        distribution = np.exp(-0.5 * (distances / sigma)**2)
        
        # Normalize distribution
        if np.sum(distribution) > 0:
            distribution = distribution / np.sum(distribution)
        
        # Distribute forces to taxels
        for i in range(self.n_taxels_x):
            for j in range(self.n_taxels_y):
                # Normal force (y-component in local frame)
                readings[i, j, 0] += distribution[i, j] * abs(contact_force[1])
                
                # Tangential forces (x and z components in local frame)
                readings[i, j, 1] += distribution[i, j] * contact_force[0]
                readings[i, j, 2] += distribution[i, j] * contact_force[2]
    
    def _add_noise(self):
        """Add Gaussian noise to the readings."""
        # Add noise to left readings
        noise = np.random.normal(0, self.noise_level, self.left_readings.shape)
        self.left_readings += noise * self.left_readings
        
        # Add noise to right readings
        noise = np.random.normal(0, self.noise_level, self.right_readings.shape)
        self.right_readings += noise * self.right_readings
    
    def get_readings(self, model=None, data=None):
        """
        Get the current tactile readings.
        
        Args:
            model: MuJoCo model (optional, will use stored model if not provided)
            data: MuJoCo data (optional, will use stored data if not provided)
            
        Returns:
            Tuple of (left_readings, right_readings)
        """
        if model is not None and data is not None:
            self.process_contacts(model, data)
        
        return self.left_readings, self.right_readings
    
    def visualize_reading(self, reading, ax=None, title=None):
        """
        Visualize a single tactile reading (either left or right).
        
        Args:
            reading: The tactile reading to visualize (n_taxels_x, n_taxels_y, 3)
            ax: Matplotlib axis to plot on (optional)
            title: Title for the plot (optional)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Extract normal forces (component 0)
        normal_forces = reading[:, :, 0]
        
        # Create colormap for visualization
        cmap = plt.cm.plasma
        vmax = np.max(normal_forces) if np.max(normal_forces) > 0 else 1.0
        
        # Plot the normal forces as a heatmap
        im = ax.imshow(normal_forces.T, cmap=cmap, origin='lower', vmax=vmax)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Normal Force (N)')
        
        # Draw tangential forces as arrows
        if normal_forces.max() > 0.1:  # Only draw arrows if there's significant contact
            for i in range(self.n_taxels_x):
                for j in range(self.n_taxels_y):
                    if normal_forces[i, j] > 0.1 * normal_forces.max():
                        # Scale arrow by tangential force
                        tx = reading[i, j, 1]
                        tz = reading[i, j, 2]
                        
                        # Normalize and scale
                        tangential_mag = np.sqrt(tx**2 + tz**2)
                        if tangential_mag > 0.01:
                            scale = min(0.4, tangential_mag / normal_forces.max())
                            ax.arrow(i, j, scale * tx / tangential_mag, scale * tz / tangential_mag,
                                    head_width=0.1, head_length=0.2, fc='white', ec='white')
        
        # Set labels and title
        ax.set_xlabel('X Taxel Index')
        ax.set_ylabel('Z Taxel Index')
        if title:
            ax.set_title(title)
        
        return ax
    
    def visualize_readings(self, readings_type="normal"):
        """
        Visualize tactile readings for both fingers.
        
        Args:
            readings_type: Type of readings to visualize ("normal", "tangential_x", "tangential_z")
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Select the component to visualize
        component = 0  # Normal force by default
        if readings_type == "tangential_x":
            component = 1
        elif readings_type == "tangential_z":
            component = 2
        
        # Left finger
        left_data = self.left_readings[:, :, component]
        im_left = axes[0].imshow(left_data.T, cmap=plt.cm.plasma, origin='lower')
        axes[0].set_title("Left Finger")
        axes[0].set_xlabel("X Taxel Index")
        axes[0].set_ylabel("Z Taxel Index")
        plt.colorbar(im_left, ax=axes[0])
        
        # Right finger
        right_data = self.right_readings[:, :, component]
        im_right = axes[1].imshow(right_data.T, cmap=plt.cm.plasma, origin='lower')
        axes[1].set_title("Right Finger")
        axes[1].set_xlabel("X Taxel Index")
        plt.colorbar(im_right, ax=axes[1])
        
        # Add title
        component_names = ["Normal Force", "Tangential Force X", "Tangential Force Z"]
        plt.suptitle(f"Tactile Readings: {component_names[component]}")
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Create a tactile sensor
    sensor = TactileSensor()
    
    # Set up some test data
    sensor.left_readings[1, 2, 0] = 1.0  # Normal force
    sensor.left_readings[1, 2, 1] = 0.5  # Tangential force in x direction
    sensor.left_readings[1, 2, 2] = 0.3  # Tangential force in z direction
    
    sensor.right_readings[0, 1, 0] = 0.7  # Normal force
    sensor.right_readings[0, 1, 1] = 0.2  # Tangential force in x direction
    sensor.right_readings[0, 1, 2] = 0.1  # Tangential force in z direction
    
    # Visualize
    fig = sensor.visualize_readings()
    plt.savefig("tactile_test.png")
    plt.show()
