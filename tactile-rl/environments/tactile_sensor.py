import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class TactileSensor:
    """
    Simulated tactile sensor for robotic gripper.
    Implements a grid of taxels on each finger surface.
    """
    
    def __init__(self, n_taxels_x=10, n_taxels_y=10, finger_pad_size=(0.016, 0.016), noise_level=0.01):
        """
        Initialize a tactile sensor with a grid of taxels on each finger.
        
        Args:
            n_taxels_x: Number of taxels in the x direction
            n_taxels_y: Number of taxels in the y direction
            finger_pad_size: Size of the finger pad in meters (width, height)
            noise_level: Standard deviation of the Gaussian noise to add to the readings
        """
        self.n_taxels_x = n_taxels_x
        self.n_taxels_y = n_taxels_y
        self.finger_pad_size = finger_pad_size
        self.noise_level = noise_level
        
        # For the Panda gripper, the finger pads are defined as 0.016 x 0.016 m boxes
        # (based on the 0.008 half-size in the model)
        self.pad_width = finger_pad_size[0]  # x dimension
        self.pad_height = finger_pad_size[1]  # z dimension
        
        # Calculate taxel positions for both fingers
        self.left_taxel_positions = self._compute_taxel_positions("left")
        self.right_taxel_positions = self._compute_taxel_positions("right")
        
        # Initialize readings for both fingers
        self.left_readings = np.zeros((n_taxels_x, n_taxels_y, 3))  # Force readings (normal, tangential_x, tangential_y)
        self.right_readings = np.zeros((n_taxels_x, n_taxels_y, 3))
    
    def _compute_taxel_positions(self, finger):
        """
        Compute the positions of taxels on a finger pad.
        
        Args:
            finger: "left" or "right"
            
        Returns:
            Array of shape (n_taxels_x, n_taxels_y, 3) with the (x, y, z) coordinates
            of each taxel in the finger's local coordinate frame.
        """
        # Create a grid of taxels
        x_positions = np.linspace(-self.pad_width/2, self.pad_width/2, self.n_taxels_x)
        y_positions = np.linspace(-self.pad_height/2, self.pad_height/2, self.n_taxels_y)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_positions, y_positions)
        
        # Initialize positions array
        positions = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
        
        # Set x and z coordinates based on the meshgrid (y is constant at the surface)
        positions[:, :, 0] = xx  # x coordinate
        
        # Y coordinate is the surface of the pad
        if finger == "left":
            positions[:, :, 1] = -0.004  # y coordinate (pad thickness is 0.008, so half is 0.004)
        else:  # right finger
            positions[:, :, 1] = 0.004
            
        positions[:, :, 2] = yy  # z coordinate
        
        return positions
    
    def process_contacts(self, sim):
        """
        Process contact information from a MuJoCo simulation step.
        
        Args:
            sim: A MuJoCo simulation object after step() has been called
            
        Returns:
            Updated left and right finger tactile readings
        """
        # Reset readings
        self.left_readings = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
        self.right_readings = np.zeros((self.n_taxels_x, self.n_taxels_y, 3))
        
        # Get contact information
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            
            # Check if contact involves the finger pads
            geom1_name = sim.model.geom_id2name(contact.geom1)
            geom2_name = sim.model.geom_id2name(contact.geom2)
            
            # Determine which finger is in contact (if any)
            if geom1_name == "finger1_pad_collision" or geom2_name == "finger1_pad_collision":
                finger = "left"
                geom_id = contact.geom1 if geom1_name == "finger1_pad_collision" else contact.geom2
            elif geom1_name == "finger2_pad_collision" or geom2_name == "finger2_pad_collision":
                finger = "right"
                geom_id = contact.geom1 if geom1_name == "finger2_pad_collision" else contact.geom2
            else:
                continue  # Contact doesn't involve finger pads
            
            # Get contact position in world frame
            contact_pos = np.array(contact.pos)
            
            # Get contact force in world frame
            contact_force = np.zeros(6)
            sim.data.get_contact_force(i, contact_force)
            force = contact_force[:3]  # Just get the force vector
            
            # Transform the contact position to the local coordinate frame of the finger pad
            local_pos = self._world_to_local(contact_pos, sim, finger)
            
            # Transform the contact force to the local coordinate frame
            local_force = self._world_to_local_force(force, sim, finger)
            
            # Distribute the force to the surrounding taxels
            self._distribute_force(local_pos, local_force, finger)
        
        # Add noise to readings
        self._add_noise()
        
        return self.left_readings, self.right_readings
    
    def _world_to_local(self, pos_world, sim, finger):
        """
        Transform a position from world coordinates to the local finger pad coordinates.
        
        Args:
            pos_world: Position in world coordinates
            sim: MuJoCo simulation object
            finger: "left" or "right"
            
        Returns:
            Position in local finger pad coordinates
        """
        # This is a simplified transformation - in a real implementation, we would
        # use the actual transformation matrix from MuJoCo
        
        # Get the body ID for the finger tip
        if finger == "left":
            body_id = sim.model.body_name2id("finger_joint1_tip")
        else:
            body_id = sim.model.body_name2id("finger_joint2_tip")
        
        # Get the position and orientation of the body
        body_pos = sim.data.body_xpos[body_id]
        body_mat = sim.data.body_xmat[body_id].reshape(3, 3)
        
        # Transform from world to body coordinates
        pos_body = body_mat.T @ (pos_world - body_pos)
        
        # Transform from body to pad coordinates
        # According to the XML, the pad is offset from the finger tip
        if finger == "left":
            pad_offset = np.array([0, -0.005, -0.015])
        else:
            pad_offset = np.array([0, 0.005, -0.015])
        
        pos_pad = pos_body - pad_offset
        
        return pos_pad
    
    def _world_to_local_force(self, force_world, sim, finger):
        """
        Transform a force vector from world coordinates to the local finger pad coordinates.
        
        Args:
            force_world: Force vector in world coordinates
            sim: MuJoCo simulation object
            finger: "left" or "right"
            
        Returns:
            Force vector in local finger pad coordinates
        """
        # Get the body ID for the finger tip
        if finger == "left":
            body_id = sim.model.body_name2id("finger_joint1_tip")
        else:
            body_id = sim.model.body_name2id("finger_joint2_tip")
        
        # Get the orientation of the body
        body_mat = sim.data.body_xmat[body_id].reshape(3, 3)
        
        # Transform the force from world to body coordinates
        force_local = body_mat.T @ force_world
        
        return force_local
    
    def _distribute_force(self, contact_pos, contact_force, finger):
        """
        Distribute the contact force to the surrounding taxels.
        
        Args:
            contact_pos: Contact position in local finger pad coordinates
            contact_force: Contact force in local finger pad coordinates
            finger: "left" or "right"
        """
        # Simple force distribution model: distribute the force to the 4 surrounding taxels
        # based on the distance to each taxel
        
        # Find the closest taxels
        if finger == "left":
            positions = self.left_taxel_positions
            readings = self.left_readings
        else:
            positions = self.right_taxel_positions
            readings = self.right_readings
        
        # Calculate distances to all taxels
        distances = np.sqrt(
            (positions[:, :, 0] - contact_pos[0])**2 + 
            (positions[:, :, 2] - contact_pos[2])**2
        )
        
        # Use Gaussian weighting for force distribution
        sigma = 0.005  # Gaussian width parameter (5mm)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights = weights / np.sum(weights)  # Normalize
        
        # Normal force is in the y direction (perpendicular to the pad surface)
        normal_force = np.abs(contact_force[1])  # Take absolute value
        
        # Tangential forces are in the x and z directions
        tangential_x = contact_force[0]
        tangential_z = contact_force[2]
        
        # Distribute forces to all taxels according to weights
        for i in range(self.n_taxels_x):
            for j in range(self.n_taxels_y):
                if finger == "left":
                    self.left_readings[i, j, 0] += weights[i, j] * normal_force
                    self.left_readings[i, j, 1] += weights[i, j] * tangential_x
                    self.left_readings[i, j, 2] += weights[i, j] * tangential_z
                else:
                    self.right_readings[i, j, 0] += weights[i, j] * normal_force
                    self.right_readings[i, j, 1] += weights[i, j] * tangential_x
                    self.right_readings[i, j, 2] += weights[i, j] * tangential_z
    
    def _add_noise(self):
        """Add Gaussian noise to the sensor readings."""
        self.left_readings += np.random.normal(0, self.noise_level, self.left_readings.shape)
        self.right_readings += np.random.normal(0, self.noise_level, self.right_readings.shape)
    
    def visualize_readings(self, readings_type="normal"):
        """
        Visualize the tactile sensor readings.
        
        Args:
            readings_type: Type of readings to visualize
                "normal" - normal force
                "tangential" - tangential force magnitude
                "tangential_x" - tangential force in x direction
                "tangential_y" - tangential force in y direction
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Select the appropriate readings
        if readings_type == "normal":
            left_data = self.left_readings[:, :, 0]
            right_data = self.right_readings[:, :, 0]
            title = "Normal Force"
        elif readings_type == "tangential":
            left_data = np.sqrt(self.left_readings[:, :, 1]**2 + self.left_readings[:, :, 2]**2)
            right_data = np.sqrt(self.right_readings[:, :, 1]**2 + self.right_readings[:, :, 2]**2)
            title = "Tangential Force Magnitude"
        elif readings_type == "tangential_x":
            left_data = self.left_readings[:, :, 1]
            right_data = self.right_readings[:, :, 1]
            title = "Tangential Force (X)"
        elif readings_type == "tangential_y":
            left_data = self.left_readings[:, :, 2]
            right_data = self.right_readings[:, :, 2]
            title = "Tangential Force (Y)"
        
        # Plot
        vmax = max(np.max(left_data), np.max(right_data))
        vmin = 0 if readings_type == "normal" or readings_type == "tangential" else -vmax
        
        im1 = ax1.imshow(left_data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Left Finger {title}")
        ax1.set_xlabel("Taxel X")
        ax1.set_ylabel("Taxel Y")
        
        im2 = ax2.imshow(right_data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Right Finger {title}")
        ax2.set_xlabel("Taxel X")
        ax2.set_ylabel("Taxel Y")
        
        # Add colorbars
        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.colorbar(im2, cax=cax2)
        
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
