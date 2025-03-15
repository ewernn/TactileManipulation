def generate_tactile_grid_xml(rows=3, cols=4, pad_width=0.01, pad_height=0.03, 
                             left_y_offset=0.006, right_y_offset=-0.006, z_base=0.01):
    """
    Generate XML for a tactile grid sensor system compatible with MuJoCo schema.
    
    Returns:
    - sites_xml: XML for site definitions (to add to finger bodies)
    - sensors_xml: XML for sensor definitions (to add to sensor section)
    """
    sites_left = []
    sites_right = []
    sensors = []
    
    # Calculate spacing
    x_spacing = pad_width / (cols - 1) if cols > 1 else 0
    z_spacing = pad_height / (rows - 1) if rows > 1 else 0
    
    # Center the grid on the finger
    x_start = -pad_width/2
    z_start = z_base
    
    # Generate sites and touch sensors
    for i in range(rows):
        for j in range(cols):
            x = x_start + j * x_spacing
            z = z_start + i * z_spacing
            
            # Left finger taxel
            left_name = f"left_taxel_{i}_{j}"
            sites_left.append(f'<site name="{left_name}" pos="{x:.6f} {left_y_offset} {z:.6f}" size="0.001" rgba="0 0 1 0.5" type="sphere" />')
            sensors.append(f'<touch name="{left_name}_sensor" site="{left_name}" />')
            
            # Right finger taxel
            right_name = f"right_taxel_{i}_{j}"
            sites_right.append(f'<site name="{right_name}" pos="{x:.6f} {right_y_offset} {z:.6f}" size="0.001" rgba="1 0 0 0.5" type="sphere" />')
            sensors.append(f'<touch name="{right_name}_sensor" site="{right_name}" />')
    
    return {
        "left_finger_sites": "\n        ".join(sites_left),
        "right_finger_sites": "\n        ".join(sites_right),
        "sensors": "\n  ".join(sensors)
    }

# Example usage:
xml_parts = generate_tactile_grid_xml()
print("Add to left_finger body:")
print(xml_parts["left_finger_sites"])
print("\nAdd to right_finger body:")
print(xml_parts["right_finger_sites"])
print("\nAdd to sensor section:")
print("<sensor>")
print(xml_parts["sensors"])
print("</sensor>")
