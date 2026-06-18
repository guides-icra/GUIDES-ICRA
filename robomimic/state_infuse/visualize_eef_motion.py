import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Load your N x 13 array
# Format: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, delta_gripper, base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z]
data = np.load("")  # Replace with actual file path

# Compute cumulative sum for gripper motion (End-Effector)
gripper_positions = np.cumsum(data[:, :3], axis=0)  # XYZ position changes
rotations = np.cumsum(data[:, 3:6], axis=0)  # RPY (Not used in visualization)
gripper_state = np.cumsum(data[:, 6], axis=0)  # Cumulative gripper state

# Extract robot base positions & quaternions
base_positions = data[:, 7:10]  # Robot base XYZ
quaternions = data[:, 10:14]  # Robot base orientation (w, x, y, z)

# Transform gripper positions to align with the base coordinate system
transformed_positions = np.zeros_like(gripper_positions)

for i in range(len(gripper_positions)):
    # Get the rotation matrix from quaternion
    r = R.from_quat(quaternions[i])  # Quaternion format: [x, y, z, w]

    # Transform gripper position to align with the base frame
    transformed_positions[i] = r.apply(gripper_positions[i]) + base_positions[i]

# Normalize gripper size for better visualization
gripper_min_size = 0.01  # Smallest gripper span
gripper_max_size = 0.15  # Largest gripper span
gripper_size_range = gripper_max_size - gripper_min_size
gripper_span_size = gripper_min_size + (gripper_state - np.min(gripper_state)) / (
            np.max(gripper_state) - np.min(gripper_state)) * gripper_size_range

# Initialize 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("End-Effector Trajectory with Dynamic Gripper Span")

# Plot settings
line, = ax.plot([], [], [], 'b-', lw=2)  # Path line
point, = ax.plot([], [], [], 'ro', markersize=8)  # Current position
gripper_span, = ax.plot([], [], [], 'k-', lw=3)  # Gripper span

# Set axis limits dynamically
ax.set_xlim(np.min(transformed_positions[:, 0]), np.max(transformed_positions[:, 0]))
ax.set_ylim(np.min(transformed_positions[:, 1]), np.max(transformed_positions[:, 1]))
ax.set_zlim(np.min(transformed_positions[:, 2]), np.max(transformed_positions[:, 2]))


# Update function for animation
def update(frame):
    # Plot the trajectory line
    line.set_data(transformed_positions[:frame + 1, 0], transformed_positions[:frame + 1, 1])
    line.set_3d_properties(transformed_positions[:frame + 1, 2])

    # Plot the gripper's current position
    point.set_data([transformed_positions[frame, 0]], [transformed_positions[frame, 1]])
    point.set_3d_properties([transformed_positions[frame, 2]])

    # Compute the gripper span endpoints (left and right)
    gripper_direction = np.array([1, 0, 0])  # Default gripper direction (adjust if needed)
    r = R.from_quat(quaternions[frame])  # Get rotation matrix from quaternion
    gripper_direction = r.apply(gripper_direction)  # Rotate the direction to match the robot frame

    # Compute the two span endpoints
    span_half = gripper_span_size[frame] / 2
    span_left = transformed_positions[frame] - span_half * gripper_direction
    span_right = transformed_positions[frame] + span_half * gripper_direction

    # Update the gripper span visualization
    gripper_span.set_data([span_left[0], span_right[0]], [span_left[1], span_right[1]])
    gripper_span.set_3d_properties([span_left[2], span_right[2]])

    return line, point, gripper_span


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(transformed_positions), interval=50, blit=False)

# Show animation
plt.show()

# === SAVE ANIMATION TO A VIDEO FILE ===
save_as_video = True  # Set to False if you don't need to save

if save_as_video:
    # Save as MP4 (Requires FFmpeg)
    Writer = animation.FFMpegWriter
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("robot_animation.mp4", writer=writer)
    print("Video saved as robot_animation.mp4")

    # Save as GIF (Alternative)
    ani.save("robot_animation.gif", writer=animation.PillowWriter(fps=20))
    print("GIF saved as robot_animation.gif")
