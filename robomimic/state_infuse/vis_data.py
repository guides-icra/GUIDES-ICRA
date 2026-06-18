import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm
import re

plt.rcParams['font.family'] = 'Arial'


def split_camel_case(text):
    # Use regular expression to split the camel case string
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text).split()


def wrap_text(text, width=10):
    """Wrap text to a specific character width."""
    text = text.replace('PnP', 'PNP')  # Replace underscores with spaces
    words = split_camel_case(text)
    wrapped = ['']

    for i, w in enumerate(words):
        if len(wrapped[-1]) + len(w) < width:
            wrapped[-1] += w
        else:
            wrapped.append(w)

    return '\n'.join(wrapped).replace('PNP', 'PnP')  # Replace spaces with underscores


# Load the updated CSV file with the new "Horizon" column
file_path = '/Users/gaominquan/Documents/Hopkins-Learning/Research/2024-summer/plot-data.csv'  # Replace with the actual path
data = pd.read_csv(file_path)

# Sort data by "Unique Actions GPT4o Provide" in descending order
data_sorted = data.sort_values(by="Unique Actions GPT4o Provide", ascending=False).reset_index(drop=True)

# Apply square root transformation to the "Unique Actions GPT4o Provide" for plotting
data_sorted["Sqrt Unique Actions"] = np.sqrt(data_sorted["Unique Actions GPT4o Provide"])

# Apply square root transformation to the "relative improvement rate" for the second subplot
data_sorted["Sqrt Relative Improvement Rate"] = np.sqrt(data_sorted["relative improvement rate"])

# Generate a color map for each task based on the task order
num_tasks = len(data_sorted)
colors = cm.tab20(np.linspace(0, 1, num_tasks))  # Generate enough colors for each task
color_mapping = {task: colors[i] for i, task in enumerate(data_sorted["Tasks"])}  # Map each task to a unique color

# Calculate the group averages (for example, divide data into 3 groups of 8 tasks each)
group_sizes = [8, 8, 8]
start = 0
group_averages = []

for size in group_sizes:
    end = start + size
    # Calculate the average "relative improvement rate" for each group
    avg_relative_improvement = data_sorted["relative improvement rate"][start:end].mean()
    group_averages.append(avg_relative_improvement)
    start = end  # Move to the next group

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # 16:9 aspect ratio

fig.subplots_adjust(bottom=0.3, wspace=0.4)  # Increase bottom margin and add more space between subplots

# First subplot for the Square Root of Unique Actions
bar_colors = [color_mapping[task] for task in data_sorted["Tasks"]]
bars = ax1.bar(range(num_tasks), data_sorted["Sqrt Unique Actions"], color=bar_colors)
ax1.set_xlabel("Tasks", fontsize=14, fontfamily='Arial')  # Adjust x-axis label font
ax1.set_ylabel("square root the unique action number", fontsize=14, fontfamily='Arial')  # Adjust y-axis label font
ax1.set_title("$Diff_r$ with respect to the unique actions GPT4o Provides", fontsize=14, fontfamily='Arial')  # Adjust title font
ax1.set_xticks([])  # Hide x-axis labels
ax1.tick_params(axis='both', which='major', labelsize=12)  # Set tick label size

# Add dotted boxes around custom groups with sizes 8-8-8 in the first subplot
start = 0
for size in group_sizes:
    end = start + size
    box = patches.Rectangle(
        (start - 0.5, 0),                # (x, y) starting point
        size,                            # width of the box (number of bars in the group)
        max(data_sorted["Sqrt Unique Actions"]) * 1.1,  # height of the box
        linewidth=1.5,
        edgecolor='gray',
        linestyle='--',
        facecolor='none'
    )
    ax1.add_patch(box)
    start = end

# Add a secondary y-axis for the average relative improvement rates in the first plot
ax1_right = ax1.twinx()
ax1_right.set_ylabel("Mean of $Diff_r$", fontsize=14, fontfamily='Arial')
ax1_right.set_ylim(0, 100)
ax1_right.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size

# Plot the group average relative improvement rates as dots and connect with step lines
group_midpoints = [sum(group_sizes[:i]) + size / 2 for i, size in enumerate(group_sizes)]
ax1_right.plot(group_midpoints, group_averages, 'o', color='red', markersize=8)
for i in range(len(group_midpoints) - 1):
    ax1_right.plot([group_midpoints[i], group_midpoints[i+1]], [group_averages[i], group_averages[i]], linestyle='--', color='red')
    ax1_right.plot([group_midpoints[i+1], group_midpoints[i+1]], [group_averages[i], group_averages[i+1]], linestyle='--', color='red')

# Second subplot for Horizon vs. Square Root of Relative Improvement Rate with square markers
scatter_colors = [color_mapping[task] for task in data_sorted["Tasks"]]
ax2.scatter(data_sorted["Horizon"], data_sorted["Sqrt Relative Improvement Rate"], color=scatter_colors, s=100, marker='s')  # Square marker 's'
ax2.set_xlabel("Horizon", fontsize=14, fontfamily='Arial')  # Adjust x-axis label font
ax2.set_ylabel("$\sqrt{Diff_r}$", fontsize=14, fontfamily='Arial')  # Adjust y-axis label font
ax2.set_title("$Diff_r$ vs Horizon Length for Each Task", fontsize=14, fontfamily='Arial')  # Adjust title font
ax2.tick_params(axis='both', which='major', labelsize=12)  # Set tick label size

# Remove this part to eliminate the fitting blue dotted line
# Create a polynomial fit for the trend line in the second subplot
# coefficients = np.polyfit(data_sorted["Horizon"], data_sorted["Sqrt Relative Improvement Rate"], 2)
# poly_trend = np.poly1d(coefficients)
# x_smooth = np.linspace(data_sorted["Horizon"].min(), data_sorted["Horizon"].max(), 200)
# y_smooth = poly_trend(x_smooth)
# ax2.plot(x_smooth, y_smooth, linestyle='--', color='blue', linewidth=1)

# Add a custom legend below both plots
legend_ax = fig.add_axes([0.05, 0.00, 0.8, 0.17])  # x, y, width, height for custom legend area
legend_ax.axis("off")  # Hide the axis

for i, (task, color) in enumerate(color_mapping.items()):
    if i % 8 == 1:
        f = 0.09
    elif i % 8 == 2:
        f = 0.095
    elif i % 8 == 3:
        f = 0.095
    elif i % 8 == 4:
        f = 0.095
    elif i % 8 == 5:
        f = 0.095
    elif i % 8 == 6:
        f = 0.1
    elif i % 8 == 7:
        f = 0.1
    else:
        f = 0.11

    x_pos = 0.05 + (i % 8) * f  # Adjust horizontal spacing for each task
    y_pos = 0.9 - (i // 8) * 0.35  # Adjust vertical position for rows

    # Draw colored square and task name
    legend_ax.plot(x_pos, y_pos, marker='s', color=color, markersize=12)  # Color square
    legend_ax.text(x_pos + 0.02, y_pos, task, color="black", ha="left", va="center", fontsize=10)  # Task name in black

# Save the plot to a file
plt.savefig('plot.png')
