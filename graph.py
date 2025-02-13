import matplotlib.pyplot as plt
import numpy as np

# Define the grid sizes tested in the benchmarks
grid_sizes = [10000, 25000, 50000, 100000]

# Extracted execution times from performance_results.txt
sequential_times = [2.24770, 5.59528, 10.79711, 21.23850]
parallel_times = [2.24670, 5.62904, 10.70089, 21.39217]
gpu_times = [0.35461, 0.03248, 0.03346, 0.03615]

# Compute speedup
cpu_speedup = np.array(sequential_times) / np.array(parallel_times)
gpu_speedup = np.array(sequential_times) / np.array(gpu_times)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set up the first y-axis for execution time
ax1.set_xlabel("Grid Size", fontsize=14, fontweight="bold")
ax1.set_ylabel("Execution Time (seconds)", fontsize=14, color="black")
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_xticks(grid_sizes)
ax1.set_xticklabels(grid_sizes, fontsize=12)

# Plot execution times with clearer styles
ax1.plot(grid_sizes, sequential_times, label="Sequential Execution", marker='o', linestyle='--', color='red', linewidth=2, markersize=8)
ax1.plot(grid_sizes, parallel_times, label="Parallel Execution (OpenMP)", marker='s', linestyle='-', color='blue', linewidth=2, markersize=8)
ax1.plot(grid_sizes, gpu_times, label="GPU Execution (CUDA)", marker='^', linestyle='-', color='green', linewidth=2, markersize=8)

# Create a secondary y-axis for speedup values
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup", fontsize=14, color="purple")
ax2.tick_params(axis='y', labelcolor="purple")

# Plot speedup with a different style
ax2.plot(grid_sizes, cpu_speedup, label="CPU Speedup", marker='d', linestyle=':', color='purple', linewidth=2, markersize=8)
ax2.plot(grid_sizes, gpu_speedup, label="GPU Speedup", marker='x', linestyle=':', color='orange', linewidth=2, markersize=8)

# Set grid for better visibility
ax1.grid(True, linestyle="--", alpha=0.6)

# Add titles and legends with improved placement
fig.suptitle("Performance Comparison of SWE Solver", fontsize=16, fontweight="bold")

ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)

# Save the plot as an image file
plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
