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

# Create the figure
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot execution times
ax1.set_xlabel("Grid Size")
ax1.set_ylabel("Execution Time (seconds)", color='black')
ax1.plot(grid_sizes, sequential_times, label="Sequential Execution", marker='o', linestyle='--', color='red')
ax1.plot(grid_sizes, parallel_times, label="Parallel Execution (OpenMP)", marker='s', linestyle='-', color='blue')
ax1.plot(grid_sizes, gpu_times, label="GPU Execution (CUDA)", marker='^', linestyle='-', color='green')
ax1.tick_params(axis='y', labelcolor='black')

# Add a second y-axis for speedup values
ax2 = ax1.twinx()
ax2.set_ylabel("Speedup", color='purple')
ax2.plot(grid_sizes, cpu_speedup, label="CPU Speedup", marker='d', linestyle=':', color='purple')
ax2.plot(grid_sizes, gpu_speedup, label="GPU Speedup", marker='x', linestyle=':', color='orange')
ax2.tick_params(axis='y', labelcolor='purple')

# Title and legend
fig.suptitle("Performance Comparison of SWE Solver")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True)

# Save the plot as an image file
plt.savefig("performance_comparison.png")

# Show the plot
plt.show()
