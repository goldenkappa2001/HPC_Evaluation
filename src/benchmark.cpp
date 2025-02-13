//benchmark.cpp
#include "benchmark.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>


/**
 * Runs a benchmark test for a given solver mode.
 * Measures execution time and logs the results.
 * 
 * @param solver Reference to the SWE solver instance.
 * @param mode Execution mode: "sequential", "parallel", or "gpu".
 * @param steps Number of time steps to simulate.
 * @param time Variable to store execution time.
 */
 
void benchmark(SWESolver& solver, const std::string& mode, int steps, double& time) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (mode == "sequential") solver.runSequential(steps);
    else if (mode == "parallel") solver.runParallel(steps);
    else if (mode == "gpu") solver.runGPU(steps);
    
    auto end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration<double>(end - start).count();

    // Print execution details
    std::cout << "Mode: " << mode 
              << " | Execution Time: " << std::fixed << std::setprecision(6) << time << " seconds\n";
}


/**
 * Runs benchmark tests for various grid sizes and thread configurations.
 * Logs the results to a file in a formatted manner.
 * 
 * @param results_file Reference to the output file stream.
 */


void run_benchmarks(std::ofstream& results_file) {
    int grid_sizes[] = {10000, 25000, 50000, 100000}; 
    int steps = 10000;
    int num_threads[] = {8, 16, 32, 64};

    results_file << "==========================================================================================================\n";
    results_file << " SHALLOW WATER EQUATION - PERFORMANCE RESULTS\n";
    results_file << "==========================================================================================================\n";
    results_file << " Grid Size | Threads | Sequential Time (s) | Parallel Time (s) | GPU Time (s) | CPU Speedup | GPU Speedup\n";
    results_file << "----------------------------------------------------------------------------------------------------------\n";

    for (int gs : grid_sizes) {
        for (int nt : num_threads) {
            SWESolver solver(gs, 1.0, 0.01, nt);

            double seq_time, par_time, gpu_time;

            std::cout << "\nRunning benchmarks for Grid Size: " << gs << " with " << nt << " threads...\n";

            benchmark(solver, "sequential", steps, seq_time);
            benchmark(solver, "parallel", steps, par_time);
            benchmark(solver, "gpu", steps, gpu_time);

            double cpu_speedup = seq_time / par_time;
            double gpu_speedup = seq_time / gpu_time;

            results_file << std::setw(10) << std::left << gs << " | "
                         << std::setw(7) << std::left << nt << " | "
                         << std::setw(20) << std::left << std::fixed << std::setprecision(5) << seq_time << " | "
                         << std::setw(18) << std::left << par_time << " | "
                         << std::setw(12) << std::left << gpu_time << " | "
                         << std::setw(12) << std::left << cpu_speedup << " | "
                         << std::setw(12) << std::left << gpu_speedup << "\n";
        }
    }

    results_file << "----------------------------------------------------------------------------------------------------------\n";
}
