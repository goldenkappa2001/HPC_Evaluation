//main.cpp
#include "swe_solver.h"
#include "benchmark.h"
#include <iostream>
#include <fstream>

/**
 * Main entry point of the program.
 * Runs the benchmark tests and writes results to a file.
 *
 * @return Exit status of the program.
 */
 
int main() {
    std::ofstream results_file("performance_results.txt");

    // Run the benchmark tests for sequential, parallel, and GPU execution.
    run_benchmarks(results_file);
    results_file.close();
    
    std::cout << "Benchmark completed. Results saved in performance_results.txt\n";
    return 0;
}

