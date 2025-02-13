//benchmark.h
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "swe_solver.h"
#include <fstream>
#include <string>


/**
 * Runs a benchmark test for a given solver mode.
 * Measures execution time and logs the results.
 * 
 * @param solver Reference to the SWE solver instance.
 * @param mode Execution mode: "sequential", "parallel", or "gpu".
 * @param steps Number of time steps to simulate.
 * @param results_file Reference to the output file stream.
 */
 
void benchmark(SWESolver& solver, const std::string& mode, int steps, std::ofstream& results_file);

/**
 * Runs benchmark tests for various grid sizes and thread configurations.
 * Logs the results to a file in a formatted manner.
 * 
 * @param results_file Reference to the output file stream.
 */
 
void run_benchmarks(std::ofstream& results_file);

#endif // BENCHMARK_H
