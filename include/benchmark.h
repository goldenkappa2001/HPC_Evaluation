//benchmark.h
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "swe_solver.h"
#include <fstream>
#include <string>

void benchmark(SWESolver& solver, const std::string& mode, int steps, std::ofstream& results_file);

void run_benchmarks(std::ofstream& results_file);

#endif // BENCHMARK_H
