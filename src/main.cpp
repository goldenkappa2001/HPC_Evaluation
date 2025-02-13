//main.cpp
#include "swe_solver.h"
#include "benchmark.h"
#include <iostream>
#include <fstream>

int main() {
    std::ofstream results_file("performance_results.txt");
    run_benchmarks(results_file);
    results_file.close();
    
    std::cout << "Benchmark completed. Results saved in performance_results.txt\n";
    return 0;
}

