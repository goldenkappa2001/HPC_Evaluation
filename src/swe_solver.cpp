//swe_solver.cpp
#include "swe_solver.h"
#include "runge_kutta.h"
#include <iostream>
#include <chrono>

SWESolver::SWESolver(int grid_size, double dx, double dt, int num_threads)
    : grid_size(grid_size), dx(dx), dt(dt), num_threads(num_threads)
{
    // Initialize vectors
    eta.resize(grid_size, 0.0);
    u.resize(grid_size, 0.0);
}

void SWESolver::runSequential(int steps) {
    auto start = std::chrono::high_resolution_clock::now();

    // Example: sequential calculation using rungeKuttaStep
    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            double newVal = rungeKuttaStep(eta[j], u[j]);
            eta[j] += newVal;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    execution_time = end - start;
    printExecutionTime();
}

void SWESolver::runParallel(int steps) {
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            double newVal = rungeKuttaStep(eta[j], u[j]);
            eta[j] += newVal;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    execution_time = end - start;
    printExecutionTime();
}


void SWESolver::printExecutionTime() {
    std::cout << "Execution time: " << execution_time.count() << " seconds\n";
}
