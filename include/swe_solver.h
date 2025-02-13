//swe_solver.h
#ifndef SWE_SOLVER_H
#define SWE_SOLVER_H

#include <vector>
#include <chrono>

class SWESolver {
public:
    SWESolver(int grid_size, double dx, double dt, int num_threads);

    void runSequential(int steps);
    void runParallel(int steps);
    void runGPU(int steps);

    void printExecutionTime();

private:
    int grid_size;
    double dx;
    double dt;
    int num_threads;

    std::vector<double> eta;
    std::vector<double> u;

    std::chrono::duration<double> execution_time;
};

#endif // SWE_SOLVER_H
