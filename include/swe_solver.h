//swe_solver.h
#ifndef SWE_SOLVER_H
#define SWE_SOLVER_H

#include <vector>
#include <chrono>

/**
 * Class to solve the Shallow Water Equations (SWE) numerically.
 * Supports sequential, parallel, and GPU execution modes.
 */
 
class SWESolver {
public:

	/**
	 * Constructor to initialize solver parameters.
	 * 
	 * @param grid_size Number of grid points in the simulation.
	 * @param dx Spatial resolution.
	 * @param dt Time step for simulation.
	 * @param num_threads Number of threads for parallel execution.
	 */
	 
    SWESolver(int grid_size, double dx, double dt, int num_threads);

	/**
    * Runs the simulation sequentially (single-threaded execution).
    * 
    * @param steps Number of time steps to simulate.
    */
    
    void runSequential(int steps);
    
    /**
     * Runs the simulation in parallel using OpenMP.
     * 
     * @param steps Number of time steps to simulate.
     */
     
    void runParallel(int steps);

    /**
     * Runs the simulation on GPU (to be implemented in CUDA).
     * 
     * @param steps Number of time steps to simulate.
     */
     
    void runGPU(int steps);

	/**
     * Prints the execution time of the last run.
     */
     
    void printExecutionTime();

private:
    int grid_size;		///< Number of grid points
    double dx;			///< Spatial resolution
    double dt;			///< Time step
    int num_threads;	///< Number of threads used in parallel execution

    std::vector<double> eta;	///< Stores water level values
    std::vector<double> u;		///< Stores velocity values
	
    std::chrono::duration<double> execution_time; ///< Stores execution duration
};

#endif // SWE_SOLVER_H
