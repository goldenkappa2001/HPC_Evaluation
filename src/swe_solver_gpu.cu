//swe_solver_gpu.cu
#include "swe_solver.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

/**
 * CUDA kernel for updating SWE variables using Runge-Kutta method.
 * Each thread updates one grid point.
 * 
 * @param d_eta Device array for water level values.
 * @param d_u Device array for velocity values.
 * @param grid_size Number of grid points.
 */
 
__global__ void rungeKuttaKernel(double *d_eta, double *d_u, int grid_size, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < grid_size - 1) {
        double k1 = dt * (0.5 * d_eta[i] + d_u[i]);
        double k2 = dt * (0.5 * (d_eta[i] + k1) + d_u[i]);
        double k3 = dt * (0.5 * (d_eta[i] + k2) + d_u[i]);
        double k4 = dt * (0.5 * (d_eta[i] + k3) + d_u[i]);
        d_eta[i] += (k1 + 2*k2 + 2*k3 + k4) / 6.0;
    }
}


/**
 * Runs the SWE simulation on GPU using CUDA.
 * Allocates memory, launches the kernel, and transfers results back.
 * 
 * @param steps Number of time steps to simulate.
 */
 
void SWESolver::runGPU(int steps) {
    double *d_eta, *d_u;
    size_t size = grid_size * sizeof(double);

	// Allocate memory on GPU
    cudaMalloc((void**)&d_eta, size);
    cudaMalloc((void**)&d_u, size);

	// Copy data from CPU to GPU
    cudaMemcpy(d_eta, eta.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), size, cudaMemcpyHostToDevice);

	// Define CUDA execution configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid = (grid_size + threadsPerBlock - 1) / threadsPerBlock;

	// Launch kernel multiple times for time stepping
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < steps; ++t) {
        rungeKuttaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_eta, d_u, grid_size, dt);
        cudaDeviceSynchronize(); // Ensure execution order
    }
 
    auto end = std::chrono::high_resolution_clock::now();
    execution_time = end - start;

    // Copy results back to CPU
    cudaMemcpy(eta.data(), d_eta, size, cudaMemcpyDeviceToHost);

	// Free GPU memory
    cudaFree(d_eta);
    cudaFree(d_u);

    printExecutionTime();
}
