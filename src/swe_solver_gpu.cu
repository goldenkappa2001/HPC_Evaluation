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
 
__global__ void rungeKuttaKernel(double *d_eta, double *d_u, int grid_size, double dx, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < grid_size - 1) {
        // Compute k1
        double k1_eta = dt * (0.5 * d_eta[i] + d_u[i]);
        double k1_u = dt * (-9.81 * (d_eta[i] / dx));

        // Compute k2
        double k2_eta = dt * (0.5 * (d_eta[i] + 0.5 * k1_eta) + (d_u[i] + 0.5 * k1_u));
        double k2_u = dt * (-9.81 * ((d_eta[i] + 0.5 * k1_eta) / dx));

        // Compute k3
        double k3_eta = dt * (0.5 * (d_eta[i] + 0.5 * k2_eta) + (d_u[i] + 0.5 * k2_u));
        double k3_u = dt * (-9.81 * ((d_eta[i] + 0.5 * k2_eta) / dx));

        // Compute k4
        double k4_eta = dt * (0.5 * (d_eta[i] + k3_eta) + (d_u[i] + k3_u));
        double k4_u = dt * (-9.81 * ((d_eta[i] + k3_eta) / dx));

        // Update the values using RK4 formula
        d_eta[i] += (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta) / 6.0;
        d_u[i] += (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6.0;
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
        rungeKuttaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_eta, d_u, grid_size, dx, dt);
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
