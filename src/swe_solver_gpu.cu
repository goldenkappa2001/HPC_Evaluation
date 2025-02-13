//swe_solver_gpu.cu
#include "swe_solver.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void updateGPUKernel(double* d_eta, double* d_u, int grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 0 && i < grid_size) {
        double k1 = 0.5 * d_eta[i] + d_u[i];
        d_eta[i] += k1;
    }
}

void SWESolver::runGPU(int steps) {
    double *d_eta, *d_u;
    size_t size = grid_size * sizeof(double);

    cudaMalloc((void**)&d_eta, size);
    cudaMalloc((void**)&d_u, size);

    cudaMemcpy(d_eta, eta.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (grid_size + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < steps; ++t) {
        updateGPUKernel<<<blocksPerGrid, threadsPerBlock>>>(d_eta, d_u, grid_size);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    execution_time = end - start;
    cudaMemcpy(eta.data(), d_eta, size, cudaMemcpyDeviceToHost);

    cudaFree(d_eta);
    cudaFree(d_u);

    printExecutionTime();
}
