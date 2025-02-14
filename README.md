# 🌊 Shallow Water Equations Solver (HPC Implementation) 🚀  

[![GPU Acceleration](https://img.shields.io/badge/GPU-Accelerated-blue)](https://developer.nvidia.com/cuda-zone)
[![OpenMP Parallelization](https://img.shields.io/badge/OpenMP-Supported-orange)](https://www.openmp.org/)
[![C++](https://img.shields.io/badge/Language-C++-blue)](https://isocpp.org/)

## 📌 Project Overview  
This repository contains a **High-Performance Computing (HPC) implementation** of a **Shallow Water Equations (SWE) solver** using **C++ with OpenMP and CUDA**. The solver is optimized to run in **three execution modes**:  

✔ **Sequential Execution** (Standard CPU Processing)  
✔ **Parallel Execution** (OpenMP Multithreading)  
✔ **GPU-Accelerated Execution** (CUDA for NVIDIA GPUs)  

The goal of this project is to analyze **parallel performance, scalability, and computational efficiency** using different hardware configurations.

---

## 🛠️ Features  
✅ **Fourth-Order Runge-Kutta (RK4) Integration** for accurate numerical stability  
✅ **OpenMP Parallelization** for multi-core CPU execution  
✅ **CUDA Acceleration** for massive parallel speedup  
✅ **Configurable Grid Size & Threads** for benchmarking  
✅ **Performance Metrics & Logging** with execution time and speedup calculations  

---

## 📁 Repository Organization  

```plaintext
📦 hpc  
 ┣ 📂 repo                             # Main project folder  
 ┃ ┣ 📂 benchmark                      # Performance evaluation & benchmarking results  
 ┃ ┃ ┣ 📜 performance_results.txt      # Duplicate copy of performance logs  
 ┃ ┃ ┣ 🖼️ performance_comparison.png   # Graphical performance comparison  
 ┃ ┣ 📂 build                          # Compiled binaries and object files  
 ┃ ┃ ┣ 📜 performance_results.txt      # Logged execution times & speedup analysis
 ┃ ┣ 📂 include                        # Header files  
 ┃ ┃ ┣ 📜 benchmark.h                  # Benchmarking function declarations  
 ┃ ┃ ┣ 📜 swe_solver.h                 # Shallow Water Equations (SWE) solver header  
 ┃ ┃ ┣ 📜 runge_kutta.h                # Runge-Kutta method header  
 ┃ ┣ 📂 src                            # Source code files  
 ┃ ┃ ┣ 📜 benchmark.cpp                # Benchmarking implementation (moved from benchmark/)  
 ┃ ┃ ┣ 📜 main.cpp                     # Entry point for mode selection and execution  
 ┃ ┃ ┣ 📜 swe_solver.cpp               # SWE solver (CPU implementation)  
 ┃ ┃ ┣ 📜 swe_solver_gpu.cu            # SWE solver (GPU implementation - CUDA)  
 ┃ ┃ ┣ 📜 runge_kunta.cpp              # Runge-Kutta numerical integration  
 ┃ ┣ 📜 CMakeLists.txt                 # CMake configuration file  
 ┣ 📜 README.md                        # Project documentation  
````

---

## 🚀 Getting Started  

### **1️⃣ Requirements**  
- C++ Compiler (GCC or Clang)  
- OpenMP Support (`#pragma omp parallel`)  
- CUDA Toolkit (for GPU execution)  
- CMake (for easy compilation)  

### **2️⃣ Build & Compile**  
Use CMake to configure and build the project:  

```bash
mkdir build && cd build
cmake ..
make
```
### **3️⃣ Running the Solver** 
```bash
./swe_simulator
```

## 📊 Performance Analysis  

The following table presents the **execution times and speedup factors** for different grid sizes and thread configurations. The speedup values compare **OpenMP (CPU parallelization) and CUDA (GPU acceleration)** against the **sequential execution time**.

| Grid Size | Threads |   Sequential Time (s)   |   Parallel Time (s)   |   GPU Time (s)  | CPU Speedup | GPU Speedup |
|-----------|---------|-------------------------|-----------------------|-----------------|-------------|-------------|
| 10,000    | 8       | 2.21960                 | 2.21180               | 0.43365         | 1.00        | 5.11        |
| ...       | ...     | ...                     | ...                   | ...             | ...         | ...         |
| 100,000   | 64      | 21.13648                | 21.24483              | 0.13491         | 0.99        | 156.67      |

📌 **Note:** Execution times are approximate and may vary slightly due to **system load and scheduling**.  
📌 **Speedup Calculation:**  
\[
\text{Speedup} = \frac{\text{Sequential Execution Time}}{\text{Parallel or GPU Execution Time}}
\]








