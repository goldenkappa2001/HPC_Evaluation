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
### **2️⃣ Running the Solver** 
```bash
./swe_simulator
```









