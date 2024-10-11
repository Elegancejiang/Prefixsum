# Reduction

-------------------

## Introduction

This program aims to implement and optimize parallel reduction operations in CUDA. It is based on NVIDIA's PPT "Optimizing Parallel Reduction in CUDA" and fully implements the five-step optimization strategy proposed in the document. After each optimization step, we verify the correctness of the optimized results by comparing them with the original reduction results. On the NVIDIA GeForce RTX 4090 and NVIDIA GeForce RTX 3060 Laptop platforms, we tested the execution time of the program after each optimization step and compared it with the reduction summation functions of the Thrust and CUB libraries. The test results show that the reduction operation after the five-step optimization is slightly faster than thrust::reduce function.

## Installation
#### Step 1 : Assert NVIDIA GPU and CUDA is available
To execute the Reduction program, verify that your system is equipped with an NVIDIA GPU and has the CUDA installed.

#### Step 2: Verify Your GPU's Compute Capability
The `nvcc` compilation command in the Makefile requires the compute capability of your GPU device (device 0). If you are unsure about the compute capability of your GPU device (device 0), you can run the script `GPU_information.sh` (execute the command `sh GPU_information.sh` in the terminal) to obtain this information.

#### Step 3: Confirm the Correct Makefile Settings
You can replace `arch=compute_86,code=sm_86` in the compilation command of the Makefile with the compute capability of your GPU device (device 0). Once this is done, you can proceed to execute the Makefile to compile the Reduction program(execute the command `make` in the terminal).

## Executing the example code of Reduction
First, execute the script `GPU_information.sh` to determine the compute capability of your GPU device (device 0). Next, modify the compute capability section in the Makefile's compilation command and then run the Makefile.

To run the Reduction program, simply execute the following command in the terminal:

```
./reduction 1024000
```

This command will run the Reduction program with an operation array length of 1024000.

## Release versions

#### <p align='left'>Version 1.0.0 (Oct. 11, 2024) </p>

* Added the five-step optimization strategy for reduction.
