# HPC_Final_Program
DESCRIPTION:Parallel Gaussian elimination method for solving systems of linear equations

Parallel Gaussian Elimination Program

Overview

This program implements the parallel Gaussian elimination method for solving systems of linear equations. It uses MPI (Message Passing Interface) to distribute the computational workload across multiple processes, enabling efficient handling of large matrices.

Author

Jiawei Zhang
Date of Creation

July 11, 2024
Features

Parallel Gaussian Elimination: The program utilizes MPI to parallelize the Gaussian elimination process, significantly reducing computation time for large-scale matrices.
Block-based Computation: The matrix is divided into blocks, and each process handles a subset of the matrix, contributing to the overall solution.
Performance Comparison: The program also performs a serial (sequential) Gaussian elimination to compare the results and execution time with the parallel version.
Prerequisites

MPI: The program requires MPI to be installed on your system.

Installation on Ubuntu: sudo apt-get install libopenmpi-dev openmpi-bin
C++ Compiler: A C++ compiler with MPI support is required.

To compile the program, use the provided Makefile or compile manually with the following command:

Usage

mpirun --allow-run-as-root -np [number_of_processes] ./program2 [matrix_size]

Example code: mpirun --allow-run-as-root -np 4 ./program2 16

Output MPI Computation: The program outputs the time taken for the parallel computation using MPI. Serial Computation: After the parallel computation, the program runs a sequential Gaussian elimination for comparison. Result Verification: The program compares the results of the parallel and serial computations and outputs whether they match.
