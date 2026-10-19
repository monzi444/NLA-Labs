# Numerical Linear Algebra - Course Challenges

This repository contains the solutions and implementations for the laboratory challenges of the Numerical Linear Algebra course, part of the Master's Degree in High Performance Computing Engineering at Politecnico di Milano.

---

## Repository Purpose

The repository is organized by challenge providing C++ implementations:

- **Challenge 1 (`challenge1/`)**: Focuses on 2D image processing, discrete convolution, and sparse linear systems using the Eigen library. Key tasks include:
  - Applying smoothing, sharpening, and Laplacian edge-detection filters.
  - Constructing sparse convolution matrices in Compressed Row/Column Storage (CRS/CCS).
  - Importing and exporting data in Matrix Market format (`.mtx`).
  - Restoring degraded images by solving large sparse linear systems with iterative Krylov solvers (Conjugate Gradient, BiCGSTAB) and preconditioning techniques.

- **Challenge 2 (`challenge2/`)**: Specification and upcoming solution for the second course challenge.

---

## Prerequisites

To build and run the code, you need:

- A modern C++ compiler supporting C++20 (such as `g++` 10+ or `clang++`).
- **Eigen 3**: Template library for linear algebra (including unsupported modules).
- **LIS (Library of Iterative Solvers for Linear Systems)**: A scalable library for solving linear systems and eigenvalue problems with iterative methods.

## Building and Running

### Challenge 1

1. **Move into the challenge directory**:
   ```bash
   cd challenge1
   ```

2. **Compile the source files**:
   ```bash
   g++ -std=c++20 -O3 -I /usr/include/eigen3 challenge1.cpp utils/image_tools.cpp -o challenge1
   ```
   *(Adjust the `-I` include path if your Eigen headers are installed in a different directory, such as `/usr/local/include/eigen3` or Homebrew's include path).*

3. **Run the program**:
   Pass the path to an input image (for example, the provided sample image in `resources/deer.jpg`):
   ```bash
   ./challenge1 <input-path>
   ```

4. **Outputs**:
   The execution generates filtered images and data files stored directly in `challenge1/resources/`:
   - Filtered images: `blurred.png`, `noisy.png`, `noisy_blurred.png`, `sharpened.png`, and `dectioned.png`.
   - Linear system solutions: `system_solution.png` and `system_solution_2.png`.
   - Matrix Market files and logs: `h2.mtx`, `sol.mtx`, `w.mtx`, and `hist.txt` (convergence history).

---

### Challenge 2

Refer to the problem description in `challenge2/Challenge2.pdf` for specifications and details. Build and run instructions will be updated here as development progresses.

---

## Repository Structure

```plaintext
NLA-Labs/
├── README.md              # Project documentation
├── challenge1/
│   ├── Challenge1.pdf     # Challenge 1 problem description
│   ├── challenge1.cpp     # Main program for Challenge 1
│   ├── convolution.cpp    # Standalone convolution test script
│   ├── libs/              # Third-party image headers (stb_image, stb_image_write)
│   ├── utils/             # Image conversion utilities (image_tools.h / .cpp)
│   └── resources/         # Input images, generated outputs, and matrix files
└── challenge2/
    ├── Challenge2.pdf     # Challenge 2 problem description
    └── ...
```
