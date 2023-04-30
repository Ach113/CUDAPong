#include <time.h>
// #include <stdio.h>
// #include <stdlib.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void outer_kernel(double* a, double* b, double* result, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        result[i * n + j] = a[i] * b[j];
    }
}

double* outer(double* a, double* b, int m, int n) {
    double* d_a, * d_b, * d_result;
    double* result = new double[m * n];

    // Allocate memory on device
    cudaMalloc(&d_a, m * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_result, m * n * sizeof(double));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Call kernel
    outer_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_result, m, n);

    // Copy output data from device to host
    cudaMemcpy(result, d_result, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

int main() {
    double a[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 };
    double b[] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16 };
    // double a[] = {1, 2, 3};
    // double b[] = {4, 5, 6};
    int m = sizeof(a) / sizeof(double);
    int n = sizeof(b) / sizeof(double);

    double* result = outer(a, b, m, n);

    // Print the result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] result;

    return 0;
}


// __global__ void outer_kernel(const double* a, const double* b, double* result, const int m, const int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < m && j < n) {
//         result[i * n + j] = a[i] * b[j];
//     }
// }

// std::vector<std::vector<double>> outer(const std::vector<double>& a, const std::vector<double>& b) {
//     int m = a.size();
//     int n = b.size();
//     std::vector<std::vector<double>> result(m, std::vector<double>(n));

//     double* d_a, *d_b, *d_result;

//     cudaMalloc((void**)&d_a, m * sizeof(double));
//     cudaMalloc((void**)&d_b, n * sizeof(double));
//     cudaMalloc((void**)&d_result, m * n * sizeof(double));

//     cudaMemcpy(d_a, a.data(), m * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

//     dim3 block_size(16, 16);
//     dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

//     outer_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, m, n);

//     cudaMemcpy(result.data(), d_result, m * n * sizeof(double), cudaMemcpyDeviceToHost);

//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_result);

//     return result;
// }

// std::vector<std::vector<double>> outer(const std::vector<double>& a, const std::vector<double>& b) {
//     int m = a.size();
//     int n = b.size();
//     std::vector<std::vector<double>> result(m, std::vector<double>(n));

//     std::cout << "CUDA Alloc\n";
//     // Allocate memory on device
//     double* dev_a;
//     double* dev_b;
//     double* dev_result;
//     cudaMalloc((void**)&dev_a, m * sizeof(double));
//     cudaMalloc((void**)&dev_b, n * sizeof(double));
//     cudaMalloc((void**)&dev_result, m * n * sizeof(double));

//     std::cout << "Copy In Data from Host to Dev\n";
//     // Copy input data from host to device
//     cudaMemcpy(dev_a, a.data(), m * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

//     std::cout << "Calling Outer Kernel\n";
//     // Launch kernel
//     dim3 blockDim(16, 16);
//     dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
//     outer_kernel<<<gridDim, blockDim>>>(dev_a, dev_b, dev_result, m, n);

//     std::cout << "Copying Kernel Outer Result to Host\n";
//     // Copy output data from device to host
//     cudaMemcpy(result.data(), dev_result, m * n * sizeof(double), cudaMemcpyDeviceToHost);

//     std::cout << "Freeing CUDA Mem\n";
//     // Free memory on device
//     cudaFree(dev_a);
//     cudaFree(dev_b);
//     cudaFree(dev_result);

//     return result;
// }

// int main() {
//     std::vector<double> a = {6, 2};
//     std::vector<double> b = {2, 5};
//     std::vector<std::vector<double>> result = outer(a, b);

//     std::cout << "Retrieved result outer\n";
//     std::cout << "result.size() = " << result.size() << std::endl;
//     std::cout << "result[0].size() = " << result[0].size() << std::endl;

//     for (int i = 0; i < result.size(); i++) {
//         for (int j = 0; j < result[i].size(); j++) {
//             std::cout << result[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }

// int main() {
//     std::vector<double> a = {1.0, 2.0, 3.0};
//     std::vector<double> b = {4.0, 5.0, 6.0};

//     std::vector<std::vector<double>> result = outer(a, b);

//     for (int i = 0; i < result.size(); i++) {
//         for (int j = 0; j < result[i].size(); j++) {
//             std::cout << result[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }