#ifndef UTIL_H
#define UTIL_H

inline __device__ float sigmoid_(float x) { return 1.0 / (1.0 + exp(-x)); }
__global__ void sigmoid(const float* inputs, float* outputs, const int size);
__global__ void relu(const float* inputs, const float* query, float* outputs, const int size);
__global__ void mul(const float* inputs, float* outputs, const int size, const float scalar);
__global__ void dotProduct(float *A, float *B, float *C, int m, int n, int k);
__global__ void outerProduct(const float* A, const float* B, float* result, int m, int n);
__global__ void transpose(const float *input, float *output, int rows, int cols);
#endif