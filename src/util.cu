#include "util.h"

#define BLOCK_SIZE 32

// sigmoid activation function CUDA kernel
__global__ void sigmoid(const float* inputs, float* outputs, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        outputs[idx] = sigmoid_(inputs[idx]);
    }
}


// relu activation function CUDA kernel
__global__ void relu(const float* inputs, float* outputs, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        outputs[idx] = inputs[idx] >= 0 ? inputs[idx] : 0;
    }
}


// multiplication by scalar
__global__ void mul(const float* inputs, float* outputs, const int size, const float scalar) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        outputs[idx] = scalar * inputs[idx];
    }
}


// square matrix multiplication kernel
__global__ void dotProduct(float *d_a, float *d_b, float *d_result, int n) {
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n*n) {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        } else {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n*n) {
            tile_b[threadIdx.y][threadIdx.x] = 0;
    	} else {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        d_result[row * n + col] = tmp;
    }
}

__global__ void outerProduct(float* A, float* B, float* C, int n) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {

        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }

        C[i * n + j] = sum;
    } 
}