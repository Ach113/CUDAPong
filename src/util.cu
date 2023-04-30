#include "util.h"

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define BLOCK_ROWS 8

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
__global__ void dotProduct(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0;

    for (int i = 0; i < (n - 1) / TILE_SIZE + 1; ++i) {
        if (row < m && i * TILE_SIZE + tx < n) {
            A_tile[ty][tx] = A[row * n + i * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = 0;
        }
        if (col < k && i * TILE_SIZE + ty < n) {
            B_tile[ty][tx] = B[(i * TILE_SIZE + ty) * k + col];
        } else {
            B_tile[ty][tx] = 0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += A_tile[ty][j] * B_tile[j][tx];
        }
        __syncthreads();
    }
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

__global__ void transpose(const float *input, float *output, int rows, int cols) {
    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = tid_x + tid_y * cols;

    if (tid_x < cols && tid_y < rows) {
        output[idx] = input[tid_y + tid_x * rows];
    }
}

__global__ void outerProduct(const float* a, const float* b, float* result, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        result[i * n + j] = a[i] * b[j];
    }
}