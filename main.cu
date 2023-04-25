#include <stdio.h>
#include <math.h>
#include <map>

#define H 200
#define BATCH_SIZE 10
#define LEARNING_RATE 1e-4
#define GAMMA 0.99
#define DECAY_RATE 0.99
#define D 80*80

#define BLOCK_SIZE 32


/***
TODO: 
	1. create Matrix class to avoid dealing with arrays and mallocs all the time
***/

inline __device__ float sigmoid_(float x) { return 1.0 / (1.0 + exp(-x)); }

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
__global__ void matMul(float *d_a, float *d_b, float *d_result, int n) {
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n) {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        } else {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n) {
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
    if(row < n && col < n) {
        d_result[row * n + col] = tmp;
    }
}


// displays matrix
void printMatrix(float *M, int rows, int cols) {
	printf("[");
	for (int i=0; i < cols; i++) {
		if (i == 0) {
			printf("[");
		} else {
			printf(" [");
		}
		for (int j=0; j < rows; j++) {
			if (j == rows - 1) {
				printf("%f", M[i*rows + j]);
			} else {
				printf("%f, ", M[i*rows + j]);
			}
		}
		if (i == cols - 1) {
			printf("]");
		} else {
			printf("],\n");
		}
	}
	printf("]\n");
}


void initMatrix(float *M, int size) {
	for (int i=0; i < size; i++) {
		M[i] = pow(-1, i % 2 == 1) * i;
	}
}


int main() {
	std::map<std::string, int*> model;
	int rows = 5; int cols = 5; int size = rows * cols;

	float *A; float *d_A;
	float *B; float *d_B;
	float *C; float *d_C;
	A = (float*)malloc(sizeof(float) * size);
	B = (float*)malloc(sizeof(float) * size);
	C = (float*)malloc(sizeof(float) * size);
	initMatrix(A, size);
	initMatrix(B, size);
	printMatrix(A, rows, cols);
	printf("\n");

	cudaMalloc(&d_A, size * sizeof(float));
	cudaMalloc(&d_B, size * sizeof(float));
	cudaMalloc(&d_C, size * sizeof(float));
	cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// scale B
	mul<<<dimGrid, dimBlock>>>(d_B, d_B, size, -1);
	cudaDeviceSynchronize();

	// A.dot(B)
	matMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, rows);
	cudaDeviceSynchronize();

	// relu
	relu<<<dimGrid, dimBlock>>>(d_C, d_C, size);
	cudaDeviceSynchronize();

	// sigmoid
	sigmoid<<<dimGrid, dimBlock>>>(d_C, d_C, size);
	cudaDeviceSynchronize();

	cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(C, rows, cols);

	free(A); free(B); free(C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	return 0;
}
