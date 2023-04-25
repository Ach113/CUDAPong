#include <stdio.h>
#include <math.h>
#include <map>

#include "util.h"

#define H 200
#define BATCH_SIZE 10
#define LEARNING_RATE 1e-4
#define GAMMA 0.99
#define DECAY_RATE 0.99
#define D 80*80

#define BLOCK_SIZE 32


const int rows = 5; 
const int cols = 5; 
const int size = rows * cols;
unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
dim3 dimGrid(grid_cols, grid_rows);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


class Matrix {
public:
    Matrix(int rows, int cols);
    ~Matrix() { cudaFree(data); };
    inline __host__ __device__ float* Matrix::getData() { return data; }
    inline __host__ __device__ int Matrix::getRows() { return rows; };
    inline __host__ __device__ int getCols() { return cols; };
    void setData(float* data);
	void init();
	void printMatrix();
private:
    float* data;
    int rows;
    int cols;
};

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    cudaMallocManaged(&data, rows * cols * sizeof(float));
}

inline void Matrix::setData(float* data) { 
	cudaMemcpy(data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice); 
}


/***
TODO: 
	1. create Matrix class to avoid dealing with arrays and mallocs all the time
***/

// displays matrix
void Matrix::printMatrix() {
	printf("[");
	for (int i=0; i < this->cols; i++) {
		if (i == 0) {
			printf("[");
		} else {
			printf(" [");
		}
		for (int j=0; j < this->rows; j++) {
			if (j == rows - 1) {
				printf("%f", this->data[i*rows + j]);
			} else {
				printf("%f, ", this->data[i*rows + j]);
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


void Matrix::init() {
	int size = this->rows * this->cols;
	for (int i=0; i < size; i++) {
		this->data[i] = pow(-1, i % 2 == 1) * i;
	}
}


int main() {
	std::map<std::string, int*> model;

	// initialize A
	Matrix A(rows, cols);
	A.init();
	A.printMatrix();
	printf("\n");
	// initialize B
	Matrix B(rows, cols);
	B.init();
	// initialize C
	Matrix C(rows, cols);

	// scale B
	mul<<<dimGrid, dimBlock>>>(B.getData(), B.getData(), size, -1);
	cudaDeviceSynchronize();
	// A.dot(B)
	matMul<<<dimGrid, dimBlock>>>(A.getData(), B.getData(), C.getData(), rows);
	cudaDeviceSynchronize();
	// relu
	relu<<<dimGrid, dimBlock>>>(C.getData(), C.getData(), size);
	cudaDeviceSynchronize();
	// sigmoid
	sigmoid<<<dimGrid, dimBlock>>>(C.getData(), C.getData(), size);
	cudaDeviceSynchronize();

	C.printMatrix();
	/*
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
	*/
	return 0;
}
