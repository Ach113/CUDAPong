#include <stdio.h>
#include <math.h>
#include <map>

#include "matrix.h"
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


/*
TODO:  
	1. convert CUDA kernels to Matrix class methods
*/

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
	
	return 0;
}
