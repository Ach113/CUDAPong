#include <stdio.h>

class Matrix {
public:
    Matrix(int rows, int cols);
    ~Matrix();
    __host__ __device__ float* getDataPointer();
    __host__ __device__ int getRows();
    __host__ __device__ int getCols();
    void setData(float* data);
	void initRandom();
	void printMatrix();
private:
    float* data_;
    int rows_;
    int cols_;
};

Matrix::Matrix(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    cudaMallocManaged(&data_, rows * cols * sizeof(float));
}

Matrix::~Matrix() {
    cudaFree(data_);
}

__host__ __device__ float* Matrix::getDataPointer() {
    return data_;
}

__host__ __device__ int Matrix::getRows() {
    return rows_;
}

__host__ __device__ int Matrix::getCols() {
    return cols_;
}

void Matrix::setData(float* data) {
    cudaMemcpy(data_, data, rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::printMatrix() {
	for (int i=0; i < cols_; ++i) {
		for (int j=0; j < rows_; ++j) {
			printf("%f ", data_[i*cols_ + j]);
		}
		printf("\n");
	}
}

void Matrix::initRandom() {
	float* data;
	data = (float*)malloc(sizeof(float) * rows_ * cols_);
	for (int i=0; i< rows_ * cols_; ++i) {
		data[i] = rand() % 256;
	}
	cudaMemcpy(data_, data, rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice);
}


__global__ void matrixAdditionKernel(Matrix* A, Matrix* B, Matrix* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i * A->getCols() + j;
	
    if (i < A->getRows() && j < A->getCols()) {
        C->getDataPointer()[index] = A->getDataPointer()[index] + B->getDataPointer()[index];
    }
}



int main() {
	int rows = 2;
	int cols = 2;

	Matrix A(rows, cols);
	Matrix B(rows, cols);
	Matrix C(rows, cols);

	A.initRandom();
	B.initRandom();
	
	A.printMatrix();
	printf("\n");
	B.printMatrix();
	printf("\n");

	matrixAdditionKernel<<<1, 1>>>(&A, &B, &C);
	cudaDeviceSynchronize();

	float* data;
	data = (float*)malloc(sizeof(float) * rows * cols);

	cudaMemcpy(data, C.getDataPointer(), sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

	for (int i=0; i < cols; ++i) {
		for (int j=0; j < rows; ++j) {
			printf("%f ", data[i*cols + j]);
		}
		printf("\n");
	}

	printf("Done...\n");
	return 0;
}
