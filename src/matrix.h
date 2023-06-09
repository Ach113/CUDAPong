#ifndef MATRIX_H
#define MATRIX_H

#include <tuple>

class Matrix {
public:
    Matrix(int rows, int cols);
    ~Matrix() { cudaFree(this->data); };
    inline __host__ __device__ float* Matrix::getData() { return data; };
    inline __host__ __device__ int Matrix::getRows() { return rows; };
    inline __host__ __device__ int getCols() { return cols; };
    inline void setData(float* data) {cudaMemcpy(data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice); };
	void init();
	void printMatrix();
    std::tuple<int, int> shape() { return {rows, cols}; };
private:
    float* data;
    int rows;
    int cols;
};
#endif