#ifndef MATRIX_H
#define MATRIX_H

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
#endif