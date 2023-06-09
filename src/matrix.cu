#include <stdio.h>
#include "matrix.h"

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    cudaMallocManaged(&data, rows * cols * sizeof(float));
}

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
		this->data[i] = ((float) rand() / (RAND_MAX));
	}
}
