#include <stdio.h>
#include <math.h>
#include <map>
#include <tuple>

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
    2. implementation of np.outer
    3. implementation of .transpose()
    4. Karpathy's code deals with 1D arrays, our Matrix is currently 2D
*/


Matrix* discountRewards(float *rewards, int size) {
    // create matrix that will store discounted rewards
    Matrix *dis = new Matrix(1, size);
    dis->init();

    float running_add = 0;
    for (int i=0; i < size; i++) {
        // reset the sum, since this was a game boundary (pong specific!)
        if (rewards[i] != 0) {
            running_add = 0;
        }
        running_add = running_add * GAMMA + rewards[i];
        dis->getData()[i] = running_add;
    }

    return dis;
}


std::tuple<Matrix*, Matrix*> policyForward(Matrix* X, Matrix* W1, Matrix* W2) {
    Matrix *h = new Matrix(rows, cols);  // hidden layer
    dotProduct<<<dimGrid, dimBlock>>>(X->getData(), W1->getData(), h->getData(), rows);
    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), size);
    Matrix *p = new Matrix(rows, cols);  // output layer
    dotProduct<<<dimGrid, dimBlock>>>(W2->getData(), h->getData(), p->getData(), rows);
    sigmoid<<<dimGrid, dimBlock>>>(p->getData(), p->getData(), size);
    return std::make_tuple(p, h);
}

//std::tuple<Matrix*, Matrix*> policyForward(Matrix* h, Matrix *p) {
//    Matrix *dw2 = new Matrix(rows, cols);
//    // TODO! needs transpose here
//    dotProduct<<<dimGrid, dimBlock>>>(h->getData(), p->getData(), dw2->getData(), rows);
//    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), size);
//}

//def policy_backward(eph, epdlogp):
//""" backward pass. (eph is array of intermediate hidden states) """
//dW2 = np.dot(eph.T, epdlogp).ravel()
//dh = np.outer(epdlogp, model['W2'])
//dh[eph <= 0] = 0 # backpro prelu
//dW1 = np.dot(dh.T, epx)
//return {'W1':dW1, 'W2':dW2}


int main() {
    // dictionary with {key: layer name, value: layer params}
	std::map<std::string, Matrix> model;

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
	// A.dot(B) | A.outer(B)
	outerProduct<<<dimGrid, dimBlock>>>(A.getData(), B.getData(), C.getData(), rows);
	cudaDeviceSynchronize();
	
	// relu
	relu<<<dimGrid, dimBlock>>>(C.getData(), C.getData(), size);
	cudaDeviceSynchronize();
	// sigmoid
	sigmoid<<<dimGrid, dimBlock>>>(C.getData(), C.getData(), size);
	cudaDeviceSynchronize();

    // transpose
    transpose<<<dimGrid, dimBlock>>>(A.getData(), A.getData(), rows, cols);
    cudaDeviceSynchronize();

	A.printMatrix();
    // print shape
    auto shape = C.shape();
    printf("Shape: (%d, %d)\n", std::get<0>(shape), std::get<1>(shape));
	
	return 0;
}
