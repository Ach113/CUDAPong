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


const int rows = 4;
const int cols = 4;
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
    for (int i=size - 1; i >= 0; i--) {
        // reset the sum, since this was a game boundary (pong specific!)
        if (rewards[i] != 0) {
            running_add = 0;
        }
        running_add = running_add * GAMMA + rewards[i];
        dis->getData()[i] = running_add;
    }

    return dis;
}


void customRelu() {

}


/*
 Feedforward of the neural network, passes inputs through 2-layer nn
 * Matrix* X: input matrix (environment state)
 * Matrix* W1: weights of hidden layer 1
 * Matrix* W2: weights of hidden layer 2
 * returns: tuple <h, p>, where h is output of layer 1, p is output of layer 2 (output layer)
 */
std::tuple<Matrix*, Matrix*> policyForward(Matrix* X, Matrix* W1, Matrix* W2) {
    Matrix *h = new Matrix(rows, cols);  // hidden layer
    // perform dot product on input and first hidden layer
    dotProduct<<<dimGrid, dimBlock>>>(X->getData(), W1->getData(), h->getData(), 0, 0, 0);
    cudaDeviceSynchronize();
    // relu activation
    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), size);
    cudaDeviceSynchronize();

    Matrix *p = new Matrix(rows, cols);  // output layer
    // perform dot product on the output of hidden layer with the output layer
    dotProduct<<<dimGrid, dimBlock>>>(W2->getData(), h->getData(), p->getData(), 0, 0, 0);
    cudaDeviceSynchronize();
    // sigmoid activation function
    sigmoid<<<dimGrid, dimBlock>>>(p->getData(), p->getData(), size);
    cudaDeviceSynchronize();

    return std::make_tuple(h, p);
}


//std::tuple<Matrix*, Matrix*> policyBackward(Matrix* eph, Matrix* epdlogp, Matrix* W2) {
//    Matrix *dh = new Matrix(rows, cols);
//    // transpose
//    transpose<<<dimGrid, dimBlock>>>(eph->getData(), eph->getData(), eph->getRows(), eph->getCols());
//    cudaDeviceSynchronize();
//
//    // dot product
//    dotProduct<<<dimGrid, dimBlock>>>(eph->getData(), epdlogp->getData(), 0, 0, 0);
//    cudaDeviceSynchronize();
//
//    // outer product
//    outerProduct<<<dimGrid, dimBlock>>>(epdlogp->getData(), W2->getData(), dh->getData(), 200);
//    cudaDeviceSynchronize();
//    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), size);
//    cudaDeviceSynchronize();
//
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
    B.printMatrix();
    printf("\n");
	// initialize C
	Matrix C(rows, cols);



    // scale B
	mul<<<dimGrid, dimBlock>>>(B.getData(), B.getData(), size, -1);
	cudaDeviceSynchronize();

//	// A.outer(B)
//  // WARNING: to make this code work, change shape of C to (rows*cols, rows*cols)
//    outerProduct<<<dimGrid, dimBlock>>>(A.getData(), B.getData(), C.getData(), rows*cols, rows*cols);
//    cudaDeviceSynchronize();

    // A.dot(B)
    dotProduct<<<dimGrid, dimBlock>>>(A.getData(), B.getData(), C.getData(), rows, cols, rows);
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

	C.printMatrix();
    // print shape
    auto shape = C.shape();
    printf("Shape: (%d, %d)\n", std::get<0>(shape), std::get<1>(shape));
	
	return 0;
}
