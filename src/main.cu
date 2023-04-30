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


const int rows = 80;
const int cols = 80*200;
// const int size = rows * cols;
unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
dim3 dimGrid(grid_cols, grid_rows);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

/*
TODO:
	1. convert CUDA kernels to Matrix class methods
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
    Matrix *h = new Matrix(200, 1);  // hidden layer
    // perform dot product on input and first hidden layer
    dotProduct<<<dimGrid, dimBlock>>>(X->getData(), W1->getData(), h->getData(), D, D, H);
    cudaDeviceSynchronize();
    // relu activation
    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), h->getData(), H);
    cudaDeviceSynchronize();

    Matrix *p = new Matrix(1, 1);  // output layer
    // perform dot product on the output of hidden layer with the output layer
    dotProduct<<<dimGrid, dimBlock>>>(W2->getData(), h->getData(), p->getData(), H, 1, H);
    cudaDeviceSynchronize();
    // sigmoid activation function
    sigmoid<<<dimGrid, dimBlock>>>(p->getData(), p->getData(), 1);
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
//    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), h->getData(), size);
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
	// std::map<std::string, Matrix> model;

	// initialize A
	Matrix X(1, D); // X shape -> (6400,)
	X.init();
	Matrix W1(D, H); // W1 shape -> (6400, 200)
    W1.init();
	Matrix W2(H, 1); // W2 shape -> (200,)
    W2.init();

    auto result = policyForward(&X, &W1, &W2);
    auto h = std::get<0>(result);
    auto p = std::get<1>(result);
    h->printMatrix();
    printf("\n");
    p->printMatrix();
    // print shape
    auto shape = h->shape();
    printf("Shape: (%d, %d)\n", std::get<0>(shape), std::get<1>(shape));
	
	return 0;
}
