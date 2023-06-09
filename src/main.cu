#include <stdio.h>
#include <math.h>
#include <map>
#include <tuple>
#include <vector>
#include <ctime>

#include "matrix.h"
#include "util.h"

#define H 200
#define D 80*80

#define BLOCK_SIZE 32

const int rows = 80;
const int cols = 200*80;
unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
dim3 dimGrid(grid_cols, grid_rows);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


/*
 Feedforward of the neural network, passes inputs through 2-layer nn
 * Matrix* X: input matrix (environment state)
 * Matrix* W1: weights of layer 1
 * Matrix* W2: weights of layer 2
 * returns: output logit of the model
 */
float policyForward(Matrix* X, Matrix* W1, Matrix* W2) {
    Matrix *h = new Matrix(H, 1);  // hidden layer
    printf("X shape: (%d, %d)\n", X->getRows(), X->getCols());
    printf("W1 shape: (%d, %d)\n", W1->getRows(), W1->getCols());
    // perform dot product on input and first hidden layer
    dotProduct<<<dimGrid, dimBlock>>>(X->getData(), W1->getData(), h->getData(), D, D, H);
    cudaDeviceSynchronize();
    // relu activation
    relu<<<dimGrid, dimBlock>>>(h->getData(), h->getData(), h->getData(), H);
    cudaDeviceSynchronize();

    //h->printMatrix();
    printf("W2 shape: (%d, %d)\n", W2->getRows(), W2->getCols());
    printf("h shape: (%d, %d)\n", h->getRows(), h->getCols());

    Matrix *p = new Matrix(1, 1);  // output layer
    // perform dot product on the output of hidden layer with the output layer
    dotProduct<<<dimGrid, dimBlock>>>(W2->getData(), h->getData(), p->getData(), H, 1, H);
    cudaDeviceSynchronize();

    p->printMatrix();
    printf("\n");

    // sigmoid activation function
    sigmoid<<<dimGrid, dimBlock>>>(p->getData(), p->getData(), 1);
    cudaDeviceSynchronize();

    // return the logit
    return p->getData()[0];
}
    

int main() {
    int action;
    float p;

    std::srand(std::time(nullptr));

    while (1) {
        // initialize A
        Matrix X(1, D); // X shape -> (6400,)
        X.init();

        Matrix W1(D, H); // W1 shape -> (6400, 200)
        W1.init();
        Matrix W2(1, H); // W2 shape -> (200,)
        W2.init();

        p = policyForward(&X, &W1, &W2);

        // get the action based on model's output
        float r = ((float) rand() / (RAND_MAX));
        printf("%f\n", r);
        action = (r < p) ? 2 : 3;

        printf("Logit: %f, action: %d\n", p, action);
        break;
    }

	
	return 0;
}
