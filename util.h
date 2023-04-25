inline __device__ float sigmoid_(float x) { return 1.0 / (1.0 + exp(-x)); }
__global__ void sigmoid(const float* inputs, float* outputs, const int size);
__global__ void relu(const float* inputs, float* outputs, const int size);
__global__ void mul(const float* inputs, float* outputs, const int size, const float scalar);
__global__ void matMul(float *d_a, float *d_b, float *d_result, int n);
