#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullptr
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0;
  float l_sum2 = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  int hidden_dim = hidden_size << 2;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_sum2 += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2
  float thread_local_data[2] = {l_sum, l_sum2};
  blockReduce<ReduceType::kSum, 2>(thread_local_data);
  l_sum = thread_local_data[0];
  l_sum2 = thread_local_data[1];

  __shared__ float mean, var;
  if (threadIdx.x == 0) {
    mean = l_sum / hidden_dim;
    var = l_sum2 / hidden_dim - mean * mean;
    vars[blockIdx.x] = var;
    if (means != nullptr) {
      means[blockIdx.x] = mean;
    }
    var += LN_EPSILON;
  }
  __syncthreads();
  // Step 3
  float rsqrt_var = rsqrtf(var);
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 s = scale_f4[idx];
    float4 b = bias_f4[idx];
    float4 result = make_float4(
        (val.x - mean) * rsqrt_var * s.x + b.x,
        (val.y - mean) * rsqrt_var * s.y + b.y,
        (val.z - mean) * rsqrt_var * s.z + b.z,
        (val.w - mean) * rsqrt_var * s.w + b.w
    );
    ln_res_f4[idx] = result;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backward kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_grad_buffer[TILE_DIM][TILE_DIM + 1];
  __shared__ float gamma_grad_buffer[TILE_DIM][TILE_DIM + 1];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1
  // load gamma and beta to shared memory

  float partial_dbeta = 0.f;
  float partial_dgamma = 0.f;

  for (int idx = threadIdx.y; idx < rows; idx += TILE_DIM) {
    // float dout = out_grad[idx * width + threadIdx.x + blockIdx.x * TILE_DIM];
    // float input = inp[idx * width + threadIdx.x + blockIdx.x * TILE_DIM];
    // partial_dbeta += dout;
    // float xhat = (input - beta_buffer[threadIdx.x]) / gamma_buffer[threadIdx.x];
    // partial_dgamma += xhat * dout;
    int col_idx = blockIdx.x * TILE_DIM + threadIdx.x;

    float dout = out_grad[idx * width + col_idx];
    float input_val = inp[idx * width + col_idx];

    float mean_val = means[idx];
    float var_val = vars[idx];

    float xhat = (input_val - mean_val) * rsqrt(var_val + LN_EPSILON);

    partial_dbeta += dout;
    partial_dgamma += xhat * dout;
  }
  // Step 2
  betta_grad_buffer[threadIdx.x][threadIdx.y] = partial_dbeta;
  gamma_grad_buffer[threadIdx.x][threadIdx.y] = partial_dgamma;
  b.sync();

  partial_dbeta = betta_grad_buffer[threadIdx.y][threadIdx.x];
  partial_dgamma = gamma_grad_buffer[threadIdx.y][threadIdx.x];

  // Step 3
  #pragma unroll
  for (int delta = TILE_DIM / 2; delta > 0; delta /= 2) {
    float dbeta_from_partner = g.shfl_down(partial_dbeta, delta);
    float dgamma_from_partner = g.shfl_down(partial_dgamma, delta);
    if (g.thread_rank() < delta) {
      partial_dbeta += dbeta_from_partner;
      partial_dgamma += dgamma_from_partner;
    }
  }

  if (g.thread_rank() == 0) {
    gamma_grad[threadIdx.y + blockIdx.x * TILE_DIM] = partial_dgamma;
    betta_grad[threadIdx.y + blockIdx.x * TILE_DIM] = partial_dbeta;
  }
  // Step 4
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backward kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  int original_hidden_dim = hidden_dim * 4;
  const int MAX_DIM = 1024;
  const T* inp_row = inp + blockIdx.x * original_hidden_dim;
  const T* out_grad_row = out_grad + blockIdx.x * original_hidden_dim;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp_row);
  const float4 *dout_f4 = reinterpret_cast<const float4 *>(out_grad_row);

  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *beta_f4 = reinterpret_cast<const float4 *>(betta);
  float var_val = vars[blockIdx.x];
  float mean_val = means[blockIdx.x];
  float rsqrt_var = rsqrt(var_val + LN_EPSILON);

  __shared__ float4 s_dxhat[MAX_DIM];
  __shared__ float4 s_xhat[MAX_DIM];

  float dxhat_sum = 0;
  float dxhat_xhat_sum = 0;

  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 inp_val = inp_f4[idx];
    float4 dout_val = dout_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    float4 beta_val = beta_f4[idx];

    float4 dxhat;
    dxhat.x = dout_val.x * gamma_val.x;
    dxhat.y = dout_val.y * gamma_val.y;
    dxhat.z = dout_val.z * gamma_val.z;
    dxhat.w = dout_val.w * gamma_val.w;
    s_dxhat[idx] = dxhat;

    float4 xhat;
    xhat.x = (inp_val.x - mean_val) * rsqrt_var;
    xhat.y = (inp_val.y - mean_val) * rsqrt_var;
    xhat.z = (inp_val.z - mean_val) * rsqrt_var;
    xhat.w = (inp_val.w - mean_val) * rsqrt_var;
    s_xhat[idx] = xhat;

    dxhat_sum += dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    dxhat_xhat_sum += dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z + dxhat.w * xhat.w;
  }
  // Step 2
  // block reduce
  float sums[2] = {dxhat_sum, dxhat_xhat_sum};
  blockReduce<ReduceType::kSum, 2>(sums);
   
  // Step 3
  // total_sums[0]: sum(dxhat)
  // total_sums[1]: sum(dxhat * xhat)
  __shared__ float total_sums[2];
  if (threadIdx.x == 0) {
    total_sums[0] = sums[0];
    total_sums[1] = sums[1];
  }
  __syncthreads();
 
  // Step 4
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dxhat = s_dxhat[idx];
    float4 xhat = s_xhat[idx];

    float4 dinp_val;
    dinp_val.x = (dxhat.x - (total_sums[0] + xhat.x * total_sums[1]) / original_hidden_dim) * rsqrt_var;
    dinp_val.y = (dxhat.y - (total_sums[0] + xhat.y * total_sums[1]) / original_hidden_dim) * rsqrt_var;
    dinp_val.z = (dxhat.z - (total_sums[0] + xhat.z * total_sums[1]) / original_hidden_dim) * rsqrt_var;
    dinp_val.w = (dxhat.w - (total_sums[0] + xhat.w * total_sums[1]) / original_hidden_dim) * rsqrt_var;

    reinterpret_cast<float4 *>(inp_grad)[blockIdx.x * hidden_dim + idx] = dinp_val;
  }

  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
