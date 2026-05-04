/*
 * EvoGrad CUDA Kernels
 *
 * Custom CUDA kernels for evolutionary operators.  These are compiled
 * via torch.utils.cpp_extension when CUDA is available and fall back
 * to the pure-PyTorch implementations in operators.py otherwise.
 *
 * Kernels
 * -------
 *  tournament_select_kernel   — P parallel tournaments in one pass
 *  uniform_crossover_kernel   — fused crossover for P/2 pairs
 *  gaussian_mutate_kernel     — vectorised Gaussian noise addition
 *  rastrigin_fitness_kernel   — batched benchmark fitness evaluation
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>

// ------------------------------------------------------------------ //
// Utilities
// ------------------------------------------------------------------ //

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// ------------------------------------------------------------------ //
// Tournament selection
//
// Each thread handles one output slot.
// Reads k random candidates from the fitness array and emits the
// winner's index into out_idx[slot].
//
// Grid: (P,)  Block: 1
// ------------------------------------------------------------------ //

__global__ void tournament_select_kernel(
    const float* __restrict__ fitness,   // (P,)
    int*         __restrict__ out_idx,   // (P,)
    int P, int k, unsigned long seed
) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= P) return;

    curandState rng;
    curand_init(seed + slot, 0, 0, &rng);

    int best_idx = (int)(curand_uniform(&rng) * P) % P;
    float best_fit = fitness[best_idx];

    for (int i = 1; i < k; ++i) {
        int cand = (int)(curand_uniform(&rng) * P) % P;
        if (fitness[cand] > best_fit) {
            best_fit = fitness[cand];
            best_idx = cand;
        }
    }
    out_idx[slot] = best_idx;
}

// ------------------------------------------------------------------ //
// Uniform crossover
//
// Each thread handles one gene position for one pair.
// mask > rate  →  child1 gets parent1's gene (else swap).
//
// Grid: (P/2, G)  Block: 1
// ------------------------------------------------------------------ //

__global__ void uniform_crossover_kernel(
    const float* __restrict__ parents,   // (P, G)
    float*       __restrict__ offspring, // (P, G)
    int P, int G, float rate, unsigned long seed
) {
    int pair = blockIdx.x;               // which pair (0 .. P/2-1)
    int gene = blockIdx.y;               // which gene (0 .. G-1)
    if (pair >= P / 2 || gene >= G) return;

    curandState rng;
    curand_init(seed + pair * G + gene, 0, 0, &rng);

    int p1 = pair * 2;
    int p2 = pair * 2 + 1;
    float g1 = parents[p1 * G + gene];
    float g2 = parents[p2 * G + gene];
    float u  = curand_uniform(&rng);

    offspring[p1 * G + gene] = (u < rate) ? g1 : g2;
    offspring[p2 * G + gene] = (u < rate) ? g2 : g1;
}

// ------------------------------------------------------------------ //
// Gaussian mutation
//
// Grid: (P, G)  Block: 1
// ------------------------------------------------------------------ //

__global__ void gaussian_mutate_kernel(
    const float* __restrict__ genomes, // (P, G)
    float*       __restrict__ out,     // (P, G)
    int P, int G, float sigma, float rate, unsigned long seed
) {
    int ind  = blockIdx.x;
    int gene = blockIdx.y;
    if (ind >= P || gene >= G) return;

    curandState rng;
    curand_init(seed + ind * G + gene, 0, 0, &rng);

    float val = genomes[ind * G + gene];
    float u   = curand_uniform(&rng);
    if (u < rate) {
        val += curand_normal(&rng) * sigma;
    }
    out[ind * G + gene] = val;
}

// ------------------------------------------------------------------ //
// Rastrigin fitness (batch)
//
// Grid: (P,)  Reduction over genes via shared memory
// ------------------------------------------------------------------ //

__global__ void rastrigin_fitness_kernel(
    const float* __restrict__ genomes, // (P, G)
    float*       __restrict__ fitness, // (P,)
    int P, int G, float A
) {
    extern __shared__ float sdata[];
    int ind  = blockIdx.x;
    int tid  = threadIdx.x;
    if (ind >= P) return;

    // Each thread accumulates a partial sum over genes
    float sum = 0.0f;
    for (int g = tid; g < G; g += blockDim.x) {
        float x = genomes[ind * G + g];
        sum += x * x - A * cosf(2.0f * M_PI * x);
    }
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        fitness[ind] = -(A * G + sdata[0]);   // negated: higher = better
    }
}


// ------------------------------------------------------------------ //
// PyTorch extension bindings
// ------------------------------------------------------------------ //

torch::Tensor tournament_select_cuda(
    torch::Tensor fitness, int k, int64_t seed
) {
    int P = fitness.size(0);
    auto out_idx = torch::empty({P}, torch::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(fitness.device()));
    int threads = 256;
    int blocks  = (P + threads - 1) / threads;
    tournament_select_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(),
        out_idx.data_ptr<int>(),
        P, k, (unsigned long)seed
    );
    return out_idx.to(torch::kInt64);
}

torch::Tensor uniform_crossover_cuda(
    torch::Tensor parents, float rate, int64_t seed
) {
    int P = parents.size(0);
    int G = parents.size(1);
    auto offspring = torch::empty_like(parents);
    dim3 grid(P / 2, G);
    uniform_crossover_kernel<<<grid, 1>>>(
        parents.data_ptr<float>(),
        offspring.data_ptr<float>(),
        P, G, rate, (unsigned long)seed
    );
    return offspring;
}

torch::Tensor gaussian_mutate_cuda(
    torch::Tensor genomes, float sigma, float rate, int64_t seed
) {
    int P = genomes.size(0);
    int G = genomes.size(1);
    auto out = torch::empty_like(genomes);
    dim3 grid(P, G);
    gaussian_mutate_kernel<<<grid, 1>>>(
        genomes.data_ptr<float>(),
        out.data_ptr<float>(),
        P, G, sigma, rate, (unsigned long)seed
    );
    return out;
}

torch::Tensor rastrigin_cuda(torch::Tensor genomes, float A) {
    int P = genomes.size(0);
    int G = genomes.size(1);
    auto fitness = torch::empty({P}, genomes.options());
    int threads = std::min(G, 1024);
    rastrigin_fitness_kernel<<<P, threads, threads * sizeof(float)>>>(
        genomes.data_ptr<float>(),
        fitness.data_ptr<float>(),
        P, G, A
    );
    return fitness;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tournament_select",  &tournament_select_cuda,
          "Tournament selection (CUDA)");
    m.def("uniform_crossover",  &uniform_crossover_cuda,
          "Uniform crossover (CUDA)");
    m.def("gaussian_mutate",    &gaussian_mutate_cuda,
          "Gaussian mutation (CUDA)");
    m.def("rastrigin_fitness",  &rastrigin_cuda,
          "Rastrigin fitness evaluation (CUDA)");
}
