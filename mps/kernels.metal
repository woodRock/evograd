/*
 * EvoGrad Metal Shaders
 *
 * Custom GPU kernels for evolutionary operators on Apple Silicon.
 * Compiled via metalcompute when available; used as a reference
 * implementation when writing future PyTorch MPS custom ops.
 *
 * Design notes
 * ------------
 * - Random numbers use a PCG32 generator seeded per-thread.  Metal
 *   has no built-in PRNG so we inline one.  PCG32 passes BigCrush
 *   and costs ~2 instructions per call.
 * - Gaussian samples use Box-Muller.  Each thread generates a pair;
 *   the second value is discarded (acceptable for mutation where
 *   independence across genes matters more than throughput).
 * - Threadgroup sizes are tuned for the M2's 32-lane SIMD width.
 *   Tournament and mutation map one thread per output element.
 *   Fitness reduction uses threadgroup shared memory.
 */

#include <metal_stdlib>
using namespace metal;

// ------------------------------------------------------------------ //
// PCG32 inline PRNG
// ------------------------------------------------------------------ //

inline uint pcg32_next(thread uint& state) {
    uint old = state * 747796405u + 2891336453u;
    state    = old;
    uint word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

// Maps a uint32 sample to [0, 1)
inline float u32_to_01(uint x) {
    // Use top 24 bits for float mantissa precision
    return float(x >> 8u) * (1.0f / float(1u << 24u));
}

// Box-Muller: returns standard normal from two uniform samples
inline float box_muller_z0(uint u1_raw, uint u2_raw) {
    float u1 = u32_to_01(u1_raw);
    float u2 = u32_to_01(u2_raw);
    u1 = max(u1, 1e-7f);                    // guard against log(0)
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}


// ------------------------------------------------------------------ //
// Tournament selection
//
// One thread per output slot.  Reads k random fitness values and
// writes the index of the winner.
//
// Buffers
//   0  fitness    : float[P]        (read)
//   1  out_indices: uint[P]         (write)
//   2  params     : uint4           x=(P, k, seed, _)
// ------------------------------------------------------------------ //

kernel void tournament_select(
    const device float* fitness    [[ buffer(0) ]],
    device       uint*  out_idx    [[ buffer(1) ]],
    constant     uint4& params     [[ buffer(2) ]],  // (P, k, seed, _)
    uint gid [[ thread_position_in_grid ]]
) {
    uint P    = params.x;
    uint k    = params.y;
    uint seed = params.z;

    if (gid >= P) return;

    uint rng = seed ^ (gid * 2654435761u);   // initial state per thread
    pcg32_next(rng);                          // discard first output

    uint best_idx = pcg32_next(rng) % P;
    float best_fit = fitness[best_idx];

    for (uint i = 1u; i < k; i++) {
        uint cand = pcg32_next(rng) % P;
        float cf  = fitness[cand];
        if (cf > best_fit) {
            best_fit = cf;
            best_idx = cand;
        }
    }
    out_idx[gid] = best_idx;
}


// ------------------------------------------------------------------ //
// Uniform crossover
//
// Grid: (P/2 * G) linear threads; each handles one gene for one pair.
//
// Buffers
//   0  parents   : float[P * G]    (read)
//   1  offspring : float[P * G]    (write)
//   2  params    : uint4            x=(P, G, seed, _)
//   3  rate_buf  : float[1]         crossover rate
// ------------------------------------------------------------------ //

kernel void uniform_crossover(
    const device float*  parents    [[ buffer(0) ]],
    device       float*  offspring  [[ buffer(1) ]],
    constant     uint4&  params     [[ buffer(2) ]],
    const device float*  rate_buf   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint P    = params.x;
    uint G    = params.y;
    uint seed = params.z;
    float rate = rate_buf[0];

    uint total_pairs = P / 2u;
    if (gid >= total_pairs * G) return;

    uint pair = gid / G;
    uint gene = gid % G;

    uint rng = seed ^ (pair * G + gene) * 2246822519u;
    pcg32_next(rng);
    float u = u32_to_01(pcg32_next(rng));

    uint p1 = pair * 2u;
    uint p2 = pair * 2u + 1u;
    float g1 = parents[p1 * G + gene];
    float g2 = parents[p2 * G + gene];

    offspring[p1 * G + gene] = (u < rate) ? g1 : g2;
    offspring[p2 * G + gene] = (u < rate) ? g2 : g1;
}


// ------------------------------------------------------------------ //
// Gaussian mutation
//
// Grid: P * G threads, one per (individual, gene).
//
// Buffers
//   0  genomes_in : float[P * G]  (read)
//   1  genomes_out: float[P * G]  (write)
//   2  params     : uint4          x=(P, G, seed, _)
//   3  hyper      : float2         x=sigma, y=rate
// ------------------------------------------------------------------ //

kernel void gaussian_mutate(
    const device float*  genomes_in  [[ buffer(0) ]],
    device       float*  genomes_out [[ buffer(1) ]],
    constant     uint4&  params      [[ buffer(2) ]],
    const device float*  hyper       [[ buffer(3) ]],  // [sigma, rate]
    uint gid [[ thread_position_in_grid ]]
) {
    uint P    = params.x;
    uint G    = params.y;
    uint seed = params.z;
    float sigma = hyper[0];
    float rate  = hyper[1];

    if (gid >= P * G) return;

    uint rng = seed ^ gid * 3266489917u;
    pcg32_next(rng);

    float val = genomes_in[gid];
    float u   = u32_to_01(pcg32_next(rng));

    if (u < rate) {
        uint s1 = pcg32_next(rng);
        uint s2 = pcg32_next(rng);
        val += box_muller_z0(s1, s2) * sigma;
    }
    genomes_out[gid] = val;
}


// ------------------------------------------------------------------ //
// SBX crossover (Simulated Binary Crossover)
//
// Grid: P/2 * G threads.
//
// Buffers
//   0  parents   : float[P * G]
//   1  offspring : float[P * G]
//   2  params    : uint4  x=(P, G, seed, _)
//   3  eta_buf   : float[1]  distribution index (e.g. 20.0)
// ------------------------------------------------------------------ //

kernel void sbx_crossover(
    const device float*  parents    [[ buffer(0) ]],
    device       float*  offspring  [[ buffer(1) ]],
    constant     uint4&  params     [[ buffer(2) ]],
    const device float*  eta_buf    [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint P    = params.x;
    uint G    = params.y;
    uint seed = params.z;
    float eta = eta_buf[0];

    uint total_pairs = P / 2u;
    if (gid >= total_pairs * G) return;

    uint pair = gid / G;
    uint gene = gid % G;

    uint rng = seed ^ (pair * G + gene) * 2246822519u;
    pcg32_next(rng);
    float u = u32_to_01(pcg32_next(rng));
    u = clamp(u, 1e-9f, 1.0f - 1e-9f);

    float exp = 1.0f / (eta + 1.0f);
    float beta;
    if (u <= 0.5f) {
        beta = pow(2.0f * u, exp);
    } else {
        beta = pow(1.0f / (2.0f * (1.0f - u)), exp);
    }

    uint p1 = pair * 2u;
    uint p2 = pair * 2u + 1u;
    float g1 = parents[p1 * G + gene];
    float g2 = parents[p2 * G + gene];

    offspring[p1 * G + gene] = 0.5f * ((1.0f + beta) * g1 + (1.0f - beta) * g2);
    offspring[p2 * G + gene] = 0.5f * ((1.0f - beta) * g1 + (1.0f + beta) * g2);
}


// ------------------------------------------------------------------ //
// Rastrigin fitness (batch)
//
// One thread per individual.  Iterates over genome in a simple loop
// (no reduction needed for small G; for large G a threadgroup
// reduction would be faster — left as an exercise).
//
// Buffers
//   0  genomes  : float[P * G]
//   1  fitness  : float[P]
//   2  params   : uint4  x=(P, G, _, _)
//   3  A_buf    : float[1]
// ------------------------------------------------------------------ //

kernel void rastrigin_fitness(
    const device float*  genomes  [[ buffer(0) ]],
    device       float*  fitness  [[ buffer(1) ]],
    constant     uint4&  params   [[ buffer(2) ]],
    const device float*  A_buf    [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint P = params.x;
    uint G = params.y;
    float A = A_buf[0];

    if (gid >= P) return;

    float sum = A * float(G);
    for (uint g = 0u; g < G; g++) {
        float x = genomes[gid * G + g];
        sum += x * x - A * cos(2.0f * M_PI_F * x);
    }
    fitness[gid] = -sum;   // negate so higher = better
}


// ------------------------------------------------------------------ //
// Ackley fitness
// ------------------------------------------------------------------ //

kernel void ackley_fitness(
    const device float* genomes  [[ buffer(0) ]],
    device       float* fitness  [[ buffer(1) ]],
    constant     uint4& params   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint P = params.x;
    uint G = params.y;
    if (gid >= P) return;

    const float a = 20.0f, b = 0.2f, c = 2.0f * M_PI_F;
    float sum_sq = 0.0f, sum_cos = 0.0f;
    for (uint g = 0u; g < G; g++) {
        float x = genomes[gid * G + g];
        sum_sq  += x * x;
        sum_cos += cos(c * x);
    }
    float invG = 1.0f / float(G);
    float cost = -a * exp(-b * sqrt(sum_sq * invG))
                 - exp(sum_cos * invG)
                 + a + M_E_F;
    fitness[gid] = -cost;
}
