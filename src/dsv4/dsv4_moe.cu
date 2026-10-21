// src/dsv4/dsv4_moe.cu - the S2 MoE half-layer of the DeepSeek V4-Flash engine path (see dsv4_moe.h).
//
// Kernel roster (all NEW in this translation unit; no shared kernel file carries the V4 semantics):
//   dsv4_hc_pre_kernel        hc_pre (ffn side) + sinkhorn(20), dsv4ref.cpp:415-470's exact order
//   dsv4_gate_scores_kernel   sqrtsoftplus router scores (double-accumulated dots)
//   dsv4_route_kernel         tid2eid hash selection / top-6 on score+bias (bias selects only), weights
//                             from the pre-bias scores normalised * route_scale (double sum)
//   dsv4_nvfp4_gemv_kernel    the E3 grouped NVFP4 GEMV (36 B blocks of 4 UE4M3 scales + 32 E2M1 codes,
//                             dsv4ref.cpp:94-135's bit-level contract), batched over an expert's tokens
//   dsv4_swiglu_clamp_kernel  the E2 V4 variant: silu(min(g, 10)) * clamp(u, +-10) [* router weight]
//   dsv4_scatter_rows_kernel  [S7e] one entry's down output -> its comb6 cell (plain store, no atomics)
//   dsv4_moe_combine_kernel   [S7e] the k = 6 combine in a FIXED order (ascending expert id, shared last)
//   dsv4_hc_post_kernel       hc_post, dsv4ref.cpp:488-498
// reused: rms_norm_weighted (elementwise, S1-validated), native_quantize_q8_1 + native_q8_0_mmvq (the
// shared expert, the q36 production path; Q8_0 carries no sink so the plain entry points apply).
//
// The host fallback (STRATA_DSV4_MOE_CPU=1, the slice's CPU-pool path) mirrors dsv4ref.cpp's dequant +
// dot_fp32 + expert_apply exactly and doubles as the GPU kernels' numerical cross-check.

#include "dsv4_moe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/dsv4/attn.hpp"
#include "strata/dsv4/geometry.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"

#define CKM(x)                                                                                                    \
    do {                                                                                                          \
        cudaError_t e_ = (x);                                                                                     \
        if (e_ != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "strata-dsv4: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,       \
                         __LINE__);                                                                               \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

namespace strata::dsv4 {
namespace {

// ---------------------------------------------------------------------------
// The NVFP4 bit-level contract, dsv4ref.cpp:94-135 verbatim semantics (validated there against the
// official checkpoint): a [rows, cols] tensor is cols/64 blocks of 36 B per row: 4 UE4M3 scale bytes then
// 32 E2M1 code bytes; sub-block s covers 16 values, byte s*8+j's low nibble -> elem s*16+j, high nibble
// -> elem s*16+j+8; value = FP4_LUT[nib] * UE4M3(scale_s) * s2 (s2 the blob tail's per-role scale).
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ float dsv4_ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;   // 0x7f is NaN; ggml yields 0
    const int e = (x >> 3) & 0xf, m = x & 7;
#ifdef __CUDA_ARCH__
    if (e == 0) return ldexpf((float) m, -9);
    return ldexpf(1.0f + (float) m / 8.0f, e - 7);
#else
    if (e == 0) return std::ldexp((float) m, -9);
    return std::ldexp(1.0f + (float) m / 8.0f, e - 7);
#endif
}

__device__ __constant__ float kFP4Lut[16] = {0.f,  .5f, 1.f,  1.5f, 2.f,  3.f,  4.f,  6.f,
                                             -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// [S7b] exact arithmetic E2M1 decode: produces the identical bit pattern as kFP4Lut[n]
// (e>0: (2+m)*2^(e-2); e==0: m?0.5:0; sign bit 3) without a constant/shared memory load.
__device__ __forceinline__ float dsv4_fp4_dec(uint32_t nib) {
    const uint32_t s = (nib & 8u) << 28;
    const uint32_t e = (nib >> 1) & 3u, mm = nib & 1u;
    const uint32_t mag = e ? (((e + 126u) << 23) | (mm << 22)) : (mm ? 0x3F000000u : 0u);
    return __int_as_float(mag | s);
}

constexpr float kFP4LutHost[16] = {0.f,  .5f, 1.f,  1.5f, 2.f,  3.f,  4.f,  6.f,
                                   -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float dsv4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// hc_pre (the ffn half's fn/base/scale), dsv4ref.cpp:447-470.  One block per token, 256 threads; the sq
// and the 24 mix dots reduce in double (dsv4ref's reduction precision), sinkhorn(20) runs on thread 0 in
// the reference's exact iteration order (row softmax + eps; col-norm; 19 x (row-norm, col-norm)).
// ---------------------------------------------------------------------------
__global__ void dsv4_hc_mix_kernel(const float* __restrict__ x, const float* __restrict__ fn,
                                   int64_t n, double rms_eps, float* __restrict__ mixes) {
    // [S7b] one block per (token, mix dot): the old single-block kernel ran 24 sequential
    // full-width reductions per token (~0.26 ms/call at decode).  Each (token, dot) block keeps
    // the EXACT old per-dot accumulation order (thread-strided float partial -> double shared
    // tree 128..1), so every dot result is bit-identical.
    constexpr int kThreads = 256;
    const int64_t n4 = 4 * n;
    const float* xt = x + (int64_t) blockIdx.x * n4;
    const int r = blockIdx.y;
    __shared__ double red[kThreads];
    const int tid = threadIdx.x;
    float sq = 0.f;
    for (int64_t i = tid; i < n4; i += kThreads) sq += xt[i] * xt[i];
    red[tid] = (double) sq;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float rsqrt = (float) (1.0 / sqrt(red[0] / (double) n4 + rms_eps));
    const float* f = fn + (int64_t) r * n4;
    float acc = 0.f;
    for (int64_t i = tid; i < n4; i += kThreads) acc += f[i] * xt[i];
    red[tid] = (double) acc;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) mixes[(int64_t) blockIdx.x * 24 + r] = (float) red[0] * rsqrt;
}

__global__ void dsv4_hc_fin_kernel(const float* __restrict__ x, const float* __restrict__ base,
                                   const float* __restrict__ scale, const float* __restrict__ mixes,
                                   int64_t n, double hc_eps, float* __restrict__ y,
                                   float* __restrict__ post, float* __restrict__ comb) {
    // [S7b] serial sigmoid/sinkhorn on thread 0 + per-element float combine: bit-identical to the
    // old kernel's tail (mixes arrive via global memory with unchanged values).
    constexpr int kThreads = 256;
    const int64_t n4 = 4 * n;
    const float* xt = x + (int64_t) blockIdx.x * n4;
    const float* mx = mixes + (int64_t) blockIdx.x * 24;
    __shared__ float sh_pre[4];
    const int tid = threadIdx.x;
    __shared__ float sh_cb[16];
    __shared__ double sh_sum[4];
    const float eps = (float) hc_eps;
    if (tid < 4) {
        // sigmoid gates: each channel's math unchanged, one channel per thread
        sh_pre[tid] = dsv4_sigmoid(mx[tid] * scale[0] + base[tid]) + eps;
        post[(int64_t) blockIdx.x * 4 + tid] = 2.0f * dsv4_sigmoid(mx[4 + tid] * scale[1] + base[4 + tid]);
    }
    if (tid < 16) sh_cb[tid] = mx[8 + tid] * scale[2] + base[8 + tid];
    __syncthreads();
    // hc_sinkhorn20 (dsv4ref.cpp:415): row softmax + eps, col-norm, 19 x (row-norm, col-norm)
    if (tid < 4) {
        // row softmax: one row per thread; max/sum keep the serial j order
        float mxv = sh_cb[tid * 4];
        for (int j = 1; j < 4; ++j) mxv = fmaxf(mxv, sh_cb[tid * 4 + j]);
        for (int j = 0; j < 4; ++j) sh_cb[tid * 4 + j] = expf(sh_cb[tid * 4 + j] - mxv);
    }
    __syncthreads();
    if (tid < 4) {
        double sum = 0;
        for (int j = 0; j < 4; ++j) sum += sh_cb[tid * 4 + j];
        sh_sum[tid] = sum;
    }
    __syncthreads();
    if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / sh_sum[tid >> 2] + (double) eps);
    __syncthreads();
    for (int iter = 0; iter < 20; ++iter) {
        if (tid < 4) {  // col-norm: column tid, serial c order
            double s = 0;
            for (int c = 0; c < 4; ++c) s += sh_cb[c * 4 + tid];
            sh_sum[tid] = s;
        }
        __syncthreads();
        if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / (sh_sum[tid & 3] + (double) eps));
        __syncthreads();
        if (iter == 19) break;
        if (tid < 4) {  // row-norm: row tid, serial j order
            double s = 0;
            for (int j = 0; j < 4; ++j) s += sh_cb[tid * 4 + j];
            sh_sum[tid] = s;
        }
        __syncthreads();
        if (tid < 16) sh_cb[tid] = (float) (sh_cb[tid] / (sh_sum[tid >> 2] + (double) eps));
        __syncthreads();
    }
    if (tid < 16) comb[(int64_t) blockIdx.x * 16 + tid] = sh_cb[tid];
    __syncthreads();
    float* yt = y + (int64_t) blockIdx.x * n;
    for (int64_t i = tid; i < n; i += kThreads) {
        float acc = 0.f;
        for (int c = 0; c < 4; ++c) acc += sh_pre[c] * xt[(int64_t) c * n + i];
        yt[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Router scores: scores[j][e] = sqrt(softplus(x[j] . gate_w[e])), softplus the reference's (x > 20 ? x :
// log1p(exp(x))).  One block per token, 8 warps; a warp strides the experts, its 32 lanes the 4096-wide
// dot, accumulated in double (a flipped near-tie expert pick is the dominant replay-noise source; the
// double dot keeps the engine's scores at least as accurate as the host reference's).
// ---------------------------------------------------------------------------
__global__ void dsv4_gate_scores_kernel(const float* __restrict__ x, const uint16_t* __restrict__ gw,
                                        int64_t n, int ne, float* __restrict__ scores) {
    const float* xt = x + (int64_t) blockIdx.x * n;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // [S7b-step3] one warp per expert, experts spread over blockIdx.y blocks (the S7a loop ran all
    // ne experts on ONE block: 256 fp64 dots of length n on a single SM measured 51 ms/token on
    // L20, whose fp64 rate is 1:64).  Per-expert accumulation order is unchanged (lane stride 32 +
    // warp shuffle reduce), so the scores are bit-identical to the single-block version.
    const int e = blockIdx.y * 8 + warp;
    if (e < ne) {
        const uint16_t* wr = gw + (int64_t) e * n;   // raw bf16 rows (2 B/elem in the arena)
        double acc = 0;
        for (int64_t i = lane; i < n; i += 32)
            acc += (double) xt[i] * (double) __bfloat162float(__ushort_as_bfloat16(wr[i]));
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            float v = (float) acc;
            v = v > 20.0f ? v : log1pf(expf(v));
            scores[(int64_t) blockIdx.x * ne + e] = sqrtf(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Selection + weights, one thread per token.  Hash layers: idx = tid2eid[tok] (weights STILL gathered
// from the gate scores - MANIFEST trap 14).  Routed layers: top-NK by score + bias with the reference's
// tie-break (higher key first, then the smaller index; bias enters the selection only - trap 13).
// Weights: score / double(sum) * route_scale (dsv4ref.cpp:1170-1176).
// ---------------------------------------------------------------------------
__global__ void dsv4_route_kernel(const float* __restrict__ scores, const float* __restrict__ bias,
                                  const int32_t* __restrict__ tid2eid,
                                  const int32_t* __restrict__ tokens, int s, int ne, int nk,
                                  double route_scale, int32_t* __restrict__ idx,
                                  float* __restrict__ wgt) {
    // [S7b] one warp per token (the old kernel gave each token ONE thread: 6x256 serial
    // global-latency loads ~= 0.21 ms/call at decode).  Selection rule is bit-identical to the
    // serial scan: strictly-greater wins, ties keep the smallest expert id.  Lane-local ascending
    // scans with strict '>' preserve that inside a lane; the butterfly reduce breaks cross-lane
    // ties by smaller id.  nk <= 8 guaranteed by dsv4_moe_ws_new.
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= s) return;
    const float* sc = scores + (int64_t) j * ne;
    int sel[8];
    if (tid2eid) {
        const int64_t tok = tokens[j];
        for (int k = 0; k < nk; ++k) sel[k] = tid2eid[tok * nk + k];
    } else {
        const float* bs = bias;  // shared [ne] bias vector (not per token)
        for (int k = 0; k < nk; ++k) {
            float bv = -INFINITY;
            int be = ne;
            for (int e = lane; e < ne; e += 32) {
                bool dup = false;
                for (int q = 0; q < k; ++q)
                    if (sel[q] == e) { dup = true; break; }
                if (dup) continue;
                const float v = sc[e] + bs[e];
                if (v > bv) { bv = v; be = e; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oe = __shfl_xor_sync(0xffffffffu, be, off);
                if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
            }
            sel[k] = be;
        }
    }
    if (lane == 0) {
        double sum = 0;
        for (int k = 0; k < nk; ++k) sum += sc[sel[k]];
        for (int k = 0; k < nk; ++k) {
            idx[(int64_t) j * nk + k] = sel[k];
            wgt[(int64_t) j * nk + k] = (float) (sc[sel[k]] / sum * route_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// The E3 NVFP4 grouped GEMV: y[m][row] = s2 * (W_nvfp4[row] . x[m]), one 36 B block per lane.  cols is
// 4096 (gate/up) or 2048 (down), so a row's blocks are 64 or 32 - the threads-per-row group; 256 threads
// cover 4 (resp. 8) rows per block.  x row per batch item m comes from xbase + xidx[m] * xstride (the
// normed MoE input gathered per pair) or, with xidx == nullptr, from xbase + m * xstride (the down
// projection reading this slice's own hidden buffer).  The tail scale s2 folds at the end (one multiply;
// dsv4ref folds it per block scale - the difference is float rounding, ~1e-7 relative).
// ---------------------------------------------------------------------------
__global__ void dsv4_nvfp4_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ xbase,
                                       int64_t xstride, const int32_t* __restrict__ xidx, int cols,
                                       int rows, float s2, float* __restrict__ y) {
    extern __shared__ float sx[];
    const int tpr = cols >> 6;             // threads per row: 64 or 32
    const int rpb = 256 / tpr;             // rows per block: 4 or 8
    const int rloc = threadIdx.x / tpr;    // row within block
    const int lane = threadIdx.x % tpr;    // the 36 B block this thread owns
    const int row = blockIdx.x * rpb + rloc;
    const int64_t m = blockIdx.y;
    const float* xr = xbase + (xidx ? (int64_t) xidx[m] : m) * xstride;
    for (int i = threadIdx.x; i < cols; i += 256) sx[i] = xr[i];
    __syncthreads();
    if (row >= rows) return;   // grids below are exact multiples of rpb; this never splits a sync point
    const uint8_t* blk = w + ((int64_t) row * tpr + lane) * 36;
    const float d[4] = {dsv4_ue4m3(blk[0]), dsv4_ue4m3(blk[1]), dsv4_ue4m3(blk[2]), dsv4_ue4m3(blk[3])};
    const uint8_t* qs = blk + 4;
    const float* x0 = sx + lane * 64;
    float acc = 0.f;
    for (int s = 0; s < 4; ++s) {
        const float dv = d[s];
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            acc += kFP4Lut[byte & 0x0f] * dv * x0[s * 16 + j];
            acc += kFP4Lut[byte >> 4] * dv * x0[s * 16 + 8 + j];
        }
    }
    // reduce over the tpr lanes of this row (one or two warps)
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[8][2];
    if ((lane & 31) == 0) part[rloc][lane >> 5] = acc;
    __syncthreads();
    if (lane == 0) {
        float total = part[rloc][0];
        if (tpr == 64) total += part[rloc][1];
        y[m * rows + row] = total * s2;
    }
}

// ---------------------------------------------------------------------------
// S7a: THE GROUPED FORM OF THE E3 GEMV - one launch covers EVERY staged expert of a layer, not one launch
// per expert.  grid = (row blocks, max entries per group, groups); blockIdx.z picks the expert group, its
// blob base comes from a device pointer table (staging slots now, VRAM-cache/peer slots later - the kernel
// only ever sees a device address), the per-role tail scale s2 is read from the staged blob's own tail
// (the same 16 B the host used to memcpy out), and the batch item is the GLOBAL entry index
// ebase[z] + blockIdx.y so gate/up gather x rows through xidx (the entry's token) while down reads this
// slice's own hidden rows (xidx == nullptr).  The arithmetic is byte-for-byte the per-expert kernel's;
// blocks of a group with fewer entries than the grid's y extent exit before the shared load.
// ---------------------------------------------------------------------------
__global__ void dsv4_nvfp4_gemv_grouped_kernel(const uint8_t* const* __restrict__ blobs, int64_t role_off,
                                               int64_t tail_off, int s2i, const float* __restrict__ xbase,
                                               int64_t xstride, const int32_t* __restrict__ xidx,
                                               const int32_t* __restrict__ ebase, int cols, int rows,
                                               float* __restrict__ y) {
    // [S7b] weight rows staged in shared as one contiguous 9216 B, 16 B-aligned region
    // (cooperative uint4, 3 batched loads in flight per thread); qs unpacked from 8 LDS.32
    // into registers; fp4 values decoded arithmetically (no LUT loads).  x is read straight
    // from global as float4 (16 B-aligned: xbase/stride multiples established at the launch
    // sites): it is small and L1-resident across blocks, and dropping the sx stage frees
    // ~16 KB of shared, doubling resident blocks/SM.  Every per-thread VALUE and the
    // accumulation order are unchanged -> bit-identical output.
    extern __shared__ uint8_t sw[];
    const int z = blockIdx.z;
    const int64_t m = (int64_t) ebase[z] + blockIdx.y;
    if ((int64_t) blockIdx.y >= (int64_t) ebase[z + 1] - ebase[z]) return;   // padded y slot: no entry
    const int tpr = cols >> 6;             // threads per row: 64 or 32
    const int rpb = 256 / tpr;             // rows per block: 4 or 8
    const int rloc = threadIdx.x / tpr;    // row within block
    const int lane = threadIdx.x % tpr;    // the 36 B block this thread owns
    const int row = blockIdx.x * rpb + rloc;
    // [S7c] x arrives in the 2048-float-chunk lane layout (dsv4_xpose2048_rows): lane L's 64-float
    // slice is scattered as 16 float4 groups, group (s,v) at (s*4+v)*128 + L*4 inside the lane's
    // 32-lane half-chunk.  Reading group (s,v) this way makes a warp's 16 B requests contiguous
    // (4 L1 wavefronts instead of 32).  Same values, same FMA order -> bit-identical.
    const float4* xv = reinterpret_cast<const float4*>(
        xbase + (xidx ? (int64_t) xidx[m] : m) * xstride + (int64_t) (lane >> 5) * 2048);
    const int c31 = lane & 31;
    const uint4* wv = reinterpret_cast<const uint4*>(blobs[z] + role_off +
                                                     (int64_t) (blockIdx.x * rpb) * (tpr * 36));
    {
        uint4* dstv = reinterpret_cast<uint4*>(sw);
        const int v = threadIdx.x;
        const uint4 a = wv[v];
        uint4 b, c;
        if (v + 256 < 576) b = wv[v + 256];
        if (v + 512 < 576) c = wv[v + 512];
        dstv[v] = a;
        if (v + 256 < 576) dstv[v + 256] = b;
        if (v + 512 < 576) dstv[v + 512] = c;
    }
    __syncthreads();
    if (row >= rows) return;   // grids below are exact multiples of rpb; this never splits a sync point
    const uint8_t* blk = sw + ((int64_t) rloc * tpr + lane) * 36;
    const float d[4] = {dsv4_ue4m3(blk[0]), dsv4_ue4m3(blk[1]), dsv4_ue4m3(blk[2]), dsv4_ue4m3(blk[3])};
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 4);
    uint32_t qv[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) qv[v] = qw[v];
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float dv = d[s];
        // x slice loaded per sub-block (4 float4 live at a time; preloading all 16 blew the
        // register budget and tanked occupancy - measured regression, then fixed).
        float4 xs[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) xs[v] = xv[(s * 4 + v) * 32 + c31];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t byte = (qv[s * 2 + (j >> 2)] >> ((j & 3) * 8)) & 0xffu;
            const float4 xa = xs[j >> 2];
            const float xj = (j & 2) ? ((j & 1) ? xa.w : xa.z) : ((j & 1) ? xa.y : xa.x);
            const float4 xb = xs[2 + (j >> 2)];
            const float xj8 = (j & 2) ? ((j & 1) ? xb.w : xb.z) : ((j & 1) ? xb.y : xb.x);
            acc += dsv4_fp4_dec(byte & 0x0f) * dv * xj;
            acc += dsv4_fp4_dec(byte >> 4) * dv * xj8;
        }
    }
    // reduce over the tpr lanes of this row (one or two warps)
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float part[8][2];
    if ((lane & 31) == 0) part[rloc][lane >> 5] = acc;
    __syncthreads();
    if (lane == 0) {
        float total = part[rloc][0];
        if (tpr == 64) total += part[rloc][1];
        // the tail scale folds at the end exactly as in the per-expert kernel (one multiply)
        const float s2 = tail_off ? ((const float*) (blobs[z] + tail_off))[s2i] : 1.0f;
        y[m * rows + row] = total * s2;
    }
}

// ---------------------------------------------------------------------------
// The E2 V4 swiglu clamp variant (dsv4ref.cpp:1104-1113): h = silu(min(g, 10)) * clamp(u, -10, 10), the
// router weight (when given) multiplied into the hidden BEFORE the down projection (trap 15).  One block
// per batch item.
// ---------------------------------------------------------------------------
__global__ void dsv4_swiglu_clamp_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                         const float* __restrict__ wgt, int inter,
                                         float* __restrict__ h, int perm) {
    // [S7c] h is written in the 2048-float-chunk lane layout the down GEMV reads
    // (dsv4_xpose2048_rows): float4 j of the entry lands at (s*4+v)*32 + L of its chunk.  The
    // arithmetic per element is unchanged (same expression, same order) -> bit-identical.
    const int64_t base = (int64_t) blockIdx.x * inter;
    const float w = wgt ? wgt[blockIdx.x] : 1.0f;
    const float4* g4 = reinterpret_cast<const float4*>(g + base);
    const float4* u4 = reinterpret_cast<const float4*>(u + base);
    float4* h4 = reinterpret_cast<float4*>(h + base);
    const int n4 = inter >> 2;
    for (int j = threadIdx.x; j < n4; j += blockDim.x) {
        const float4 gg = g4[j], uu = u4[j];
        float4 o;
        {
            const float gv = fminf(gg.x, 10.0f);
            const float uv = fminf(fmaxf(uu.x, -10.0f), 10.0f);
            o.x = (gv / (1.0f + expf(-gv))) * uv * w;
        }
        {
            const float gv = fminf(gg.y, 10.0f);
            const float uv = fminf(fmaxf(uu.y, -10.0f), 10.0f);
            o.y = (gv / (1.0f + expf(-gv))) * uv * w;
        }
        {
            const float gv = fminf(gg.z, 10.0f);
            const float uv = fminf(fmaxf(uu.z, -10.0f), 10.0f);
            o.z = (gv / (1.0f + expf(-gv))) * uv * w;
        }
        {
            const float gv = fminf(gg.w, 10.0f);
            const float uv = fminf(fmaxf(uu.w, -10.0f), 10.0f);
            o.w = (gv / (1.0f + expf(-gv))) * uv * w;
        }
        const int wq = j & 511;                        // float4 index within the 2048-float chunk
        const int64_t dq = perm ? (int64_t) (j >> 9) * 512 + ((wq >> 2) & 3) * 128 + (wq & 3) * 32 +
                                         (wq >> 4)
                               : j;
        h4[dq] = o;
    }
}

/// comb6[slot[m]][i] = src[m][i] - the scatter step of the k = 6 combine.  `slot` (token * nk + the
/// entry's rank by ascending expert id) is computed on the host, so every entry owns a distinct cell: plain
/// stores, no atomics, and the per-token accumulation order downstream is the reference's.
__global__ void dsv4_scatter_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ slot,
                                         float* __restrict__ comb, int64_t n) {
    const float* sr = src + (int64_t) blockIdx.x * n;
    float* dst = comb + (int64_t) slot[blockIdx.x] * n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = sr[i];
}

/// bout[t] = ((((c[0] + c[1]) + c[2]) + c[3]) + c[4]) + c[5], then + the shared expert (dsv4ref.cpp:1204:
/// experts in ascending expert id, the shared expert unweighted and last).  Fixed order, no atomics -> the
/// same bits on every run.
__global__ void dsv4_moe_combine_kernel(const float* __restrict__ comb, const float* __restrict__ sout,
                                        float* __restrict__ bout, int64_t n, int nk) {
    const float* ct = comb + (int64_t) blockIdx.x * nk * n;
    const float* st = sout + (int64_t) blockIdx.x * n;
    float* bt = bout + (int64_t) blockIdx.x * n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = ct[i];
        for (int k = 1; k < nk; ++k) acc += ct[(int64_t) k * n + i];
        bt[i] = acc + st[i];
    }
}

/// The comb6 cell of the entry (token, slot k) whose expert id is e: token * nk + its rank among the
/// token's slots by ascending expert id (ties by slot index).  Host side, nk (6) comparisons.
inline int32_t dsv4_comb_cell(const int32_t* idx, int64_t tok, int nk, int32_t e, int64_t k) {
    int32_t rank = 0;
    for (int64_t j = 0; j < nk; ++j) {
        const int32_t ej = idx[tok * nk + j];
        if (ej < e || (ej == e && j < k)) ++rank;
    }
    return (int32_t) (tok * nk + rank);
}

// ---------------------------------------------------------------------------
// hc_post (dsv4ref.cpp:488-498): out[c][i] = post[c] * bout[i] + sum_j comb[c*4+j] * h[j][i], j-ordered
// float adds.  Reads the old h, writes a second buffer (the engine copies it back over h).
// ---------------------------------------------------------------------------
__global__ void dsv4_hc_post_kernel(const float* __restrict__ h, const float* __restrict__ bout,
                                    const float* __restrict__ post, const float* __restrict__ comb,
                                    int64_t n, float* __restrict__ out) {
    const int64_t b = blockIdx.x;
    const float* hb = h + b * 4 * n;
    const float* xb = bout + b * n;
    const float* pb = post + b * 4;
    const float* cb = comb + b * 16;
    float* ob = out + b * 4 * n;
    for (int c = 0; c < 4; ++c)
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            float acc = pb[c] * xb[i];
            for (int j = 0; j < 4; ++j) acc += cb[c * 4 + j] * hb[(int64_t) j * n + i];
            ob[(int64_t) c * n + i] = acc;
        }
}

// ---------------------------------------------------------------------------
// The host fallback (STRATA_DSV4_MOE_CPU=1): dsv4ref.cpp's exact host arithmetic - dequant_nvfp4 +
// dot_fp32 (float, 8-way) + the E2 swiglu + Q8_0 dequant for the shared expert.
// ---------------------------------------------------------------------------
float dot_fp32_host(const float* a, const float* b, int64_t n) {
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
    }
    float acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

void dequant_nvfp4_host(const uint8_t* src, float s2, int64_t rows, int64_t cols, float* dst) {
    const int64_t nb = cols / 64;
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* brow = src + r * nb * 36;
        float* orow = dst + r * cols;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = brow + b * 36;
            const float d[4] = {dsv4_ue4m3(blk[0]) * s2, dsv4_ue4m3(blk[1]) * s2,
                                dsv4_ue4m3(blk[2]) * s2, dsv4_ue4m3(blk[3]) * s2};
            const uint8_t* qs = blk + 4;
            float* o = orow + b * 64;
            for (int s = 0; s < 4; ++s)
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qs[s * 8 + j];
                    o[s * 16 + j] = kFP4LutHost[byte & 0x0f] * d[s];
                    o[s * 16 + 8 + j] = kFP4LutHost[byte >> 4] * d[s];
                }
        }
    }
}

float fp16_to_fp32_host(uint16_t h) {
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const float sign = (h & 0x8000) ? -1.0f : 1.0f;
    if (e == 0) return sign * std::ldexp((float) m, -24);    // subnormal
    if (e == 31) return sign * std::numeric_limits<float>::infinity();
    return sign * std::ldexp(1024.0f + (float) m, e - 25);
}

void dequant_q8_0_host(const uint8_t* src, int64_t n, float* dst) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        uint16_t dh;
        std::memcpy(&dh, src + b * 34, 2);
        const float d = fp16_to_fp32_host(dh);
        const int8_t* qs = (const int8_t*) (src + b * 34 + 2);
        float* o = dst + b * 32;
        for (int j = 0; j < 32; ++j) o[j] = (float) qs[j] * d;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The workspace
// ---------------------------------------------------------------------------
struct Dsv4MoeWs {
    int64_t s = 0, n = 0, ne = 0, nk = 0, inter = 0;
    bool cpu_pool = false;
    uint64_t blob_bytes = 0;
    const char* debug_dir = nullptr;
    // per-token
    float *y = nullptr, *yt = nullptr, *bout = nullptr, *post = nullptr, *comb = nullptr, *h2 = nullptr;
    float* mixes = nullptr;  // [s][24] hc_pre dot results (kernel A -> B)
    float *scores = nullptr, *wgt = nullptr;
    int32_t* idx = nullptr;
    // expert staging + per-expert batch (m <= s * nk)
    uint8_t* blob = nullptr;
    float *gate = nullptr, *up = nullptr, *ph = nullptr, *pout = nullptr, *pw = nullptr;
    int32_t* tok = nullptr;
    // [S7e] the deterministic combine: comb6[(token, rank)][n] receives each entry's down output, so the
    // per-token sum runs in a FIXED order (ascending expert id, shared expert last, dsv4ref.cpp:1204) with
    // no atomics.  rank counts the token's slots by ascending expert id, ties by slot index (so even a
    // duplicated tid2eid entry gets its own cell).
    float* comb6 = nullptr;              ///< [s][nk][n]
    int32_t* d_slot = nullptr;           ///< device: entry -> comb6 cell (token * nk + rank)
    // [S7a] grouped staging: chunk blobs staged side by side, one grouped launch per role per chunk
    int chunk = 0;                       ///< staged blobs per grouped pass (STRATA_DSV4_MOE_CHUNK, dflt 8)
    uint8_t* blob_stage = nullptr;       ///< chunk * blob_bytes
    uint8_t** d_ptrs = nullptr;          ///< device: chunk blob addresses (staging/cache/peer slots)
    int32_t* d_ebase = nullptr;          ///< device: chunk + 1 entry offsets per group
    // [S7e] one-upload arena: the chunk's table (ptrs | ebase | tok | pw | comb cell) is packed into
    // ONE host buffer and crosses as a single H2D copy; the kernel arguments are re-derived by adding
    // each section's offset inside `tab`.  Nine small memcpys per layer became two.
    uint8_t* tab = nullptr;
    std::vector<uint8_t> h_tab;          ///< host packing buffer for `tab` (fallback: no pin)
    /// [S8] A pinned upload arena plus the event of the last copy queued out of it.  A pageable
    /// source makes cudaMemcpyAsync effectively synchronous (the driver bounces it), so reusing one
    /// staging buffer was safe with `h_tab`; a PINNED source is read by the DMA after the call
    /// returns, so the host must not touch those bytes again until the copy has completed.  `take`
    /// waits on the arena's event before handing the buffer out again, and the chunked fallback
    /// rotates two arenas so that wait is normally already satisfied.
    struct PinTab {
        uint8_t* p = nullptr;
        size_t cap = 0;
        cudaEvent_t ev = nullptr;
        bool live = false;
    };
    PinTab pin[2];                       ///< device-0 table arenas, alternating per upload
    int pin_turn = 0;
    PinTab pin_peer;                     ///< peer table arena (copied on the peer stream)
    // shared expert
    void *q8 = nullptr, *q8h = nullptr;
    float *sg = nullptr, *su = nullptr, *sh = nullptr, *sout = nullptr;
    // host mirrors
    std::vector<int32_t> h_idx;
    std::vector<float> h_wgt;
    std::vector<float> h_x, h_bout;
    // [S7a] host staging for the grouped pass
    std::vector<int32_t> h_groups;       ///< unique experts of the layer, sorted (the reference's order)
    std::vector<int32_t> h_ebase;        ///< group -> first entry (size groups + 1)
    std::vector<int32_t> h_ebase_c;      ///< chunk-local ebase upload buffer (chunk + 1)
    std::vector<int32_t> h_tokg;         ///< entry -> token
    std::vector<float> h_pwg;            ///< entry -> router weight
    std::vector<uint8_t*> h_ptrs;        ///< group -> device blob address for this chunk
    // [S8] group/entry-table build scratch, sized once at ws_new: a presence bitmap over the routed
    // experts (emitted ascending = the reference's accumulation order), the per-expert entry counts,
    // a counting-sort fill cursor, and the "evaluates on device 0" membership flag.  The S7a build
    // re-sorted a fresh vector every call and rescanned every group against every routing entry.
    std::vector<uint64_t> h_mask;        ///< (ne + 63) / 64 words: expert ids routed this call
    std::vector<int32_t> h_cnt;          ///< ne: routing entries per expert id
    std::vector<int32_t> h_cur;          ///< ne: counting-sort fill cursor (-1: not in this table)
    std::vector<uint8_t> h_in0;          ///< ne: this expert's entries go to the device-0 table
    // [S7a-2] predictive warm: the previous pass's routing per layer; the file tier's I/O workers pread
    // the layers just ahead into the page cache while this layer computes (STRATA_IO_PREFETCH).
    int n_layers = 0;                    ///< set by the engine after ws_new (wrap for the warm window)
    int pf_depth = 2;                    ///< warm window in layers (STRATA_IO_PREFETCH_DEPTH)
    std::vector<std::vector<int64_t>> h_prev_route;  ///< layer -> the last pass's routing (nk ids)
    // [S7a-3] the VRAM expert tier (owned by the engine; either may be null)
    strata::core::ExpertCache* cache[2] = {nullptr, nullptr};
    int64_t cache_hits = 0, cache_misses = 0;
    // [S7a-3b] peer compute: device 1 evaluates ITS home experts locally (its own VRAM at full
    // bandwidth) and only the small per-entry outputs cross PCIe.  Measured on this box: a kernel on
    // device 0 reading device 1's memory over P2P collapses to ~0.9 GiB/s inside the GEMV (a plain
    // streaming probe still gets 9.5 GiB/s), so the "dev0 kernel reads dev1 slots" layout is not
    // viable; computing on the peer and DMA-ing entries*n*4 bytes back is.
    struct PeerWs {
        bool active = false;
        cudaStream_t stream = nullptr;     // created on device 1
        cudaEvent_t ev_y = nullptr;        // device 0: y is ready to copy
        cudaEvent_t ev_done = nullptr;     // device 1: pout is ready to bring back
        float* y1 = nullptr;               // s*n copy of the MoE input on device 1
        float *gate = nullptr, *up = nullptr, *ph = nullptr, *pout = nullptr, *pw = nullptr;
        int32_t* tok = nullptr;
        uint8_t** d_ptrs = nullptr;        // group -> blob address in device-1 VRAM
        int32_t* d_ebase = nullptr;        // groups + 1 entry offsets
        uint8_t* tab = nullptr;            // [S7e] one-upload arena for the peer groups (device 1)
    } peer;
    // host staging for the peer pass (one grouped pass covers every device-1 group; no chunking)
    std::vector<int32_t> h_groups0;
    std::vector<int32_t> h_groups1;
    std::vector<uint8_t*> h_ptrs1;
    std::vector<int32_t> h_ebase1, h_tokg1;
    std::vector<float> h_pwg1;
    std::vector<int32_t> h_slotg, h_slotg1;   ///< [S7e] entry -> comb6 cell (device-0 groups / peer)
    // [S7b] raw routing dump (STRATA_DSV4_ROUTE_DUMP) + host-phase timing (STRATA_DSV4_TIME).
    // Both are instrumentation only: no behavior change when the env vars are unset.
    FILE* route_dump = nullptr;        // records: header RTE1 + per call {layer, s, s*nk ids}
    bool time_on = false;
    bool overlap_shared = false;       // [S7e] shared expert on a side stream (default on)
    double tm_pre = 0, tm_sync = 0, tm_group = 0, tm_chunks = 0, tm_tail = 0;
    int64_t moe_calls = 0;
    // [S7b] heat-driven admission: decayed per-(layer, expert) routing counts; a miss whose layer
    // quota is full swaps out the coldest resident when clearly hotter (STRATA_DSV4_EVICT_GAIN,
    // default 1.25).  Empty table (no profile seed): eviction off, first-touch only (S7a policy).
    std::vector<float> heat;           // n_layers * ne, indexed [layer * ne + expert]
    double evict_gain = 1.25;
    int64_t evictions = 0;
    // [S7b] GPU-section events (STRATA_DSV4_TIME): stream-time split of the GPU branch
    cudaEvent_t tev[10] = {};
    // [S7e] the shared expert reads the normed input but never the routing decision, so it runs on its
    // own stream: its ~50 us of GPU work then covers the host gap the route D2H sync leaves behind.
    cudaStream_t sh_stream = nullptr;   // side stream (device 0, non-blocking)
    cudaEvent_t ev_y = nullptr;         // y ready (main) -> sh_stream may start
    cudaEvent_t ev_sh = nullptr;        // shared expert done -> the combine's wait
    cudaEvent_t sev[2] = {};            // shared-expert GPU section (both ends on sh_stream)
    double tg_chunks = 0, tg_peer = 0, tg_shared = 0, tg_hcpost = 0, tg_route = 0;
    double tg_hcpre = 0, tg_rms = 0, tg_gate = 0, tg_topk = 0;
    // [S9] cold-blob I/O attribution (STRATA_DSV4_TIME): host wall time of the blob assembly
    // (experts.blob -> pread / memcpy from the mapping) and of the pageable H2D that follows it, plus
    // the single-table path's own pack / table-upload / launch split.  Instrumentation only.
    double tm_cold_read = 0, tm_cold_fill = 0, tm_cold_staged = 0;
    double tm_pack = 0, tm_tabh2d = 0, tm_lanch = 0;
    int64_t cold_n = 0, cold_staged_n = 0, cold_pipe = 0;
    /// [S9] The cold-blob pipeline: a small ring of page-locked blob-sized slots.  A cold expert is
    /// read into a ring slot, then copied to its cache slot by a DMA queued on a BYPASS stream, and
    /// the stream that runs the kernels only waits on that copy's event.  The old path read the blob
    /// and then copied it from pageable memory on the kernel stream, which blocks the host for the
    /// whole bounce (~2.1 ms per 13.5 MiB blob, 6.4 GB/s effective).  Nothing here changes which bytes
    /// land where: the same blob is copied to the same cache slot, the same kernels read it.
    struct ColdPipe {
        bool on = false;
        int n = 0;                        ///< ring slots (STRATA_DSV4_COLDPIPE_SLOTS, default 6)
        size_t bytes = 0;
        uint8_t* host[8] = {};            ///< pinned, one blob each
        cudaEvent_t ev0[8] = {};          ///< device 0: the copy OUT of host[i] finished
        cudaEvent_t ev1[8] = {};          ///< device 1: same, recorded on the peer device
        int state[8] = {};                ///< 0 free, 1 the reader owns it, 2 ready, 3 DMA queued
        int sdev[8] = {};                 ///< device the queued DMA targets (state 3)
        int64_t key[8] = {};              ///< layer * 256 + expert of the blob in the slot
        int next = 0;                     ///< start of the round-robin scan
        int64_t queued = 0, fell_back = 0;
        int threads = 4;                  ///< reader threads per blob (STRATA_DSV4_COLDPIPE_THREADS)
    };
    ColdPipe cp;
};

// [S8] A/B switch for the pinned upload arenas (0 = the pre-S8 pageable staging path)
static const bool kBisNoPin = std::getenv("STRATA_DSV4_MOE_NOPIN") != nullptr;

// [S8] hand out a pinned upload arena only once the DMA that last read it has completed; returns
// nullptr when the arena is unusable (the caller then falls back to a pageable vector, whose copy is
// synchronous, i.e. the pre-S8 behaviour).
static uint8_t* dsv4_pintab_take(Dsv4MoeWs::PinTab& t, size_t need) {
    if (t.p == nullptr || t.ev == nullptr || need > t.cap) return nullptr;
    if (t.live) {
        if (t.ev != nullptr && cudaEventSynchronize(t.ev) != cudaSuccess) {
            (void) cudaGetLastError();
        }
        t.live = false;
    }
    return t.p;
}

// remember that `H` is the source of a copy just queued on `st` (a pageable fallback needs nothing)
static void dsv4_pintab_queued(Dsv4MoeWs::PinTab& t, uint8_t* H, cudaStream_t st) {
    if (H == nullptr || H != t.p) return;
    if (cudaEventRecord(t.ev, st) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFreeHost(t.p);
        t.p = nullptr;
        t.cap = 0;
        return;
    }
    t.live = true;
}

Dsv4MoeWs* dsv4_moe_ws_new(const Dsv4Geometry& g, int64_t s, std::string& err) {
    if (s <= 0 || s > 4096 || g.n_expert_used > 8) {
        err = "dsv4_moe: bad token count or topk";
        return nullptr;
    }
    Dsv4MoeWs* ws = new Dsv4MoeWs();
    ws->s = s;
    ws->n = g.n_embd;
    ws->ne = g.n_expert;
    ws->nk = g.n_expert_used;
    ws->inter = g.n_ff;
    // [S7c] the grouped NVFP4 GEMV reads x/ph in the 2048-float-chunk lane layout
    // (dsv4_xpose2048_rows); every dsv4 shape (n 4096, inter 2048) qualifies.
    if ((ws->n % 2048) != 0 || (ws->inter % 2048) != 0) {
        err = "dsv4_moe: S7c gemv layout needs n and n_ff multiples of 2048";
        delete ws;
        return nullptr;
    }
    ws->cpu_pool = std::getenv("STRATA_DSV4_MOE_CPU") != nullptr;
    ws->debug_dir = std::getenv("STRATA_DSV4_MOE_DEBUG");
    if (const char* renv = std::getenv("STRATA_DSV4_ROUTE_DUMP")) {
        ws->route_dump = std::fopen(renv, "wb");
        if (!ws->route_dump)
            std::fprintf(stderr, "dsv4_moe: STRATA_DSV4_ROUTE_DUMP %s is not writable\n", renv);
    }
    ws->time_on = std::getenv("STRATA_DSV4_TIME") != nullptr;
    if (ws->time_on)
        for (cudaEvent_t& ev : ws->tev) {
            if (cudaEventCreate(&ev) != cudaSuccess) ev = nullptr;
        }
    // [S7e] the shared-expert side stream (STRATA_DSV4_SHARED_STREAM=0 turns the overlap off, which is
    // the pre-S7e scheduling: same values either way, only the fill of the host gap differs)
    if (const char* ssenv = std::getenv("STRATA_DSV4_SHARED_STREAM"); ssenv == nullptr || *ssenv != '0')
        ws->overlap_shared = true;
    if (ws->overlap_shared) {
        if (cudaStreamCreateWithFlags(&ws->sh_stream, cudaStreamNonBlocking) != cudaSuccess) {
            err = "dsv4_moe: shared-expert side stream";
            delete ws;
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&ws->ev_y, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ws->ev_sh, cudaEventDisableTiming) != cudaSuccess) {
            err = "dsv4_moe: shared-expert events";
            delete ws;
            return nullptr;
        }
        if (ws->time_on)
            for (cudaEvent_t& ev : ws->sev) {
                if (cudaEventCreate(&ev) != cudaSuccess) ev = nullptr;
            }
    }
    if (const char* genv = std::getenv("STRATA_DSV4_EVICT_GAIN")) {
        ws->evict_gain = std::atof(genv);
        if (ws->evict_gain < 1.0) ws->evict_gain = 1.0;
    }
    const int64_t n = ws->n, inter = ws->inter, m = s * ws->nk;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    ws->blob_bytes = lay.max_blob;
    CKM(cudaMalloc(&ws->y, (size_t) (s * n) * 4));
    CKM(cudaMalloc(&ws->yt, (size_t) (s * n) * 4));   // [S7c] transposed MoE input for the GEMV
    CKM(cudaMalloc(&ws->bout, (size_t) (s * n) * 4));
    CKM(cudaMalloc(&ws->post, (size_t) (s * 4) * 4));
    CKM(cudaMalloc(&ws->mixes, (size_t) (s * 24) * 4));
    CKM(cudaMalloc(&ws->comb, (size_t) (s * 16) * 4));
    CKM(cudaMalloc(&ws->h2, (size_t) (s * 4 * n) * 4));
    CKM(cudaMalloc(&ws->scores, (size_t) (s * ws->ne) * 4));
    CKM(cudaMalloc(&ws->wgt, (size_t) m * 4));
    CKM(cudaMalloc(&ws->idx, (size_t) m * 4));
    CKM(cudaMalloc(&ws->blob, (size_t) ws->blob_bytes));
    {
        const char* cenv = std::getenv("STRATA_DSV4_MOE_CHUNK");
        ws->chunk = cenv ? std::atoi(cenv) : 8;
        if (ws->chunk < 1) ws->chunk = 1;
        if (ws->chunk > 32) ws->chunk = 32;   // 32 x 14.16 MB = 453 MB of staging at most
        CKM(cudaMalloc(&ws->blob_stage, (size_t) ws->chunk * ws->blob_bytes));
        CKM(cudaMalloc(&ws->d_ptrs, (size_t) ws->chunk * sizeof(uint8_t*)));
        CKM(cudaMalloc(&ws->d_ebase, (size_t) (ws->chunk + 1) * 4));
        {
            // [S8] the packed table arena is sized for the WHOLE layer (every group's blob pointer
            // plus every entry's tok/pw/comb cell), so one table can serve one launch set; the
            // chunk-sized path below only ever uses a prefix of it.
            const size_t gmax = (size_t) ws->s * (size_t) ws->nk;
            CKM(cudaMalloc(&ws->tab, (size_t) 64 + gmax * sizeof(uint8_t*) + (gmax + 1) * 4 +
                                         3 * gmax * 4));
            ws->h_ptrs.resize(gmax);
            ws->h_ebase_c.resize((size_t) ws->chunk + 1);
            // pinned host arena the table is packed into: a pageable cudaMemcpyAsync makes the
            // driver stage through its own bounce buffer and stalls the calling thread, which showed
            // up as ~0.1 ms per chunk of host time (each layer has one copy per chunk).
            // [S8] two device-0 arenas (the chunked fallback rotates them) and one for the peer:
            // the peer copy is queued on another stream, so it must never share a host buffer with
            // a device-0 upload.  Portable, so either device context can address them.
            const size_t cap0 =
                (size_t) 64 + gmax * sizeof(uint8_t*) + (gmax + 1) * 4 + 3 * gmax * 4;
            const size_t cap1 = (size_t) 64 + (size_t) ws->ne * sizeof(uint8_t*) +
                                ((size_t) ws->ne + 1) * 4 + 3 * gmax * 4;
            Dsv4MoeWs::PinTab* tabs[3] = {&ws->pin[0], &ws->pin[1], &ws->pin_peer};
            const size_t caps[3] = {cap0, cap0, cap1};
            bool pin_ok = !kBisNoPin;
            for (int t = 0; pin_ok && t < 3; ++t) {
                if (cudaHostAlloc((void**) &tabs[t]->p, caps[t], cudaHostAllocPortable) !=
                    cudaSuccess) {
                    (void) cudaGetLastError();
                    tabs[t]->p = nullptr;
                    tabs[t]->cap = 0;
                    pin_ok = false;
                } else {
                    tabs[t]->cap = caps[t];
                }
            }
            // the upload events belong to the device whose stream they are recorded on: the two
            // device-0 arenas get theirs here, the peer arena gets its own in ws_set_cache (device 1)
            if (pin_ok) {
                for (int t = 0; t < 2; ++t) {
                    if (cudaEventCreateWithFlags(&tabs[t]->ev, cudaEventDisableTiming) !=
                        cudaSuccess) {
                        (void) cudaGetLastError();
                        pin_ok = false;
                    }
                }
            }
            if (!pin_ok) {
                for (int t = 0; t < 3; ++t) {
                    if (tabs[t]->p != nullptr) cudaFreeHost(tabs[t]->p);
                    tabs[t]->p = nullptr;
                    tabs[t]->cap = 0;
                }
                std::fprintf(stderr, "dsv4_moe: pinned table arenas unavailable; using pageable "
                                     "staging\n");
            }
        }
        if (const char* denv = std::getenv("STRATA_IO_PREFETCH_DEPTH")) {
            ws->pf_depth = std::atoi(denv);
            if (ws->pf_depth < 1) ws->pf_depth = 1;
            if (ws->pf_depth > 8) ws->pf_depth = 8;
        }
    }
    // [S9] cold-blob pipeline (device 0 side): a pinned ring of blob-sized slots plus the bypass
    // stream and events the copies out of them are queued on.  STRATA_DSV4_COLDPIPE=0 keeps the old
    // in-line pageable fill; anything that fails to allocate turns the pipeline off the same way.
    {
        const char* cpenv = std::getenv("STRATA_DSV4_COLDPIPE");
        // [S9] OFF by default: measured 2-3 ms/pass SLOWER than the in-line pageable copy on both
        // cards (R=1 41.0 -> 44.0, R=6 128.0 -> 130.5), because the pass is GPU-bound and the host's
        // pageable copy already overlaps with it.  Turn it on with STRATA_DSV4_COLDPIPE=1.
        ws->cp.on = cpenv != nullptr && std::atoi(cpenv) != 0;
        int nslots = 4;
        if (const char* nenv = std::getenv("STRATA_DSV4_COLDPIPE_SLOTS")) nslots = std::atoi(nenv);
        if (nslots < 2) nslots = 2;
        if (nslots > 8) nslots = 8;
        if (const char* tenv = std::getenv("STRATA_DSV4_COLDPIPE_THREADS"))
            ws->cp.threads = std::atoi(tenv);
        if (ws->cp.threads < 1) ws->cp.threads = 1;
        if (ws->cp.threads > 8) ws->cp.threads = 8;
        if (ws->cp.on) {
            ws->cp.bytes = (size_t) ws->blob_bytes;
            bool ok = true;
            int i = 0;
            for (; ok && i < nslots; ++i)
                ok = cudaHostAlloc((void**) &ws->cp.host[i], ws->cp.bytes, cudaHostAllocPortable) ==
                         cudaSuccess &&
                     cudaEventCreateWithFlags(&ws->cp.ev0[i], cudaEventDisableTiming) == cudaSuccess;
            ws->cp.n = ok ? nslots : 0;
            ws->cp.on = ok;
            if (!ok) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "dsv4_moe: cold-blob pipeline unavailable; using the in-line "
                                     "pageable copy\n");
            }
        }
    }
    CKM(cudaMalloc(&ws->gate, (size_t) (m * inter) * 4));
    CKM(cudaMalloc(&ws->up, (size_t) (m * inter) * 4));
    CKM(cudaMalloc(&ws->ph, (size_t) (m * inter) * 4));
    CKM(cudaMalloc(&ws->pout, (size_t) (m * n) * 4));
    CKM(cudaMalloc(&ws->pw, (size_t) m * 4));
    CKM(cudaMalloc(&ws->tok, (size_t) m * 4));
    CKM(cudaMalloc(&ws->q8, (size_t) (s * (n / 32) * 36)));
    CKM(cudaMalloc(&ws->q8h, (size_t) (s * (inter / 32) * 36)));
    CKM(cudaMalloc(&ws->sg, (size_t) (s * inter) * 4));
    CKM(cudaMalloc(&ws->su, (size_t) (s * inter) * 4));
    CKM(cudaMalloc(&ws->sh, (size_t) (s * inter) * 4));
    CKM(cudaMalloc(&ws->sout, (size_t) (s * n) * 4));
    CKM(cudaMalloc(&ws->comb6, (size_t) (m * n) * 4));   // [S7e] the deterministic combine buffer
    CKM(cudaMalloc(&ws->d_slot, (size_t) m * 4));
    ws->h_idx.resize((size_t) m);
    ws->h_wgt.resize((size_t) m);
    ws->h_x.resize((size_t) (s * n));
    ws->h_bout.resize((size_t) (s * n));
    // [S8] size every host staging buffer the per-layer table build touches ONCE; the per-call
    // clear()/resize() below then never allocates.  Tables are bounded by ne groups / m entries.
    ws->h_groups.reserve((size_t) ws->ne);
    ws->h_ebase.reserve((size_t) ws->ne + 1);
    ws->h_ebase1.reserve((size_t) ws->ne + 1);
    ws->h_groups0.reserve((size_t) ws->ne);
    ws->h_groups1.reserve((size_t) ws->ne);
    ws->h_ptrs1.reserve((size_t) ws->ne);
    ws->h_tokg.reserve((size_t) m);
    ws->h_pwg.reserve((size_t) m);
    ws->h_slotg.reserve((size_t) m);
    ws->h_tokg1.reserve((size_t) m);
    ws->h_pwg1.reserve((size_t) m);
    ws->h_slotg1.reserve((size_t) m);
    ws->h_mask.resize((size_t) ((ws->ne + 63) / 64));
    ws->h_cnt.resize((size_t) ws->ne);
    ws->h_cur.resize((size_t) ws->ne);
    ws->h_in0.resize((size_t) ws->ne);
    return ws;
}

void dsv4_moe_ws_set_layers(Dsv4MoeWs* ws, int n_layers) {
    if (!ws) return;
    ws->n_layers = n_layers;
    if (ws->route_dump) {
        std::fwrite("RTE1", 1, 4, ws->route_dump);
        const uint32_t hdr[3] = {(uint32_t) n_layers, (uint32_t) ws->ne, (uint32_t) ws->nk};
        std::fwrite(hdr, 4, 3, ws->route_dump);
        std::fflush(ws->route_dump);
    }
}

void dsv4_moe_ws_set_cache(Dsv4MoeWs* ws, strata::core::ExpertCache* c0,
                           strata::core::ExpertCache* c1) {
    if (!ws) return;
    ws->cache[0] = c0;
    ws->cache[1] = c1;
    if (c1 != nullptr && !ws->peer.active) {
        // one-time peer workspace on device 1 (idempotent; set_cache runs once per pass)
        const int64_t m = ws->s * ws->nk, n = ws->n, inter = ws->inter;
        if (cudaSetDevice(1) == cudaSuccess) {
            Dsv4MoeWs::PeerWs& p = ws->peer;
            // ev_done lives on device 1 (recorded on the peer stream); ev_y lives on device 0
            // (recorded on the engine stream once y is final) - events are single-device objects
            bool ok = cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking) == cudaSuccess &&
                      cudaEventCreateWithFlags(&p.ev_done, cudaEventDisableTiming) == cudaSuccess &&
                      cudaMalloc(&p.y1, (size_t) (ws->s * n) * 4) == cudaSuccess &&
                      cudaMalloc(&p.gate, (size_t) (m * inter) * 4) == cudaSuccess &&
                      cudaMalloc(&p.up, (size_t) (m * inter) * 4) == cudaSuccess &&
                      cudaMalloc(&p.ph, (size_t) (m * inter) * 4) == cudaSuccess &&
                      cudaMalloc(&p.pout, (size_t) (m * n) * 4) == cudaSuccess &&
                      cudaMalloc(&p.pw, (size_t) m * 4) == cudaSuccess &&
                      cudaMalloc(&p.tok, (size_t) m * 4) == cudaSuccess &&
                      cudaMalloc(&p.d_ptrs, (size_t) ws->ne * sizeof(uint8_t*)) == cudaSuccess &&
                      cudaMalloc(&p.d_ebase, (size_t) (ws->ne + 1) * 4) == cudaSuccess &&
                      cudaMalloc(&p.tab, (size_t) 64 + (size_t) ws->ne * sizeof(uint8_t*) +
                                            ((size_t) ws->ne + 1) * 4 + 3 * (size_t) m * 4) ==
                          cudaSuccess;
            // the peer table arena's upload event lives on device 1, like the stream it is recorded on
            ok = ok && cudaEventCreateWithFlags(&ws->pin_peer.ev, cudaEventDisableTiming) ==
                           cudaSuccess;
            // [S9c] the peer's per-slot completion events are device-1 objects
            if (ws->cp.on) {
                bool cp1 = true;
                for (int i = 0; cp1 && i < ws->cp.n; ++i)
                    cp1 = cudaEventCreateWithFlags(&ws->cp.ev1[i], cudaEventDisableTiming) == cudaSuccess;
                if (!cp1) {
                    (void) cudaGetLastError();
                    std::fprintf(stderr, "dsv4_moe: peer cold-blob events unavailable; device-1 fills "
                                         "stay in-line\n");
                }
            }
            cudaSetDevice(0);
            ok = ok && cudaEventCreateWithFlags(&p.ev_y, cudaEventDisableTiming) == cudaSuccess;
            p.active = ok;
            if (!ok)
                std::fprintf(stderr, "dsv4_moe: peer workspace on device 1 failed to initialise; "
                                     "device-1 experts will fall back to device-0 staging\n");
        }
    }
}

void dsv4_moe_ws_cache_stats(const Dsv4MoeWs* ws, int64_t& hits, int64_t& misses) {
    hits = ws ? ws->cache_hits : 0;
    misses = ws ? ws->cache_misses : 0;
}

void dsv4_moe_ws_set_heat(Dsv4MoeWs* ws, const float* heat, int64_t n) {
    if (!ws || !heat || n <= 0) return;
    ws->heat.assign(heat, heat + n);
}

int64_t dsv4_moe_ws_evictions(const Dsv4MoeWs* ws) { return ws ? ws->evictions : 0; }

void dsv4_moe_ws_reset_timing(Dsv4MoeWs* ws) {
    if (!ws) return;
    ws->tm_pre = ws->tm_sync = ws->tm_group = ws->tm_chunks = ws->tm_tail = 0;
    ws->tg_chunks = ws->tg_peer = ws->tg_shared = ws->tg_hcpost = ws->tg_route = 0;
    ws->tg_hcpre = ws->tg_rms = ws->tg_gate = ws->tg_topk = 0;
    ws->tm_cold_read = ws->tm_cold_fill = ws->tm_cold_staged = 0;
    ws->tm_pack = ws->tm_tabh2d = ws->tm_lanch = 0;
    ws->cold_n = ws->cold_staged_n = 0;
    ws->cold_pipe = 0;
    ws->moe_calls = 0;
}

// [S9c] ring bookkeeping: a slot a DMA was queued out of returns to the pool once that copy has
// finished (same-stream event, so there is no cross-stream consumer to order against).
static void dsv4_cp_release(Dsv4MoeWs* ws) {
    Dsv4MoeWs::ColdPipe& cp = ws->cp;
    for (int i = 0; i < cp.n; ++i) {
        if (cp.state[i] != 3) continue;
        const cudaError_t q = cudaEventQuery(cp.sdev[i] ? cp.ev1[i] : cp.ev0[i]);
        if (q == cudaSuccess) { cp.state[i] = 0; cp.key[i] = -1; }
        else if (q != cudaErrorNotReady) { (void) cudaGetLastError(); cp.state[i] = 0; cp.key[i] = -1; }
    }
}

static int dsv4_cp_acquire(Dsv4MoeWs* ws) {
    Dsv4MoeWs::ColdPipe& cp = ws->cp;
    if (!cp.on || cp.n <= 0) return -1;
    for (int i = 0; i < cp.n; ++i) {
        const int s = (cp.next + i) % cp.n;
        if (cp.state[s] == 0) { cp.next = (s + 1) % cp.n; return s; }
    }
    return -1;
}

// [S9c] one blob out of the file tier into a pinned slot, with `n` threads.  Measured on this box: one
// core copies from the page cache at ~5.7 GB/s, four threads at ~4.9 GB/s EACH, so a 13.5 MB blob takes
// ~0.7 ms instead of ~2.3 ms.  The bytes are the same ones the demand path copies: this is a plain
// memcpy of the same mapping (a page that is not cached yet still faults in on first touch).
static void dsv4_cp_read_mt(const uint8_t* src, uint8_t* dst, size_t bytes, int nthreads) {
    if (nthreads <= 1 || bytes < (size_t) 4 << 20) {
        std::memcpy(dst, src, bytes);
        return;
    }
    const size_t chunk = (bytes + (size_t) nthreads - 1) / (size_t) nthreads;
    std::thread th[8];
    int started = 0;
    for (int t = 0; t < nthreads; ++t) {
        const size_t off = chunk * (size_t) t;
        if (off >= bytes) break;
        const size_t len = std::min(chunk, bytes - off);
        th[started++] = std::thread([src, dst, off, len] { std::memcpy(dst + off, src + off, len); });
    }
    for (int t = 0; t < started; ++t) th[t].join();
}

// [S9c] a cold blob's fill: read it into a pinned ring slot (multi-threaded), then queue the H2D from
// pinned memory on the kernel stream itself.  The host no longer pays the pageable bounce; the device
// order is exactly the old one (the copy is in-line in the stream), so no slot can be overwritten early.
static bool dsv4_cp_fill(Dsv4MoeWs* ws, strata::core::ExpertCache* C,
                         strata::core::FileExpertSource& experts, int64_t layer, int32_t e,
                         int32_t slot, cudaStream_t stream, int dev, int64_t blob_bytes,
                         std::string& err) {
    Dsv4MoeWs::ColdPipe& cp = ws->cp;
    const int s = dsv4_cp_acquire(ws);
    if (s < 0) return false;
    const uint8_t* src = experts.blob(layer, e);
    if (src == nullptr) return false;
    const bool ton = ws->time_on;
    const auto r0 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    dsv4_cp_read_mt(src, cp.host[s], (size_t) blob_bytes, cp.threads);
    if (ton) {
        const auto r1 = std::chrono::steady_clock::now();
        ws->tm_cold_read += std::chrono::duration<double, std::milli>(r1 - r0).count();
        ++ws->cold_n;
    }
    if (!C->fill_slot(slot, cp.host[s], (void*) stream, err, blob_bytes)) return false;
    cudaEvent_t* const ev = dev ? cp.ev1 : cp.ev0;
    if (cudaEventRecord(ev[s], stream) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    cp.state[s] = 3;
    cp.sdev[s] = dev;
    ++cp.queued;
    return true;
}

// [S7b] one miss's slot: admit first-touch; when the layer's quota is full, swap the coldest
// resident of the same layer - never one this pass routed (its slot is about to be read) - for a
// clearly hotter new expert.  The fill is queued on `stream` either way, so the slot's bytes are
// ordered before every kernel that reads them on that stream.  kNotResident = staging fallback.
static int32_t dsv4_admit_heat(Dsv4MoeWs* ws, strata::core::ExpertCache* C,
                               strata::core::FileExpertSource& experts, int64_t layer, int32_t e,
                               const std::vector<int32_t>& routed, cudaStream_t stream,
                               int64_t blob_bytes, std::string& err, bool& failed, int dev = 0) {
    failed = false;
    int32_t slot = C->admit(layer, e);
    if (slot < 0 && !ws->heat.empty()) {
        const float* hrow = ws->heat.data() + (size_t) layer * ws->ne;
        int32_t victim = -1;
        float vmin = std::numeric_limits<float>::max();
        for (int64_t e2 = 0; e2 < ws->ne; ++e2) {
            if (C->slot_of(layer, e2) < 0) continue;
            if (std::find(routed.begin(), routed.end(), (int32_t) e2) != routed.end()) continue;
            if (hrow[e2] < vmin) {
                vmin = hrow[e2];
                victim = (int32_t) e2;
            }
        }
        if (victim >= 0 && hrow[e] > vmin * (float) ws->evict_gain) {
            slot = C->slot_of(layer, victim);
            C->replace(layer, victim, e);
            ++ws->evictions;
        }
    }
    if (slot < 0) return strata::core::kNotResident;
    // [S9c] the pinned, multi-threaded fill: the host pays the read (~0.7 ms) but not the pageable
    // bounce (~2.1 ms); a full ring falls through to the old in-line copy.
    if (ws->cp.on && dsv4_cp_fill(ws, C, experts, layer, e, slot, stream, dev, blob_bytes, err)) {
        ++ws->cold_pipe;
        return slot;
    }
    ++ws->cp.fell_back;
    err.clear();
    // [S9] attribution: the blob assembly (pread or page-cache memcpy) and the pageable H2D that
    // follows it are two different costs; both land on this thread today.
    const auto ct0 = ws->time_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const uint8_t* blob = experts.blob(layer, e);
    const auto ct1 = ws->time_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!blob) {
        err = "dsv4_moe: no blob for the routed expert";
        failed = true;
        return strata::core::kNotResident;
    }
    if (!C->fill_slot(slot, blob, stream, err, blob_bytes)) {
        failed = true;
        return strata::core::kNotResident;
    }
    if (ws->time_on) {
        const auto ct2 = std::chrono::steady_clock::now();
        ws->tm_cold_read += std::chrono::duration<double, std::milli>(ct1 - ct0).count();
        ws->tm_cold_fill += std::chrono::duration<double, std::milli>(ct2 - ct1).count();
        ++ws->cold_n;
    }
    experts.note_async_read(blob, stream);
    return slot;
}

void dsv4_moe_ws_free(Dsv4MoeWs* ws) {
    if (!ws) return;
    if (ws->route_dump) {
        std::fclose(ws->route_dump);
        ws->route_dump = nullptr;
    }
    // [S7e] the shared-expert side stream and its hand-off events
    if (ws->ev_y != nullptr) cudaEventDestroy(ws->ev_y);
    if (ws->ev_sh != nullptr) cudaEventDestroy(ws->ev_sh);
    for (cudaEvent_t ev : ws->sev)
        if (ev != nullptr) cudaEventDestroy(ev);
    if (ws->sh_stream != nullptr) cudaStreamDestroy(ws->sh_stream);
    if (ws->time_on && ws->moe_calls > 0) {
        std::fprintf(stderr,
                     "dsv4_moe: host phase ms/call over %lld calls: pre %.3f route-sync %.3f "
                     "group %.3f chunks %.3f tail %.3f (sum %.3f)\n",
                     (long long) ws->moe_calls, ws->tm_pre / ws->moe_calls,
                     ws->tm_sync / ws->moe_calls, ws->tm_group / ws->moe_calls,
                     ws->tm_chunks / ws->moe_calls, ws->tm_tail / ws->moe_calls,
                     (ws->tm_pre + ws->tm_sync + ws->tm_group + ws->tm_chunks + ws->tm_tail) /
                         ws->moe_calls);
        if (ws->tev[0] != nullptr)
            std::fprintf(stderr,
                         "dsv4_moe: gpu section ms/call: hc_pre %.3f rmsnorm %.3f gate %.3f "
                         "route %.3f chunks %.3f peer-collect %.3f shared %.3f hc_post+copy %.3f\n",
                         ws->tg_hcpre / ws->moe_calls, ws->tg_rms / ws->moe_calls,
                         ws->tg_gate / ws->moe_calls, ws->tg_topk / ws->moe_calls,
                         ws->tg_chunks / ws->moe_calls, ws->tg_peer / ws->moe_calls,
                         ws->tg_shared / ws->moe_calls, ws->tg_hcpost / ws->moe_calls);
        std::fprintf(stderr,
                     "dsv4_moe: cold io ms/call: admitted %.2f (read %.3f + h2d %.3f = %.3f/blob)"
                     " staged %.2f (h2d %.3f) | single-table pack %.3f tab-h2d %.3f launch %.3f"
                     " | coldpipe %.2f/call (fell back %.2f)/call\n",
                     (double) ws->cold_n / ws->moe_calls, ws->tm_cold_read / ws->moe_calls,
                     ws->tm_cold_fill / ws->moe_calls,
                     ws->cold_n > 0 ? (ws->tm_cold_read + ws->tm_cold_fill) / (double) ws->cold_n : 0.0,
                     (double) ws->cold_staged_n / ws->moe_calls, ws->tm_cold_staged / ws->moe_calls,
                     ws->tm_pack / ws->moe_calls, ws->tm_tabh2d / ws->moe_calls,
                     ws->tm_lanch / ws->moe_calls, (double) ws->cold_pipe / ws->moe_calls,
                     (double) ws->cp.fell_back / ws->moe_calls);
        std::fprintf(stderr, "dsv4_moe: coldpipe ring %d slots, %d reader threads, fills %lld\n",
                     ws->cp.n, ws->cp.threads, (long long) ws->cp.queued);
    }
    for (cudaEvent_t& ev : ws->tev)
        if (ev != nullptr) cudaEventDestroy(ev);
    cudaFree(ws->y);
    cudaFree(ws->yt);
    cudaFree(ws->bout);
    cudaFree(ws->post);
    cudaFree(ws->mixes);
    cudaFree(ws->comb);
    cudaFree(ws->h2);
    cudaFree(ws->scores);
    cudaFree(ws->wgt);
    cudaFree(ws->idx);
    cudaFree(ws->blob);
    cudaFree(ws->blob_stage);
    cudaFree(ws->d_ptrs);
    cudaFree(ws->d_ebase);
    cudaFree(ws->tab);
    for (Dsv4MoeWs::PinTab* t : {&ws->pin[0], &ws->pin[1], &ws->pin_peer}) {
        if (t->ev != nullptr) cudaEventDestroy(t->ev);
        t->ev = nullptr;
        if (t->p != nullptr) cudaFreeHost(t->p);
        t->p = nullptr;
        t->cap = 0;
        t->live = false;
    }
    cudaFree(ws->gate);
    cudaFree(ws->up);
    cudaFree(ws->ph);
    cudaFree(ws->pout);
    cudaFree(ws->pw);
    cudaFree(ws->tok);
    cudaFree(ws->q8);
    cudaFree(ws->q8h);
    cudaFree(ws->sg);
    cudaFree(ws->su);
    cudaFree(ws->sh);
    cudaFree(ws->sout);
    cudaFree(ws->comb6);
    cudaFree(ws->d_slot);
    // [S9] cold-blob pipeline: the device-0 objects here, the device-1 ones with the peer workspace
    for (int i = 0; i < 8; ++i) {
        if (ws->cp.ev0[i]) cudaEventDestroy(ws->cp.ev0[i]);
        if (ws->cp.host[i] != nullptr) cudaFreeHost(ws->cp.host[i]);
    }
    if (ws->peer.active) {
        if (cudaSetDevice(1) == cudaSuccess) {
            for (int i = 0; i < 8; ++i)
                if (ws->cp.ev1[i]) cudaEventDestroy(ws->cp.ev1[i]);
            cudaFree(ws->peer.y1);
            cudaFree(ws->peer.gate);
            cudaFree(ws->peer.up);
            cudaFree(ws->peer.ph);
            cudaFree(ws->peer.pout);
            cudaFree(ws->peer.pw);
            cudaFree(ws->peer.tok);
            cudaFree(ws->peer.d_ptrs);
            cudaFree(ws->peer.d_ebase);
            cudaFree(ws->peer.tab);
            if (ws->peer.stream) cudaStreamDestroy(ws->peer.stream);
            if (ws->peer.ev_y) cudaEventDestroy(ws->peer.ev_y);
            if (ws->peer.ev_done) cudaEventDestroy(ws->peer.ev_done);
            cudaSetDevice(0);
        }
    }
    delete ws;
}

namespace {

/// The shared expert on the GPU: quantize the normed input once, two Q8_0 mmvq (gate/up), the E2 swiglu
/// unweighted, requantize the hidden, the down mmvq, then add into bout.  1..8 columns per mmvq launch
/// (the native family's contract), so the batch goes in chunks of 8.
void shared_expert_gpu(const Dsv4Geometry& g, const Dsv4MoeLayer& w, Dsv4MoeWs* ws, int64_t s,
                       cudaStream_t stream) {
    const int64_t n = g.n_embd, inter = g.n_ff;
    for (int64_t c0 = 0; c0 < s; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, s - c0);
        strata::kernels::native_quantize_q8_1(ws->y + c0 * n,
                                              (char*) ws->q8 + c0 * (n / 32) * 36, (int) n, nc, stream);
        strata::kernels::native_q8_0_mmvq(w.sh_gate, (char*) ws->q8 + c0 * (n / 32) * 36,
                                          ws->sg + c0 * inter, (int) n, (int) inter, nc, stream);
        strata::kernels::native_q8_0_mmvq(w.sh_up, (char*) ws->q8 + c0 * (n / 32) * 36,
                                          ws->su + c0 * inter, (int) n, (int) inter, nc, stream);
    }
    dsv4_swiglu_clamp_kernel<<<(unsigned) s, 256, 0, stream>>>(ws->sg, ws->su, nullptr, (int) inter,
                                                               ws->sh, 0);
    for (int64_t c0 = 0; c0 < s; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, s - c0);
        strata::kernels::native_quantize_q8_1(ws->sh + c0 * inter,
                                              (char*) ws->q8h + c0 * (inter / 32) * 36, (int) inter, nc,
                                              stream);
        strata::kernels::native_q8_0_mmvq(w.sh_down, (char*) ws->q8h + c0 * (inter / 32) * 36,
                                          ws->sout + c0 * n, (int) inter, (int) n, nc, stream);
    }
    // [S7e] no combine here: sout is added by dsv4_moe_combine_kernel in the fixed (experts, shared) order
}

/// The shared expert on the host (the CPU-pool path): dequant_q8_0 + dot_fp32, dsv4ref's gemm_q8.
void shared_expert_cpu(const Dsv4Geometry& g, const Dsv4MoeLayer& w, Dsv4MoeWs* ws, int64_t s) {
    const int64_t n = g.n_embd, inter = g.n_ff;
    std::vector<float> wg((size_t) inter * n), wu((size_t) inter * n), wd((size_t) n * inter);
    const int64_t gu_row = (n / 32) * 34, d_row = (inter / 32) * 34;
    // the shared expert lives on the device (a native tensor): stage it back to the host first
    std::vector<uint8_t> hg((size_t) inter * gu_row), hu((size_t) inter * gu_row),
        hd((size_t) n * d_row);
    cudaMemcpy(hg.data(), w.sh_gate, hg.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hu.data(), w.sh_up, hu.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd.data(), w.sh_down, hd.size(), cudaMemcpyDeviceToHost);
    for (int64_t r = 0; r < inter; ++r)
        dequant_q8_0_host(hg.data() + r * gu_row, n, wg.data() + r * n);
    for (int64_t r = 0; r < inter; ++r)
        dequant_q8_0_host(hu.data() + r * gu_row, n, wu.data() + r * n);
    for (int64_t r = 0; r < n; ++r)
        dequant_q8_0_host(hd.data() + r * d_row, inter, wd.data() + r * inter);
    std::vector<float> hv((size_t) inter);
    for (int64_t j = 0; j < s; ++j) {
        const float* x = ws->h_x.data() + j * n;
        for (int64_t i = 0; i < inter; ++i) {
            const float gv = std::min(dot_fp32_host(x, wg.data() + i * n, n), 10.0f);
            const float uv = std::min(std::max(dot_fp32_host(x, wu.data() + i * n, n), -10.0f), 10.0f);
            hv[(size_t) i] = (gv / (1.0f + std::exp(-gv))) * uv;
        }
        float* bout = ws->h_bout.data() + j * n;
        for (int64_t r = 0; r < n; ++r) bout[r] += dot_fp32_host(hv.data(), wd.data() + r * inter, inter);
    }
}

/// One routed expert on the host (the CPU-pool path): dequant the three roles, FP32 gemm, the E2 swiglu
/// with the router weight on the hidden, accumulate into h_bout.  dsv4ref.cpp:1099-1115 exactly.
void routed_expert_cpu(const uint8_t* blob, const strata::kernels::cpu::NativeFmt& fmt,
                       const float* x, float wgt, int64_t n, int64_t inter, float* bout) {
    float tail[4] = {1.f, 1.f, 1.f, 0.f};
    if (fmt.tail_off) std::memcpy(tail, blob + fmt.tail_off, sizeof tail);
    std::vector<float> w1((size_t) inter * n), w3((size_t) inter * n), w2((size_t) n * inter);
    dequant_nvfp4_host(blob, tail[0], inter, n, w1.data());
    dequant_nvfp4_host(blob + fmt.up_off, tail[1], inter, n, w3.data());
    dequant_nvfp4_host(blob + fmt.down_off, tail[2], n, inter, w2.data());
    std::vector<float> hv((size_t) inter);
    for (int64_t i = 0; i < inter; ++i) {
        const float gv = std::min(dot_fp32_host(x, w1.data() + i * n, n), 10.0f);
        const float uv = std::min(std::max(dot_fp32_host(x, w3.data() + i * n, n), -10.0f), 10.0f);
        hv[(size_t) i] = (gv / (1.0f + std::exp(-gv))) * uv * wgt;
    }
    for (int64_t r = 0; r < n; ++r) bout[r] += dot_fp32_host(hv.data(), w2.data() + r * inter, inter);
}

}  // namespace

// [S7e] section alignment for the packed routing table: the pointer array comes first, then the 4-byte
// scalars, so every section's device address is `arena + offset` (no separate small device buffers).
static inline size_t dsv4_al8(size_t x) { return (x + 7u) & ~(size_t) 7u; }

// STRATA_DSV4_MOE_DEBUG=<dir>: per-layer raw dumps of the MoE half's intermediates (debug/bisect aid)
void moe_debug_dump(const Dsv4MoeWs* ws, int64_t layer, const char* tag, const float* dev, int64_t cnt,
                    const float* hst) {
    if (!ws->debug_dir) return;
    char path[640];
    std::snprintf(path, sizeof path, "%s/L%02lld_%s.bin", ws->debug_dir, (long long) layer, tag);
    std::vector<float> buf((size_t) cnt);
    const float* src = hst;
    if (!src) {
        if (cudaMemcpy(buf.data(), dev, (size_t) cnt * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
        src = buf.data();
    }
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(src, 4, (size_t) cnt, f);
        std::fclose(f);
    }
}

// raw-bytes variant (e.g. the bf16 gate_w: ne*n elements of 2 B)
void moe_debug_dump_raw(const Dsv4MoeWs* ws, int64_t layer, const char* tag, const void* dev,
                        int64_t bytes) {
    if (!ws->debug_dir) return;
    char path[640];
    std::snprintf(path, sizeof path, "%s/L%02lld_%s.bin", ws->debug_dir, (long long) layer, tag);
    std::vector<uint8_t> buf((size_t) bytes);
    if (cudaMemcpy(buf.data(), dev, (size_t) bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(buf.data(), 1, (size_t) bytes, f);
        std::fclose(f);
    }
}

bool dsv4_moe_half(const Dsv4Geometry& g, int64_t layer, const Dsv4MoeLayer& w,
                   strata::core::FileExpertSource& experts, float* d_h, const int32_t* d_tokens,
                   int64_t s, Dsv4MoeWs* ws, cudaStream_t stream, std::string& err) {
    const int64_t n = g.n_embd, ne = g.n_expert, nk = g.n_expert_used, inter = g.n_ff;
    // ws->s is the workspace CAPACITY (allocated rows); a pass may run with fewer tokens
    // (decode runs s_cur=1 against a prefill-sized workspace). Every launch below uses the
    // runtime s, and all buffers were sized for ws->s rows in dsv4_moe_ws_new.
    if (!ws || ws->s < s) { err = "dsv4_moe: workspace/token-count mismatch"; return false; }
    if (!w.ffn_norm || !w.gate_w || !w.hc_fn || !w.hc_base || !w.hc_scale || !w.sh_gate || !w.sh_up ||
        !w.sh_down) {
        err = "dsv4_moe: missing layer weights (ffn_norm/gate_w/hc_ffn_*/shared expert)";
        return false;
    }
    const bool hash_layer = dsv4_is_hash_layer(layer);
    if (hash_layer != (w.tid2eid != nullptr)) {
        err = "dsv4_moe: tid2eid presence disagrees with the layer type";
        return false;
    }
    if (!hash_layer && !w.gate_b) {
        err = "dsv4_moe: routed layer without exp_probs_b";
        return false;
    }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const strata::kernels::cpu::NativeFmt& fmt = lay.fmt[(size_t) layer];
    if (fmt.n_embd != n || fmt.n_ff != inter) {
        err = "dsv4_moe: expert layout geometry mismatch";
        return false;
    }
    const auto tp0 = std::chrono::steady_clock::now();   // [S7b] phase-timing anchor
    const bool tev_on = ws->time_on && ws->tev[0] != nullptr;
    if (tev_on) CKM(cudaEventRecord(ws->tev[5], stream));

    // ---- hc_pre (ffn side) -> rmsnorm(ffn_norm) -> router scores -> selection + weights
    dsv4_hc_mix_kernel<<<dim3((unsigned) s, 24), 256, 0, stream>>>(d_h, w.hc_fn, n, g.rms_eps,
                                                                     ws->mixes);
    dsv4_hc_fin_kernel<<<(unsigned) s, 256, 0, stream>>>(d_h, w.hc_base, w.hc_scale, ws->mixes, n,
                                                         g.hc_eps, ws->y, ws->post, ws->comb);
    if (tev_on) CKM(cudaEventRecord(ws->tev[7], stream));
    strata::kernels::rms_norm_weighted(ws->y, w.ffn_norm, s, n, (float) g.rms_eps, stream);
    // [S7e] the shared expert needs the normed y and nothing else: queue it on the side stream now, so
    // its GPU work overlaps the routing chain and the host gap opened by the route D2H sync below.
    // The combine waits on ev_sh, so the observable result does not depend on the overlap.
    if (ws->overlap_shared && !ws->cpu_pool) {
        CKM(cudaEventRecord(ws->ev_y, stream));
        CKM(cudaStreamWaitEvent(ws->sh_stream, ws->ev_y, 0));
        if (tev_on) CKM(cudaEventRecord(ws->sev[0], ws->sh_stream));
        shared_expert_gpu(g, w, ws, s, ws->sh_stream);
        if (tev_on) CKM(cudaEventRecord(ws->sev[1], ws->sh_stream));
        CKM(cudaEventRecord(ws->ev_sh, ws->sh_stream));
    }
    // [S7c] the gate/up GEMV reads x in the 2048-float-chunk lane layout; one permutation per pass
    // (s x n floats) replaces the GEMV's 8x-inflated strided x reads (2048 rows x n floats).
    dsv4_xpose2048_rows(ws->y, ws->yt, s * n, stream);
    if (tev_on) CKM(cudaEventRecord(ws->tev[8], stream));
    dsv4_gate_scores_kernel<<<dim3((unsigned) s, (unsigned) ((ne + 7) / 8)), 256, 0, stream>>>(
        ws->y, w.gate_w, n, (int) ne, ws->scores);
    if (tev_on) CKM(cudaEventRecord(ws->tev[9], stream));
    dsv4_route_kernel<<<(unsigned) ((s + 7) / 8), 256, 0, stream>>>(
        ws->scores, w.gate_b, w.tid2eid, d_tokens, (int) s, (int) ne, (int) nk, g.route_scale, ws->idx,
        ws->wgt);
    if (tev_on) CKM(cudaEventRecord(ws->tev[6], stream));
    CKM(cudaGetLastError());
    moe_debug_dump(ws, layer, "moe_in", ws->y, s * n, nullptr);
    moe_debug_dump(ws, layer, "scores", ws->scores, s * ne, nullptr);
    moe_debug_dump_raw(ws, layer, "gate_w", w.gate_w, ne * n * 2);   // raw bf16, 2 B/elem
    CKM(cudaMemsetAsync(ws->bout, 0, (size_t) (s * n) * 4, stream));

    // [S7b] host-phase timing (STRATA_DSV4_TIME); tp0 spans the launch of the routing chain
    const bool ton = ws->time_on;
    const auto tp1 = std::chrono::steady_clock::now();

    // ---- the routing decision back to the host (78 int + 78 float at the golden prompt: negligible)
    CKM(cudaMemcpyAsync(ws->h_idx.data(), ws->idx, (size_t) (s * nk) * 4, cudaMemcpyDeviceToHost,
                        stream));
    CKM(cudaMemcpyAsync(ws->h_wgt.data(), ws->wgt, (size_t) (s * nk) * 4, cudaMemcpyDeviceToHost,
                        stream));
    CKM(cudaStreamSynchronize(stream));
    const auto tp2 = std::chrono::steady_clock::now();
    if (ton) {
        ws->tm_pre += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
        ws->tm_sync += std::chrono::duration<double, std::milli>(tp2 - tp1).count();
    }
    if (ws->route_dump) {
        const int32_t rhdr[2] = {(int32_t) layer, (int32_t) s};
        std::fwrite(rhdr, 4, 2, ws->route_dump);
        std::fwrite(ws->h_idx.data(), 4, (size_t) (s * nk), ws->route_dump);
    }
    // [S7b] online heat: decayed routing counts drive eviction.  The decay runs once per pass
    // (the layer-0 call); the counts accumulate per routed entry.
    if (!ws->heat.empty()) {
        if (layer == 0)
            for (float& h : ws->heat) h *= 0.99f;
        float* hrow = ws->heat.data() + (size_t) layer * ne;
        for (int64_t j = 0; j < s; ++j)
            for (int64_t k = 0; k < nk; ++k) ++hrow[ws->h_idx[(size_t) (j * nk + k)]];
    }
    if (ws->debug_dir) {
        char path[640];
        std::snprintf(path, sizeof path, "%s/L%02lld_route_idx.bin", ws->debug_dir, (long long) layer);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(ws->h_idx.data(), 4, ws->h_idx.size(), f);
            std::fclose(f);
        }
        moe_debug_dump(ws, layer, "route_wgt", nullptr, s * nk, ws->h_wgt.data());
    }

    auto tp4 = tp2;   // [S7b] the tail anchor: the GPU branch moves it past the chunk loop
    if (ws->cpu_pool) {
        // ---- the CPU-pool path: every expert on the host with dsv4ref's exact arithmetic
        CKM(cudaMemcpyAsync(ws->h_x.data(), ws->y, (size_t) (s * n) * 4, cudaMemcpyDeviceToHost, stream));
        CKM(cudaStreamSynchronize(stream));
        std::fill(ws->h_bout.begin(), ws->h_bout.end(), 0.0f);
        for (int64_t j = 0; j < s; ++j)
            for (int64_t k = 0; k < nk; ++k) {
                const int64_t e = ws->h_idx[(size_t) (j * nk + k)];
                const uint8_t* blob = experts.blob(layer, e);
                if (!blob) {
                    err = "dsv4_moe: no blob for the routed expert";
                    return false;
                }
                routed_expert_cpu(blob, fmt, ws->h_x.data() + j * n, ws->h_wgt[(size_t) (j * nk + k)], n,
                                  inter, ws->h_bout.data() + j * n);
            }
        shared_expert_cpu(g, w, ws, s);
        CKM(cudaMemcpyAsync(ws->bout, ws->h_bout.data(), (size_t) (s * n) * 4, cudaMemcpyHostToDevice,
                            stream));
    } else {
        // ---- [S7a-2] predictive warm: record this pass's routing (the last row seeds the next pass),
        // and hand the layers just ahead - as routed on the PREVIOUS pass - to the file tier's I/O
        // workers: their whole-blob preads fill the page cache while this layer's attention/MoE
        // computes, so the demand copies below (and at the next layers) find warm pages.  A wrong
        // prediction only wastes readahead; the demand path is unchanged, so the output is unchanged.
        if (ws->n_layers > 0) {
            if ((int64_t) ws->h_prev_route.size() < ws->n_layers)
                ws->h_prev_route.resize((size_t) ws->n_layers);
            std::vector<int64_t>& hist = ws->h_prev_route[(size_t) layer];
            hist.assign(ws->h_idx.end() - nk, ws->h_idx.end());   // the last row's routing
            for (int64_t d = 1; d <= ws->pf_depth; ++d) {
                const int64_t pl = (layer + d) % ws->n_layers;
                const std::vector<int64_t>& pr = ws->h_prev_route[(size_t) pl];
                if (pr.empty()) continue;
                if (ws->cache[0] != nullptr) {
                    // a pair the VRAM tier already serves needs no page-cache warm: skip it, so the
                    // workers' preads do not evict pages the demand path will actually want
                    thread_local std::vector<int64_t> miss;
                    miss.clear();
                    for (const int64_t e : pr) {
                        strata::core::ExpertCache* C =
                            ws->cache[ws->cache[1] != nullptr ? (e & 1) : 0];
                        if (C->slot_of(pl, e) < 0) miss.push_back(e);
                    }
                    if (!miss.empty()) experts.prefetch(pl, miss.data(), (int64_t) miss.size());
                } else {
                    experts.prefetch(pl, pr.data(), (int64_t) pr.size());
                }
            }
        }
        // ---- the GPU path [S7a]: the layer's unique experts grouped (sorted by id, the reference's
        // accumulation order) and staged a chunk at a time; each chunk is ONE grouped launch per role
        // (gate / up / swiglu / down / combine) instead of five launches + a blocking copy per expert.
        // The entry list is global to the layer (ebase/tok/pw), chunked group-locally.
        // [S7a-3b] with the second card's cache on, odd experts (device 1's home) that are resident
        // or admitted-on-first-touch are COMPUTED ON DEVICE 1: a kernel on device 0 reading device-1
        // slots over P2P measures ~0.9 GiB/s inside the GEMV, so the weights never cross PCIe - only
        // the input (16 KiB per token) and the outputs (entries*n*4) do.
        std::vector<int32_t>& groups = ws->h_groups;
        groups.clear();
        // [S8] the layer's unique experts come straight out of a presence bitmap in ascending expert
        // id - byte for byte the order std::sort + std::unique produced (the reference's accumulation
        // order), but O(s*nk + ne) with no sort, no vector growth and no allocation per call.
        std::vector<uint64_t>& mask = ws->h_mask;
        std::vector<int32_t>& cnt = ws->h_cnt;
        std::fill(mask.begin(), mask.end(), 0ull);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int64_t i = 0; i < s * nk; ++i) {
            const int64_t e = (int64_t) ws->h_idx[(size_t) i];
            mask[(size_t) (e >> 6)] |= 1ull << (e & 63);
            ++cnt[(size_t) e];
        }
        for (int64_t e = 0; e < ws->ne; ++e)
            if (mask[(size_t) (e >> 6)] & (1ull << (e & 63))) groups.push_back((int32_t) e);
        const unsigned gu_blocks = (unsigned) (inter / (256 / (n / 64)));        // 2048 / 4
        const unsigned d_blocks = (unsigned) (n / (256 / (inter / 64)));         // 4096 / 8
        const uint64_t blob_bytes = (uint64_t) lay.blob_bytes(layer);
        const bool peer_on = ws->cache[0] != nullptr && ws->cache[1] != nullptr && ws->peer.active;
        // [S9c] slots whose previous DMA has retired go back to the ring once per layer
        dsv4_cp_release(ws);
        std::vector<int32_t>& groups0 = ws->h_groups0;
        std::vector<int32_t>& groups1 = ws->h_groups1;
        std::vector<uint8_t*>& ptrs1 = ws->h_ptrs1;
        std::vector<uint8_t>& in0 = ws->h_in0;
        groups0.clear();
        groups1.clear();
        ptrs1.clear();
        std::fill(in0.begin(), in0.end(), (uint8_t) 0);
        if (peer_on) {
            // device-1 home: hit -> peer pass; first touch -> admit + fill on the PEER stream (the
            // fill overlaps device 0's work below); device-1 cache full -> device-0 staging fallback
            for (const int32_t e : groups) {
                if ((e & 1) == 0) {
                    groups0.push_back(e);
                    in0[(size_t) e] = 1;
                    continue;
                }
                strata::core::ExpertCache* C = ws->cache[1];
                int32_t slot = C->slot_of(layer, e);
                if (slot >= 0) {
                    ++ws->cache_hits;
                } else {
                    bool failed = false;
                    slot = dsv4_admit_heat(ws, C, experts, layer, e, groups, ws->peer.stream,
                                           (int64_t) blob_bytes, err, failed, 1);
                    if (failed) return false;
                }
                if (slot >= 0) {
                    groups1.push_back(e);
                    ptrs1.push_back(C->device_slot(slot));
                } else {
                    groups0.push_back(e);   // device-1 cache full: stage + compute on device 0
                    in0[(size_t) e] = 1;
                }
            }
        } else {
            groups0 = groups;
            for (const int32_t e : groups) in0[(size_t) e] = 1;
        }
        // entry tables for the device-0 groups.  [S8] one stable bucket pass over the s*nk routing
        // entries (they are already in (token, slot) order, so each group's slice keeps that order and
        // each entry keeps its comb cell) replaces the groups0 x s x nk rescan; the group offsets are
        // the prefix sum of the per-expert counts gathered with the bitmap above.
        std::vector<int32_t>& ebase = ws->h_ebase;
        std::vector<int32_t>& tokg = ws->h_tokg;
        std::vector<float>& pwg = ws->h_pwg;
        std::vector<int32_t>& cur = ws->h_cur;
        ebase.clear();
        tokg.clear();
        pwg.clear();
        ws->h_slotg.clear();
        std::fill(cur.begin(), cur.end(), -1);
        ebase.push_back(0);
        {
            int32_t acc = 0;
            for (const int32_t e : groups0) {
                cur[(size_t) e] = acc;
                acc += cnt[(size_t) e];
                ebase.push_back(acc);
            }
            tokg.resize((size_t) acc);
            pwg.resize((size_t) acc);
            ws->h_slotg.resize((size_t) acc);
        }
        for (int64_t j = 0; j < s; ++j)
            for (int64_t k = 0; k < nk; ++k) {
                const int64_t i = j * nk + k;
                const int32_t e = ws->h_idx[(size_t) i];
                const int32_t pos = cur[(size_t) e];
                if (pos < 0) continue;
                cur[(size_t) e] = pos + 1;
                tokg[(size_t) pos] = (int32_t) j;
                pwg[(size_t) pos] = ws->h_wgt[(size_t) i];
                ws->h_slotg[(size_t) pos] = dsv4_comb_cell(ws->h_idx.data(), j, (int) nk, e, k);  // [S7e]
            }
        // ---- the peer pass: every device-1 group in ONE grouped launch set on device 1.  The slot
        // fills queued above precede the wait on device 0's y; the outputs cross back after the
        // device-0 loop, so device 1 computes while device 0 does its own groups.
        int64_t entries1 = 0;
        if (!groups1.empty()) {
            Dsv4MoeWs::PeerWs& p = ws->peer;
            std::vector<int32_t>& ebase1 = ws->h_ebase1;
            std::vector<int32_t>& tokg1 = ws->h_tokg1;
            std::vector<float>& pwg1 = ws->h_pwg1;
            ebase1.clear();
            tokg1.clear();
            pwg1.clear();
            ws->h_slotg1.clear();
            std::fill(cur.begin(), cur.end(), -1);
            ebase1.push_back(0);
            int64_t max_m1 = 0;
            {
                // [S8] same stable bucket fill as the device-0 table, over the peer groups' entries
                int32_t acc = 0;
                for (const int32_t e : groups1) {
                    cur[(size_t) e] = acc;
                    if ((int64_t) cnt[(size_t) e] > max_m1) max_m1 = (int64_t) cnt[(size_t) e];
                    acc += cnt[(size_t) e];
                    ebase1.push_back(acc);
                }
                tokg1.resize((size_t) acc);
                pwg1.resize((size_t) acc);
                ws->h_slotg1.resize((size_t) acc);
            }
            for (int64_t j = 0; j < s; ++j)
                for (int64_t k = 0; k < nk; ++k) {
                    const int64_t i = j * nk + k;
                    const int32_t e = ws->h_idx[(size_t) i];
                    const int32_t pos = cur[(size_t) e];
                    if (pos < 0) continue;
                    cur[(size_t) e] = pos + 1;
                    tokg1[(size_t) pos] = (int32_t) j;
                    pwg1[(size_t) pos] = ws->h_wgt[(size_t) i];
                    ws->h_slotg1[(size_t) pos] =
                        dsv4_comb_cell(ws->h_idx.data(), j, (int) nk, e, k);   // [S7e]
                }
            entries1 = (int64_t) tokg1.size();
            const int64_t u1 = (int64_t) groups1.size();
            CKM(cudaEventRecord(p.ev_y, stream));
            CKM(cudaStreamWaitEvent(p.stream, p.ev_y, 0));
            CKM(cudaMemcpyPeerAsync(p.y1, 1, ws->yt, 0, (size_t) (s * n) * 4, p.stream));
            // [S7e] one packed peer upload (ptrs | ebase | tok | pw | comb cell) on the peer stream: the
            // scatter after the copy-back reads the SAME tok/slot sections, so its two late uploads go
            // away as well.
            const size_t po_p = 0;
            const size_t po_e = dsv4_al8(po_p + (size_t) u1 * sizeof(uint8_t*));
            const size_t po_t = dsv4_al8(po_e + (size_t) (u1 + 1) * 4);
            const size_t po_w = dsv4_al8(po_t + (size_t) entries1 * 4);
            const size_t pneed = po_w + (size_t) entries1 * 4;
            uint8_t* H1 = dsv4_pintab_take(ws->pin_peer, pneed);
            if (H1 == nullptr) {
                ws->h_tab.assign(pneed, 0);
                H1 = ws->h_tab.data();
            } else {
                std::memset(H1, 0, pneed);
            }
            std::memcpy(H1 + po_p, ptrs1.data(), (size_t) u1 * sizeof(uint8_t*));
            std::memcpy(H1 + po_e, ebase1.data(), (size_t) (u1 + 1) * 4);
            std::memcpy(H1 + po_t, tokg1.data(), (size_t) entries1 * 4);
            std::memcpy(H1 + po_w, pwg1.data(), (size_t) entries1 * 4);
            CKM(cudaMemcpyAsync(p.tab, H1, pneed, cudaMemcpyHostToDevice, p.stream));
            dsv4_pintab_queued(ws->pin_peer, H1, p.stream);
            CKM(cudaSetDevice(1));
            const uint8_t* const* p_ptrs = (const uint8_t* const*) p.tab;
            const int32_t* p_ebase = (const int32_t*) (p.tab + po_e);
            const int32_t* p_tok = (const int32_t*) (p.tab + po_t);
            const float* p_pw = (const float*) (p.tab + po_w);
            const dim3 gu_grid1(gu_blocks, (unsigned) max_m1, (unsigned) u1);
            const dim3 d_grid1(d_blocks, (unsigned) max_m1, (unsigned) u1);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid1, 256, 9216, p.stream>>>(
                p_ptrs, 0, (int64_t) fmt.tail_off, 0, p.y1, n, p_tok, p_ebase, (int) n, (int) inter,
                p.gate);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid1, 256, 9216, p.stream>>>(
                p_ptrs, (int64_t) fmt.up_off, (int64_t) fmt.tail_off, 1, p.y1, n, p_tok, p_ebase,
                (int) n, (int) inter, p.up);
            dsv4_swiglu_clamp_kernel<<<(unsigned) entries1, 256, 0, p.stream>>>(p.gate, p.up, p_pw,
                                                                                 (int) inter, p.ph,
                                                                                 1);
            dsv4_nvfp4_gemv_grouped_kernel<<<d_grid1, 256, 9216, p.stream>>>(
                p_ptrs, (int64_t) fmt.down_off, (int64_t) fmt.tail_off, 2, p.ph, inter, nullptr,
                p_ebase, (int) inter, (int) n, p.pout);
            CKM(cudaGetLastError());
            CKM(cudaEventRecord(p.ev_done, p.stream));
            CKM(cudaSetDevice(0));
        }
        const auto tp3 = std::chrono::steady_clock::now();
        const bool tev_ok = ton && ws->tev[0] != nullptr;
        if (tev_ok) CKM(cudaEventRecord(ws->tev[0], stream));
        const int64_t n_groups = (int64_t) groups0.size();
        // [S8] ONE packed table for the whole layer and ONE launch set per role (was: five launches
        // per chunk of eight groups plus one packed pageable upload per chunk).  Every group's blob
        // address is resolved once - a VRAM-resident expert is its cache slot directly, a miss is
        // admitted first-touch - and then the launch shape follows from whether a blob had to go to
        // the staging arena:
        //   * nothing staged -> a single grouped launch per role with blockIdx.z over ALL groups, so
        //     the GPU sees 4 x max_m x n_groups blocks instead of 4 x max_m x 8 (at R=6 that is 864
        //     blocks on 92 SMs instead of ~192);
        //   * something staged -> the chunked loop below, unchanged: blob_stage only holds `chunk`
        //     blobs, and it now reuses the addresses resolved here (it used to redo the lookups).
        // Bit-exactness: an entry's gate/up/ph/pout offsets become global instead of chunk-local and
        // blockIdx.z becomes the global group index; both are used only as addresses.  Each entry
        // still runs gate -> up -> swiglu -> down -> scatter in that order on the same x row, the
        // same weights and its own comb6 cell (dsv4_comb_cell), so the combine's ascending-expert-id
        // / shared-last order is untouched.
        int64_t n_staged = 0;
        for (int64_t g = 0; g < n_groups; ++g) {
            const int32_t e = groups0[(size_t) g];
            uint8_t* dst = nullptr;
            // device 0's loop only ever touches device 0's cache: with the peer on, an odd expert
            // here is a device-1-cache-full fallback and goes straight to staging
            strata::core::ExpertCache* C = nullptr;
            if (ws->cache[0] != nullptr && !(peer_on && (e & 1))) C = ws->cache[0];
            if (C != nullptr) {
                const int32_t resident = C->slot_of(layer, e);
                if (resident >= 0) {
                    dst = C->device_slot(resident);
                    ++ws->cache_hits;
                } else {
                    bool failed = false;
                    const int32_t slot = dsv4_admit_heat(ws, C, experts, layer, e, groups0, stream,
                                                         (int64_t) blob_bytes, err, failed);
                    if (failed) return false;
                    if (slot >= 0) dst = C->device_slot(slot);
                    ++ws->cache_misses;
                }
            }
            if (dst == nullptr) ++n_staged;
            ws->h_ptrs[(size_t) g] = dst;
        }
        // [S8] kSingle=0 forces the chunked path (A/B switch for the one-table launch set)
        static const bool kSingle = std::getenv("STRATA_DSV4_MOE_SINGLE") == nullptr ||
                                    std::atoi(std::getenv("STRATA_DSV4_MOE_SINGLE")) != 0;
        if (kSingle && n_groups > 0 && n_staged == 0) {
            int64_t max_m = 0;
            for (int64_t g = 0; g < n_groups; ++g) {
                const int64_t mz = (int64_t) ebase[(size_t) (g + 1)] - ebase[(size_t) g];
                if (mz > max_m) max_m = mz;
            }
            const int64_t entries = (int64_t) tokg.size();
            // [S8] ONE upload for the layer's table: pointers, offsets, tok, pw and comb cell packed
            // into the pinned arena (a pageable source would stall this thread while the driver
            // bounces it), then five launches for the whole layer.
            const size_t o_p = 0;
            const size_t o_e = dsv4_al8(o_p + (size_t) n_groups * sizeof(uint8_t*));
            const size_t o_t = dsv4_al8(o_e + (size_t) (n_groups + 1) * 4);
            const size_t o_w = dsv4_al8(o_t + (size_t) entries * 4);
            const size_t o_s = dsv4_al8(o_w + (size_t) entries * 4);
            const size_t need = o_s + (size_t) entries * 4;
            // the arenas alternate, and `take` waits out the copy that last used this one (with the
            // five launches of a chunk in between it is never actually still running)
            Dsv4MoeWs::PinTab& T = ws->pin[ws->pin_turn & 1];
            ws->pin_turn ^= 1;
            const auto pt0 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            uint8_t* H = dsv4_pintab_take(T, need);
            if (H == nullptr) {
                ws->h_tab.resize(need);
                H = ws->h_tab.data();
            } else {
                std::memset(H, 0, need);
            }
            std::memcpy(H + o_p, ws->h_ptrs.data(), (size_t) n_groups * sizeof(uint8_t*));
            std::memcpy(H + o_e, ebase.data(), (size_t) (n_groups + 1) * 4);
            std::memcpy(H + o_t, tokg.data(), (size_t) entries * 4);
            std::memcpy(H + o_w, pwg.data(), (size_t) entries * 4);
            std::memcpy(H + o_s, ws->h_slotg.data(), (size_t) entries * 4);
            const auto pt1 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            CKM(cudaMemcpyAsync(ws->tab, H, need, cudaMemcpyHostToDevice, stream));
            dsv4_pintab_queued(T, H, stream);
            const auto pt2 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const uint8_t* const* k_ptrs = (const uint8_t* const*) ws->tab;
            const int32_t* k_ebase = (const int32_t*) (ws->tab + o_e);
            const int32_t* k_tok = (const int32_t*) (ws->tab + o_t);
            const float* k_pw = (const float*) (ws->tab + o_w);
            const int32_t* k_slot = (const int32_t*) (ws->tab + o_s);
            const dim3 gu_grid(gu_blocks, (unsigned) max_m, (unsigned) n_groups);
            const dim3 d_grid(d_blocks, (unsigned) max_m, (unsigned) n_groups);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid, 256, 9216, stream>>>(
                k_ptrs, 0, (int64_t) fmt.tail_off, 0, ws->yt, n, k_tok, k_ebase, (int) n, (int) inter,
                ws->gate);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid, 256, 9216, stream>>>(
                k_ptrs, (int64_t) fmt.up_off, (int64_t) fmt.tail_off, 1, ws->yt, n, k_tok, k_ebase,
                (int) n, (int) inter, ws->up);
            dsv4_swiglu_clamp_kernel<<<(unsigned) entries, 256, 0, stream>>>(ws->gate, ws->up, k_pw,
                                                                             (int) inter, ws->ph, 1);
            dsv4_nvfp4_gemv_grouped_kernel<<<d_grid, 256, 9216, stream>>>(
                k_ptrs, (int64_t) fmt.down_off, (int64_t) fmt.tail_off, 2, ws->ph, inter, nullptr,
                k_ebase, (int) inter, (int) n, ws->pout);
            dsv4_scatter_rows_kernel<<<(unsigned) entries, 256, 0, stream>>>(ws->pout, k_slot,
                                                                            ws->comb6, n);
            if (ton) {
                const auto pt3 = std::chrono::steady_clock::now();
                ws->tm_pack += std::chrono::duration<double, std::milli>(pt1 - pt0).count();
                ws->tm_tabh2d += std::chrono::duration<double, std::milli>(pt2 - pt1).count();
                ws->tm_lanch += std::chrono::duration<double, std::milli>(pt3 - pt2).count();
            }
        } else {
        for (int64_t z0 = 0; z0 < n_groups; z0 += ws->chunk) {
            const int64_t u = std::min<int64_t>((int64_t) ws->chunk, n_groups - z0);
            const int32_t entry0 = ebase[(size_t) z0];
            const int64_t entries = (int64_t) ebase[(size_t) (z0 + u)] - entry0;
            int64_t max_m = 0;
            // the staging arena only serves the blobs the resolve pass could not place in a cache
            // (13.5 MiB each, so the fallback is bounded by `chunk` groups per launch set)
            for (int64_t zz = 0; zz < u; ++zz) {
                uint8_t* dst = ws->h_ptrs[(size_t) (z0 + zz)];
                if (dst == nullptr) {
                    const int32_t e = groups0[(size_t) (z0 + zz)];
                    const auto st0 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const uint8_t* blob = experts.blob(layer, e);
                    const auto st1 = ton ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    if (!blob) {
                        err = "dsv4_moe: no blob for the routed expert";
                        return false;
                    }
                    dst = ws->blob_stage + (size_t) zz * blob_bytes;
                    CKM(cudaMemcpyAsync(dst, blob, (size_t) blob_bytes, cudaMemcpyHostToDevice,
                                        stream));
                    if (ton) {
                        ws->tm_cold_read += std::chrono::duration<double, std::milli>(st1 - st0).count();
                        ws->tm_cold_staged +=
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st1).count();
                        ++ws->cold_staged_n;
                    }
                    experts.note_async_read(blob, stream);
                }
                ws->h_ptrs[(size_t) (z0 + zz)] = dst;
                const int64_t mz = (int64_t) ebase[(size_t) (z0 + zz + 1)] - ebase[(size_t) (z0 + zz)];
                if (mz > max_m) max_m = mz;
            }
            // chunk-local ebase (the kernels index gate/up/ph/pout by chunk-local entry)
            for (int64_t zz = 0; zz <= u; ++zz)
                ws->h_ebase_c[(size_t) zz] = ebase[(size_t) (z0 + zz)] - entry0;
            // [S7e] ONE packed upload for the chunk (ptrs | ebase | tok | pw | comb cell): nine small
            // memcpys per layer became two, and the sections are 8-byte aligned so each device pointer
            // is `tab + offset`.  [S8] the pack buffer is the pinned arena.
            const size_t o_p = 0;
            const size_t o_e = dsv4_al8(o_p + (size_t) u * sizeof(uint8_t*));
            const size_t o_t = dsv4_al8(o_e + (size_t) (u + 1) * 4);
            const size_t o_w = dsv4_al8(o_t + (size_t) entries * 4);
            const size_t o_s = dsv4_al8(o_w + (size_t) entries * 4);
            const size_t need = o_s + (size_t) entries * 4;
            // the arenas alternate, and `take` waits out the copy that last used this one (with the
            // five launches of a chunk in between it is never actually still running)
            Dsv4MoeWs::PinTab& T = ws->pin[ws->pin_turn & 1];
            ws->pin_turn ^= 1;
            uint8_t* H = dsv4_pintab_take(T, need);
            if (H == nullptr) {
                ws->h_tab.resize(need);
                H = ws->h_tab.data();
            } else {
                std::memset(H, 0, need);
            }
            std::memcpy(H + o_p, ws->h_ptrs.data() + z0, (size_t) u * sizeof(uint8_t*));
            std::memcpy(H + o_e, ws->h_ebase_c.data(), (size_t) (u + 1) * 4);
            std::memcpy(H + o_t, tokg.data() + entry0, (size_t) entries * 4);
            std::memcpy(H + o_w, pwg.data() + entry0, (size_t) entries * 4);
            std::memcpy(H + o_s, ws->h_slotg.data() + entry0, (size_t) entries * 4);
            CKM(cudaMemcpyAsync(ws->tab, H, need, cudaMemcpyHostToDevice, stream));
            dsv4_pintab_queued(T, H, stream);
            const uint8_t* const* k_ptrs = (const uint8_t* const*) ws->tab;
            const int32_t* k_ebase = (const int32_t*) (ws->tab + o_e);
            const int32_t* k_tok = (const int32_t*) (ws->tab + o_t);
            const float* k_pw = (const float*) (ws->tab + o_w);
            const int32_t* k_slot = (const int32_t*) (ws->tab + o_s);
            const dim3 gu_grid(gu_blocks, (unsigned) max_m, (unsigned) u);
            const dim3 d_grid(d_blocks, (unsigned) max_m, (unsigned) u);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid, 256, 9216, stream>>>(
                k_ptrs, 0, (int64_t) fmt.tail_off, 0, ws->yt, n, k_tok, k_ebase, (int) n, (int) inter,
                ws->gate);
            dsv4_nvfp4_gemv_grouped_kernel<<<gu_grid, 256, 9216, stream>>>(
                k_ptrs, (int64_t) fmt.up_off, (int64_t) fmt.tail_off, 1, ws->yt, n, k_tok, k_ebase,
                (int) n, (int) inter, ws->up);
            dsv4_swiglu_clamp_kernel<<<(unsigned) entries, 256, 0, stream>>>(ws->gate, ws->up, k_pw,
                                                                             (int) inter, ws->ph, 1);
            dsv4_nvfp4_gemv_grouped_kernel<<<d_grid, 256, 9216, stream>>>(
                k_ptrs, (int64_t) fmt.down_off, (int64_t) fmt.tail_off, 2, ws->ph, inter, nullptr,
                k_ebase, (int) inter, (int) n, ws->pout);
            dsv4_scatter_rows_kernel<<<(unsigned) entries, 256, 0, stream>>>(ws->pout, k_slot,
                                                                            ws->comb6, n);
        }
        }
        if (tev_ok) CKM(cudaEventRecord(ws->tev[1], stream));
        if (ton) {
            tp4 = std::chrono::steady_clock::now();
            ws->tm_group += std::chrono::duration<double, std::milli>(tp3 - tp2).count();
            ws->tm_chunks += std::chrono::duration<double, std::milli>(tp4 - tp3).count();
        }
        if (entries1 > 0) {
            // the peer's outputs cross back as ONE small DMA, then combine on device 0 (bout is
            // atomicAdd-ed per entry; the combine order against the device-0 groups is irrelevant)
            CKM(cudaStreamWaitEvent(stream, ws->peer.ev_done, 0));
            CKM(cudaMemcpyPeerAsync(ws->pout, 0, ws->peer.pout, 1, (size_t) (entries1 * n) * 4,
                                    stream));
            CKM(cudaMemcpyAsync(ws->tok, ws->h_tokg1.data(), (size_t) entries1 * 4,
                                cudaMemcpyHostToDevice, stream));
            CKM(cudaMemcpyAsync(ws->d_slot, ws->h_slotg1.data(), (size_t) entries1 * 4,
                                cudaMemcpyHostToDevice, stream));
            dsv4_scatter_rows_kernel<<<(unsigned) entries1, 256, 0, stream>>>(ws->pout, ws->d_slot,
                                                                            ws->comb6, n);
        }
        if (tev_ok) CKM(cudaEventRecord(ws->tev[2], stream));
        if (ws->overlap_shared) {
            CKM(cudaStreamWaitEvent(stream, ws->ev_sh, 0));   // [S7e] the side stream's results
        } else {
            shared_expert_gpu(g, w, ws, s, stream);
        }
        // [S7e] the deterministic combine: experts in ascending expert id (each entry landed in its own
        // comb6 cell), then the shared expert - the reference's order, one launch, no atomics.
        dsv4_moe_combine_kernel<<<(unsigned) s, 256, 0, stream>>>(ws->comb6, ws->sout, ws->bout, n,
                                                                  (int) ws->nk);
        if (tev_ok) CKM(cudaEventRecord(ws->tev[3], stream));
    }
    CKM(cudaGetLastError());

    // ---- hc_post into the workspace buffer, then back over h
    moe_debug_dump(ws, layer, "moe_out", ws->bout, s * n, nullptr);
    moe_debug_dump(ws, layer, "hc_post_c", ws->post, s * 4, nullptr);
    moe_debug_dump(ws, layer, "hc_comb", ws->comb, s * 16, nullptr);
    dsv4_hc_post_kernel<<<(unsigned) s, 256, 0, stream>>>(d_h, ws->bout, ws->post, ws->comb, n, ws->h2);
    CKM(cudaGetLastError());
    CKM(cudaMemcpyAsync(d_h, ws->h2, (size_t) (s * 4 * n) * 4, cudaMemcpyDeviceToDevice, stream));
    const bool tev_ok2 = ton && ws->tev[0] != nullptr;
    if (tev_ok2) CKM(cudaEventRecord(ws->tev[4], stream));
    // [S7e] the layer-end sync served the host-side tail timing and the stage dumps only: every
    // consumer of h2/d_h is a kernel on this same stream, so leaving it out lets the host queue the
    // next layer's attention while the GPU is still finishing these groups (43 pipeline drains per
    // pass off the critical path; the arithmetic is untouched).
    if (ton) CKM(cudaStreamSynchronize(stream));
    if (ton) {
        ws->tm_tail += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                 tp4)
                           .count();
        ++ws->moe_calls;
        if (tev_ok2) {
            float el = 0;
            if (cudaEventElapsedTime(&el, ws->tev[5], ws->tev[7]) == cudaSuccess) ws->tg_hcpre += el;
            if (cudaEventElapsedTime(&el, ws->tev[7], ws->tev[8]) == cudaSuccess) ws->tg_rms += el;
            if (cudaEventElapsedTime(&el, ws->tev[8], ws->tev[9]) == cudaSuccess) ws->tg_gate += el;
            if (cudaEventElapsedTime(&el, ws->tev[9], ws->tev[6]) == cudaSuccess) ws->tg_topk += el;
            if (cudaEventElapsedTime(&el, ws->tev[0], ws->tev[1]) == cudaSuccess)
                ws->tg_chunks += el;
            if (cudaEventElapsedTime(&el, ws->tev[1], ws->tev[2]) == cudaSuccess) ws->tg_peer += el;
            if (ws->sev[0] != nullptr && ws->sev[1] != nullptr &&
                cudaEventElapsedTime(&el, ws->sev[0], ws->sev[1]) == cudaSuccess)
                ws->tg_shared += el;
            if (cudaEventElapsedTime(&el, ws->tev[2], ws->tev[3]) == cudaSuccess)
                ws->tg_shared += el;
            if (cudaEventElapsedTime(&el, ws->tev[3], ws->tev[4]) == cudaSuccess)
                ws->tg_hcpost += el;
        }
    }
    return true;
}

}  // namespace strata::dsv4
