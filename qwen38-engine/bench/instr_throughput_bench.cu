// Instruction throughput on the GTX 1070 Ti (sm_61): warp instructions per clock per SM for the operations that matter to the
// attention/GEMV kernels. Each kernel runs 8 independent dependency chains per thread (enough ILP to hide the 6-cycle latency)
// with 512 threads x 4 blocks per SM resident. Clock is taken from clock64 inside the kernel (the card boosts), so the numbers
// are per SM clock, not per wall time. Operations are inline PTX so the compiler cannot fold them.
// Build: nvcc -arch=sm_61 -O3 bench/instr_throughput_bench.cu -o build/instr_throughput_bench
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int ITERS = 4096, CH = 8;

// Each op is a functor on a 32-bit register (bit pattern: float ops reinterpret it). step(a, b, d) must be one PTX instruction
// (sel is two, i2f/ex2 are single conversions) so that lane-ops per clock = instruction rate.
struct OpFfma { __device__ static void step(unsigned & a, unsigned b, unsigned d) { float x = __uint_as_float(a); asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(__uint_as_float(b)), "f"(__uint_as_float(d))); a = __float_as_uint(x); } };
struct OpFadd { __device__ static void step(unsigned & a, unsigned b, unsigned) { float x = __uint_as_float(a); asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(x) : "f"(__uint_as_float(b))); a = __float_as_uint(x); } };
struct OpIadd { __device__ static void step(unsigned & a, unsigned b, unsigned) { asm volatile("add.u32 %0, %0, %1;" : "+r"(a) : "r"(b)); } };
struct OpLop3 { __device__ static void step(unsigned & a, unsigned b, unsigned d) { asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a) : "r"(b), "r"(d)); } };
// (and / shl removed from main: ptxas folds idempotent chains, the timing is meaningless)
struct OpAnd { __device__ static void step(unsigned & a, unsigned b, unsigned) { asm volatile("and.b32 %0, %0, %1;" : "+r"(a) : "r"(b | 0xf0f0f0f0u)); } };
struct OpShl { __device__ static void step(unsigned & a, unsigned, unsigned) { asm volatile("shl.b32 %0, %0, 1;" : "+r"(a)); } };
struct OpShf { __device__ static void step(unsigned & a, unsigned b, unsigned) { asm volatile("shf.l.wrap.b32 %0, %0, %1, 5;" : "+r"(a) : "r"(b)); } };
struct OpPrmt { __device__ static void step(unsigned & a, unsigned b, unsigned d) { asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a) : "r"(b), "r"(d & 0x7777u)); } };
struct OpBfe { __device__ static void step(unsigned & a, unsigned b, unsigned) { asm volatile("bfe.u32 %0, %0, %1, 7;" : "+r"(a) : "r"(b & 15u)); } };
struct OpPopc { __device__ static void step(unsigned & a, unsigned, unsigned) { asm volatile("popc.b32 %0, %0;" : "+r"(a)); a += 0x10101u; } };
struct OpImad { __device__ static void step(unsigned & a, unsigned b, unsigned d) { asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a) : "r"(b), "r"(d)); } };
struct OpDp4a { __device__ static void step(unsigned & a, unsigned b, unsigned d) { asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(a) : "r"(b), "r"(d)); } };
struct OpDp2a { __device__ static void step(unsigned & a, unsigned b, unsigned d) { asm volatile("dp2a.lo.s32.s32 %0, %1, %2, %0;" : "+r"(a) : "r"(b), "r"(d)); } };
struct OpI2f { __device__ static void step(unsigned & a, unsigned b, unsigned) { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(a + b)); a = __float_as_uint(f) ^ 0x1234u; } };
struct OpEx2 { __device__ static void step(unsigned & a, unsigned, unsigned) { float x = __uint_as_float(a); asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x)); a = __float_as_uint(x) | 0x3f000000u & 0x3f7fffffu; } };

template <typename OP>
__global__ void __launch_bounds__(512) k_op(unsigned * out, long long * clk, unsigned seed) {
    unsigned a[CH];
    const unsigned b = (seed | 1u) * 0x01010101u + 0x3f800001u, d = seed * 3u + 0x3f000005u;
#pragma unroll
    for (int c = 0; c < CH; ++c) a[c] = 0x3f800000u + c * 77u + threadIdx.x;
    long long t0 = clock64();
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int c = 0; c < CH; ++c) OP::step(a[c], b, d);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *clk = t1 - t0;
    unsigned s = 0;
#pragma unroll
    for (int c = 0; c < CH; ++c) s ^= a[c];
    if (s == 0x9e3779b9u) out[0] = s;
}

template <typename K>
void run(const char * name, K kernel, int ops_per_iter_per_thread = CH) {
    unsigned * out; long long * clk;
    CK(cudaMalloc(&out, 64)); CK(cudaMalloc(&clk, 8));
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = nsm * 4, thr = 512 / 1;   // 4 blocks x 512 threads = 2048 threads/SM would exceed 2048? (64 warps ok)
    kernel<<<blocks, thr>>>(out, clk, 1);          // warm up
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e30f; long long cyc = 0;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(e0)); kernel<<<blocks, thr>>>(out, clk, 1); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) { best = ms; CK(cudaMemcpy(&cyc, clk, 8, cudaMemcpyDeviceToHost)); }
    }
    const double warp_instr = static_cast<double>(blocks) * (thr / 32) * ITERS * ops_per_iter_per_thread;
    const double ghz = static_cast<double>(cyc) / (best * 1e6);   // block 0 spans (almost) the whole kernel
    const double per_sm_clk = warp_instr / nsm / (best * 1e-3 * ghz * 1e9);
    std::printf("%-10s %8.3f ms   sm clock %.3f GHz   %6.1f warp-instr/clk/SM  = %6.1f lane-ops/clk/SM\n", name, best, ghz, per_sm_clk, per_sm_clk * 32);
    CK(cudaFree(out)); CK(cudaFree(clk));
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    std::printf("%s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    run("ffma", k_op<OpFfma>); run("fadd", k_op<OpFadd>); run("iadd", k_op<OpIadd>); run("lop3", k_op<OpLop3>);
    run("shf", k_op<OpShf>); run("prmt", k_op<OpPrmt>); run("bfe", k_op<OpBfe>); run("popc(+iadd)", k_op<OpPopc>);
    run("imad", k_op<OpImad>); run("dp4a", k_op<OpDp4a>); run("dp2a", k_op<OpDp2a>); run("i2f", k_op<OpI2f>); run("ex2 (MUFU)", k_op<OpEx2>);
    return 0;
}
