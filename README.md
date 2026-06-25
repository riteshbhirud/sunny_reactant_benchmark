# Reactant.jl on a batched complex dot-product (Sunny.jl KPM hot path) — CPU benchmark

A small, self-contained benchmark of one kernel pattern from **Sunny.jl** (a Julia library for
magnetic neutron-scattering / spin-wave theory). We are exploring whether Reactant is a good fit for
Sunny's GPU/CPU acceleration, and on CPU we see Reactant's XLA backend running this pattern markedly
slower than a hand-vectorized `LoopVectorization` loop. We'd like a Reactant maintainer's read on
whether that gap is fundamental to the current CPU backend or a configuration we've missed.

This directory is standalone — it does **not** depend on Sunny.jl. The kernel is inlined as a
minimal reproducer.

## The operation

Per-chain complex inner product, reduced over the vector dimension, for many chains at once — the
inner-loop reduction of a **batched Lanczos / Kernel-Polynomial-Method (KPM)** solver hot path:

```
result[c] = Σ_k conj(a[c,k]) * b[c,k]     for c in 1:N_chains
a, b :: (N_chains, vec_len)  ::  ComplexF64        # ComplexF64 is the production dtype
```

In Sunny, `N_chains = Nq × Nobs` (q-points × observables) and `vec_len = 2L` (Bogoliubov vector
length). The sizes below span what we actually run. (Sunny's Lanczos is hand-rolled, so KrylovKit is
not involved — N/A.)

## Methods (all ComplexF64; correctness checked against a plain-Julia oracle)

1. **`LoopVectorization @tturbo`** — CPU-SIMD baseline. LV has no native complex, so it is fed the
   mathematically-equivalent split real/imag form (`conj(a)*b`: `re = ar·br + ai·bi`,
   `im = ar·bi − ai·br`). This is the strongest tuned CPU baseline.
2. **`Reactant Ops.dot_general`** — the StableHLO batched-dot primitive (`batching_dimensions`).
3. **`Reactant NNlib.batched_mul`** — another path that lowers to `dot_general`.
4. **`Reactant sum(conj(a).*b; dims=2)`** — the array-reduce idiom (included to show idiom-independence).
5. **`Direct KernelAbstractions @kernel` via OpenCL/PoCL** — the production kernel form (workgroup
   reduction with `@localmem`), compiled through PoCL on CPU.

Reactant calls use `@compile sync=true` (device-synchronized → honest timing). Timing is
BenchmarkTools `@belapsed` (minimum). Each method is also profiled for effective CPU utilization so
the thread-parallelism is explicit.

## Results

Hardware: 2× Intel Xeon Silver 4214R @ 2.40 GHz; **4-CPU cgroup allocation** (`nproc = 4`).
`JULIA_NUM_THREADS = 4`. Julia 1.12.4, Reactant v0.2.267, KernelAbstractions v0.9.41,
LoopVectorization v0.12.174.

ms / call (`@belapsed`):

| N_chains | vec_len | LV `@tturbo` | Reactant `dot_general` | Reactant `batched_mul` | Reactant `reduce` | KA via PoCL | dot_general / LV |
|---------:|--------:|-------------:|-----------------------:|-----------------------:|------------------:|------------:|-----------------:|
| 200  | 800  | 0.046 | 0.672  | 0.647  | 0.616  | 0.156  | **15×** |
| 600  | 1800 | 0.946 | 19.56  | 21.87  | 20.87  | 3.065  | **21×** |
| 1500 | 800  | 1.134 | 19.49  | 19.10  | 18.25  | 3.050  | **17×** |
| 1500 | 1800 | 3.745 | 57.33  | 55.77  | 52.91  | 9.089  | **15×** |
| 1500 | 3600 | 7.700 | 112.8  | 112.6  | 119.4  | 21.21  | **15×** |

- The three Reactant idioms (`dot_general`, `batched_mul`, `reduce`) are within ~10% of each other —
  on CPU the idiom choice does not matter (they compile to the same optimized program; see below).
- All Reactant forms are **~15–21× slower than `@tturbo`** here (up to ~33× at the smallest sizes in
  our broader 24-cell sweep).
- The PoCL KA kernel is ~2–7× faster than Reactant but still ~3–4× slower than `@tturbo`.

### CPU utilization (this is why the comparison is fair)

Same 4-CPU allocation for all methods, at N=1500 / vec_len=3600 (effective cores from
`/proc/self/stat` utime+stime over a hot loop):

| method | % CPU | ≈ cores used |
|--------|------:|-------------:|
| LV `@tturbo`            | 392% | ~3.9 |
| Reactant `dot_general`  | 170% | ~1.7 |
| KA via PoCL             | 390% | ~3.9 |

`@tturbo` and PoCL both saturate the 4 cores; **XLA's `dot_general` reaches only ~1.7 of 4**. So the
gap is *not* a thread-count handicap (all had 4 cores). At N=1500/3600 the 15× wall-clock gap
decomposes into roughly a ~2.3× lower CPU parallel efficiency (1.7 vs 3.9 cores) and the remaining
~6–7× being per-core throughput.

## GPU note (for balance)

This is a CPU report, but for fairness: on an RTX 2080 Ti the kernel-free Reactant `dot_general`
*beats* our current hand-written KA kernel for large sizes (up to ~3.6× complex / ~6× real at
N=1500/vec_len=3600), while losing at small sizes. (Part of that is our KA kernel being under-tuned —
64 threads/chain.) So this is not "Reactant is slow" — it is specifically the **CPU** backend on this
batched-reduction shape that we're asking about.

## The blocker we hit trying `raise=true` on the production kernel (GPU)

We tried Reactant's headline path — consuming the existing KA kernel via `@compile raise=true` —
but it fails because the kernel uses shared memory (`@localmem`/`@synchronize`). A copy kernel
*without* shared memory raises fine; *with* `@localmem` it fails (verified in Float64, so not a
complex issue):

```
error: cannot raise op to stablehlo
  %0 = "llvm.mlir.addressof"() <{global_name = @shmem}> : () -> !llvm.ptr<3>
  %2 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<3>) -> memref<?xf64, 3>
```

`!llvm.ptr<3>` is CUDA shared memory (address space 3). Minimal reproducer (needs `CUDA` loaded for
the KA→PTX→MLIR raising path; not part of this CPU benchmark's env):

```julia
using Reactant, KernelAbstractions, CUDA
@kernel function localmem_copy!(out, @Const(x))              # FAILS to raise
    i = @index(Local, Linear); tile = @localmem Float64 (256,)
    tile[i] = x[i]; @synchronize; out[i] = tile[i]
end
@kernel function plain_copy!(out, @Const(x)); i=@index(Global,Linear); out[i]=x[i]; end   # raises OK
run(k, x) = (out=similar(x); k(KernelAbstractions.get_backend(x), 256)(out, x; ndrange=256); out)
@code_hlo raise=true run(plain_copy!,    Reactant.to_rarray(ones(256)))   # NO ERROR
@code_hlo raise=true run(localmem_copy!, Reactant.to_rarray(ones(256)))   # cannot raise op … !llvm.ptr<3>
```

This is the reason we benchmark the *kernel-free* Reactant forms above rather than raising the
production kernel.

## Verification: all three Reactant idioms compile to the same StableHLO

`@code_hlo` for ComplexF64 (1500, 3600):

```
optimize=false  (raw lowering):
  reduce form      ->  chlo.conj ; stablehlo.multiply ; stablehlo.reduce(add across dim)
  dot_general form ->  chlo.conj ; stablehlo.dot_general(batching=[0]x[0], contracting=[1]x[1])

optimize=true   (after the Enzyme-HLO optimization pipeline):
  reduce form      ->  chlo.conj ; stablehlo.dot_general(batching=[1]x[1], contracting=[0]x[0])   # rewritten
  dot_general form ->  chlo.conj ; stablehlo.dot_general(batching=[1]x[1], contracting=[0]x[0])   # identical
```

So the `reduce` idiom is rewritten into the same `dot_general` as the explicit op — which is why all
three Reactant forms have the same CPU performance. We benchmark all three so the comparison can't be
dismissed as "you used the wrong idiom."

## What we'd like to know

1. Is XLA's CPU `dot_general` reaching only ~1.7 of 4 cores expected for a **batched** dot of this
   shape (many small `(1×vec_len)·(vec_len×1)` contractions), or is there a thread-pool / Eigen
   configuration we should set? We do **not** set `XLA_FLAGS`; is there a CPU `XLA_FLAGS` (fast-math,
   vectorization, intra-op threads) that would materially change these numbers?
2. Independent of parallelism, the per-core throughput is ~6–7× below a hand-vectorized loop. Is the
   XLA CPU backend expected to trail LoopVectorization on this batched-reduction pattern, or are we
   missing something (layout, precision flags, `donated_args`, etc.)?
3. Any guidance on the `@localmem` raising path (the `!llvm.ptr<3>` error) — is shared-memory kernel
   raising on a roadmap, or is the kernel-free rewrite the intended route?

We're happy to run any configuration you suggest and report back.

## How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
JULIA_NUM_THREADS=4 julia --project=. benchmark.jl
```

(`Project.toml` + `Manifest.toml` pin exact versions. `pocl_jll` provides the CPU OpenCL device;
`benchmark.jl` loads it before `OpenCL`.) Raw output we recorded is in `benchmark_output.txt`.

### Measurement methodology (for replication)
- Reactant: `compiled = @compile sync=true f(a_ra, b_ra)` once, then `@belapsed $compiled($a_ra, $b_ra)`.
- LV: `@belapsed lv!($re,$im,$ar,$ai,$br,$bi)` (split real/imag).
- KA-PoCL: `@belapsed (kernel!(OpenCLBackend(),64)(...); KernelAbstractions.synchronize(be))`.
- Correctness: every method `≈` a plain-Julia oracle (`rtol` 1e-7..1e-8).
- CPU utilization: utime+stime delta from `/proc/self/stat` over a fixed-count hot loop ÷ wall time.

## Provenance / version note
The broader exploration this came from used Reactant v0.2.266; this self-contained env resolved to
v0.2.267 (one patch newer). Both show the same behavior. CPU benchmarks are 4-thread on the Xeon
4214R allocation above; on a larger allocation the absolute numbers change but the LV-vs-Reactant
ratio and the ~1.7-core XLA utilization were stable in our testing.
