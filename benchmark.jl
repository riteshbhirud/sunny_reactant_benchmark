# benchmark.jl — batched complex dot-product "chains" (Sunny.jl KPM/Lanczos hot path).
# Self-contained:  julia --project=. benchmark.jl   (no Sunny.jl needed).
#
# Operation (ComplexF64 throughout — the production dtype):
#     result[c] = Σ_k conj(a[c,k]) * b[c,k]     for c in 1:N_chains,  a,b :: (N_chains, vec_len)
# This is the per-chain inner product reduced in a batched Krylov (Lanczos) solver.
#
# Methods compared:
#   1. LoopVectorization @tturbo  — CPU-SIMD baseline. LV has no native complex, so we feed it the
#      mathematically-equivalent split real/imag form (conj(a)*b: re=ar·br+ai·bi, im=ar·bi−ai·br).
#   2. Reactant Ops.dot_general   — the StableHLO batched-dot primitive.
#   3. Reactant NNlib.batched_mul — another path that lowers to dot_general.
#   4. Reactant sum(conj(a).*b;dims=2) — the array-reduce idiom (included to show idiom-independence).
#   5. Direct KernelAbstractions @kernel via OpenCL/PoCL — the production kernel form, run on CPU.
# All Reactant calls use `@compile sync=true` (device-synchronized → honest timing).
# Timing: BenchmarkTools `@belapsed` (minimum). We also report effective CPU utilization (cores used)
# so the thread-parallelism of each method on the same CPU allocation is explicit.

using pocl_jll, OpenCL                 # pocl_jll MUST be loaded before OpenCL
using Reactant, KernelAbstractions, LoopVectorization, NNlib, BenchmarkTools, Printf, LinearAlgebra
const Ops = Reactant.Ops
const KA  = KernelAbstractions
const WG  = 64
Reactant.set_default_backend("cpu")

alloc_cpus = try parse(Int, readchomp(`nproc`)) catch; Sys.CPU_THREADS end
println("="^78)
@printf("Julia %s | Reactant v%s | KernelAbstractions v%s | LoopVectorization v%s\n",
        VERSION, pkgversion(Reactant), pkgversion(KernelAbstractions), pkgversion(LoopVectorization))
@printf("CPU allocation (nproc) = %d   |   JULIA_NUM_THREADS = %d (LV @tturbo parallelism)\n",
        alloc_cpus, Threads.nthreads())
try println("CPU model:", split(readchomp(`grep -m1 "model name" /proc/cpuinfo`), ":")[2]) catch end
println("="^78)

# (5) production kernel: per-chain workgroup reduction with @localmem (minimal BatchedDotChains).
@kernel inbounds=true function bdot_kernel!(result, a, b, twoL)
    chain=@index(Group,Linear); lid=@index(Local,Linear); gs=@uniform @groupsize()[1]
    lr=@localmem Float64 (WG,); li=@localmem Float64 (WG,)
    acc=ComplexF64(0); k=lid
    while k<=twoL; acc+=conj(a[chain,k])*b[chain,k]; k+=gs; end
    lr[lid]=real(acc); li[lid]=imag(acc); @synchronize
    if lid<=32; lr[lid]+=lr[lid+32]; li[lid]+=li[lid+32]; end; @synchronize
    if lid<=16; lr[lid]+=lr[lid+16]; li[lid]+=li[lid+16]; end; @synchronize
    if lid<=8;  lr[lid]+=lr[lid+8];  li[lid]+=li[lid+8];  end; @synchronize
    if lid<=4;  lr[lid]+=lr[lid+4];  li[lid]+=li[lid+4];  end; @synchronize
    if lid<=2;  lr[lid]+=lr[lid+2];  li[lid]+=li[lid+2];  end; @synchronize
    if lid==1; result[chain]=ComplexF64(lr[1]+lr[2], li[1]+li[2]); end
end
ka_run!(dr,da,db,L,be) = (bdot_kernel!(be,WG)(dr,da,db,L; ndrange=size(da,1)*WG); KA.synchronize(be); dr)

re_reduce(a,b) = vec(sum(conj(a).*b; dims=2))
re_dg(a,b)     = Ops.dot_general(Ops.conj(a), b; contracting_dimensions=([2],[2]), batching_dimensions=([1],[1]))
function re_bmm(a,b)
    N,L=size(a); A3=reshape(permutedims(Ops.conj(a),(2,1)),1,L,N); B3=reshape(permutedims(b,(2,1)),L,1,N)
    vec(NNlib.batched_mul(A3,B3))
end
function lv!(re,im,ar,ai,br,bi)
    @tturbo for c in axes(ar,1)
        x=0.0; y=0.0
        for k in axes(ar,2); x+=ar[c,k]*br[c,k]+ai[c,k]*bi[c,k]; y+=ar[c,k]*bi[c,k]-ai[c,k]*br[c,k]; end
        re[c]=x; im[c]=y
    end
end
cputime()=(s=split(read("/proc/self/stat",String)); (parse(Int,s[14])+parse(Int,s[15]))/100.0) # utime+stime sec
pcpu(f,reps)=(f(); GC.gc(); c0=cputime(); w0=time(); for _ in 1:reps; f(); end; 100*(cputime()-c0)/(time()-w0))

ocl = OpenCLBackend()
sizes=[(200,800),(600,1800),(1500,800),(1500,1800),(1500,3600)]
@printf("\nms/call (BenchmarkTools @belapsed); all five correct vs a plain-Julia oracle.\n\n")
@printf("%-5s %-7s | %-10s %-9s %-9s %-9s | %-9s | %-10s\n",
        "N","vec_len","LV@tturbo","Re-dg","Re-bmm","Re-reduce","KA-PoCL","Re-dg / LV")
println("-"^92)
rows=[]
for (N,L) in sizes
    a=randn(ComplexF64,N,L); b=randn(ComplexF64,N,L)
    ref=ComplexF64[sum(conj(view(a,c,:)).*view(b,c,:)) for c in 1:N]
    ar=real.(a); ai=imag.(a); br=real.(b); bi=imag.(b); re=zeros(N); im=zeros(N)
    lv!(re,im,ar,ai,br,bi); @assert isapprox(complex.(re,im),ref;rtol=1e-8) "LV wrong"
    t_lv=1e3*@belapsed lv!($re,$im,$ar,$ai,$br,$bi)
    ard=Reactant.to_rarray(a); brd=Reactant.to_rarray(b)
    cdg=@compile sync=true re_dg(ard,brd);     @assert isapprox(Array(cdg(ard,brd)),ref;rtol=1e-7) "dg wrong"
    cbm=@compile sync=true re_bmm(ard,brd);    @assert isapprox(Array(cbm(ard,brd)),ref;rtol=1e-7) "bmm wrong"
    crd=@compile sync=true re_reduce(ard,brd); @assert isapprox(Array(crd(ard,brd)),ref;rtol=1e-7) "reduce wrong"
    t_dg=1e3*@belapsed $cdg($ard,$brd); t_bm=1e3*@belapsed $cbm($ard,$brd); t_rd=1e3*@belapsed $crd($ard,$brd)
    da=KA.allocate(ocl,ComplexF64,N,L); copyto!(da,a); db=KA.allocate(ocl,ComplexF64,N,L); copyto!(db,b); dr=KA.allocate(ocl,ComplexF64,N)
    ka_run!(dr,da,db,L,ocl); @assert isapprox(Array(dr),ref;rtol=1e-8) "KA-PoCL wrong"
    t_ka=1e3*@belapsed ka_run!($dr,$da,$db,$L,$ocl)
    push!(rows,(N,L,t_lv,t_dg,t_bm,t_rd,t_ka))
    @printf("%-5d %-7d | %-10.3f %-9.3f %-9.3f %-9.3f | %-9.3f | %.0f×\n",N,L,t_lv,t_dg,t_bm,t_rd,t_ka,t_dg/t_lv)
end

# effective CPU parallelism at the largest cell (same allocation for all methods)
N,L=1500,3600; a=randn(ComplexF64,N,L); b=randn(ComplexF64,N,L)
ar=real.(a);ai=imag.(a);br=real.(b);bi=imag.(b);re=zeros(N);im=zeros(N)
ard=Reactant.to_rarray(a); brd=Reactant.to_rarray(b); cdg=@compile sync=true re_dg(ard,brd)
da=KA.allocate(ocl,ComplexF64,N,L);copyto!(da,a);db=KA.allocate(ocl,ComplexF64,N,L);copyto!(db,b);dr=KA.allocate(ocl,ComplexF64,N)
pl=pcpu(()->lv!(re,im,ar,ai,br,bi),200); pd=pcpu(()->cdg(ard,brd),100); pk=pcpu(()->ka_run!(dr,da,db,L,ocl),200)
@printf("\nEffective CPU utilization @ N=1500 vec_len=3600 (all under the same %d-CPU allocation):\n", alloc_cpus)
@printf("  LV @tturbo : %4.0f%% CPU (~%.1f cores)\n", pl, pl/100)
@printf("  Re-dg (XLA): %4.0f%% CPU (~%.1f cores)\n", pd, pd/100)
@printf("  KA-PoCL    : %4.0f%% CPU (~%.1f cores)\n", pk, pk/100)
println("\nBENCHMARK_DONE")
