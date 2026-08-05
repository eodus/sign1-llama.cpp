# LLM Binarization — Episode One: DBF aka _Addition Is (Almost) All You Need_

**Sasha Shlemov**, with **Drinkins, personal AI assistant** ·
[Runtime](https://github.com/eodus/sign1-llama.cpp) ·
[Scripts and results](https://github.com/eodus/dbf-sign1) ·
[Model](https://huggingface.co/eodus/Qwen3.6-35B-A3B-DBF-SIGN1)

**TL;DR:** Structural binary quantization for llama.cpp, demonstrated on the routed experts of
Qwen3.6-35B-A3B: slightly smaller than Q2_K, better model quality, and approximately the same Vulkan speed.

Can we replace most of an LLM's weights with `+1` and `-1`, making matrix multiplication effectively
addition? And does it have any merit?

The answer is **yes** to both questions. For the first question, there are two approaches. We can train a model
in a binary or ternary format from the beginning. BitNet is the best-known example
[[1]](#ref-bitnet) [[2]](#ref-bitnet158). Or we can train a floating-point model first and convert it to binary
afterward. The strongest method we found for this second path is **Double Binary Factorization (DBF)**,
introduced by Vladimír Boža and Vladimír Macko of Comenius University Bratislava [[4]](#ref-dbf). This series
begins with their method and the systems work required to make it a runnable model.

The merit is also two-fold.

**First: performance.** Binary weights replace general weight multiplication with sign selection, addition, and
popcount arithmetic. On our general-purpose consumer hardware, two binary factor products plus three diagonal
scalers run within 1.63% in prompt processing and 0.46% in token generation of one Q2_K product. The opportunity
becomes larger when the chip is designed for binary arithmetic rather than asked to emulate it through general
floating-point or integer multipliers. Microsoft's BitNet work demonstrates both the software and hardware potential
of this direction [[2]](#ref-bitnet158) [[3]](#ref-bitnetcpp).

**Second: quantization quality.** DBF established that a *structural*, SVD-like binary decomposition can
approximate LLM matrices better than leading *scalar* quantizers at the same storage budget [[4]](#ref-dbf).
We reproduce that result against the practical Q2_K baseline and test whether the matrix-level advantage
survives in a real inference stack.

Therefore binary models potentially give us both better speed and better quality at the same storage budget.
Yes, it sounds ambitious. In this series, we will defend that claim.

Post-training structural binarization remains underdeveloped. DBF provides one unusually strong algorithmic
result, but the surrounding inference ecosystem is largely missing.

Boža and Macko represent a dense weight matrix in the form D-B-D-B-D
(diagonal–binary–diagonal–binary–diagonal) and provide an efficient factorization algorithm. Their paper
establishes the algorithmic case: in the approximately two-bit-per-original-matrix-element regime, DBF is
competitive with or better than leading quantization methods. Our same-matrix measurements sharpen that result
for the practical opponent we care about here: on the matrices we tested, DBF gives a lower Frobenius residual
than llama.cpp's Q2_K at the same bit budget.

The paper focuses on the decomposition algorithm and provides a reference implementation. The remaining systems
work was open: a packed model format, a mainstream inference-engine path, and optimized kernels for comparing
DBF with Q2_K as a complete system. We could not find a community implementation covering that stack. This post
closes the end-to-end implementation gap.

Concretely, we present:

1. the [exact binary weights](https://huggingface.co/eodus/Qwen3.6-35B-A3B-DBF-SIGN1) used in our experiments:
   Qwen3.6-35B-A3B [[5]](#ref-qwen36) with its routed FFN experts converted to DBF at a Q2_K-scale bit budget;
2. the conversion pipeline used for the tested Qwen3.6-35B-A3B checkpoint. Fine-tunes and merged variants that
   preserve its exact tensor structure should require no architectural change, but we have not validated them.
   Dimensionally different hybrid-MoE models, such as Qwen3.5-35B-A3B [[10]](#ref-qwen35-35b),
   Qwen3.5-122B-A10B [[11]](#ref-qwen35-122b), and Qwen3.5-397B-A17B [[12]](#ref-qwen35-397b), require explicit
   parameterization and testing;
3. Vulkan, CPU, and CUDA/HIP kernels integrated into our public llama.cpp branch
   [[6]](#ref-sign1-llama). The optimized Vulkan path reaches near-parity with llama.cpp's Q2_K implementation
   on our hardware;
4. benchmarks showing better model quality at approximately the same size and speed as vanilla Q2_K.

The immediate motivation came from Antirez's DS4. Through considerable systems work, DS4 runs a DeepSeek-class
MoE model on a 128 GB AMD Strix Halo using a mixed IQ2_XXS/Q2_K/Q8 GGUF
[[7]](#ref-ds4). That result made the practical question unavoidable: at a budget where every quantum of quality
matters, can structural binarization preserve more of the model than aggressive scalar quantization? This led us to DBF,
then to SIGN1 and the complete DBDBD inference path described here.

Efficient kernels are hard, but a capable modern LLM makes a working implementation possible in reasonable
time. Correct and reasonably fast is tractable; finished is an iterative community process. We provide the
representation contract, the optimized Vulkan implementation used on our Radeon 8060S / Ryzen AI Max+ 395
system [[8]](#ref-halo), and working CPU and CUDA/HIP starting points. This post, the public branch, and the
specification of another target should give a capable LLM enough material to produce a useful first port.

We expect binary models to be useful beyond our poor Halo rig, but we cannot verify every architecture
ourselves. The most interesting next directions are:

1. Metal on Apple Silicon (the Mac Mini case);
2. a proper implementation and measurements on real NVIDIA CUDA hardware (our Halo runs llama.cpp's
   CUDA-source path through HIP, but that path is slower than Vulkan on this machine);
3. ARM NEON (the Raspberry Pi case). Not because it is necessarily the most practical target, but because
   it opens another part of the design space.

---

## 1. The representation

A conventional dense linear layer computes

```text
y = W x,
```

where `W ∈ R^(m×n)`. Start with the singular value decomposition. SVD tells us that any real matrix can be
represented exactly as a sum of rank-one outer products, or equivalently as two factor matrices with singular
values between them. If we keep fewer components, the representation becomes approximate; if we add components,
it converges to the original matrix.

Now impose one additional constraint: both factor matrices must contain only `-1` and `+1`. Exact SVD factors do
not generally satisfy that constraint, so the decomposition becomes approximate. We can compensate by using more
binary components. The role of the intermediate dimension `r` is therefore familiar from low-rank factorization:
more components provide more capacity, at a predictable storage cost. The representation gains capacity as `r`
grows, although unlike SVD, DBF does not produce one nested, ordered sequence of exact prefixes.

DBF adds diagonal scalers around and between the binary factors. At the representation level, their purpose is
simple: binary matrices carry signs and structure but very little magnitude information; the diagonals restore
input, intermediate, and output scales with little additional storage. Understanding how the optimizer finds
those scalers belongs to the DBF paper rather than this implementation story.

DBF therefore approximates `W` as

```text
W ≈ D_out · U · D_mid · V · D_in,
```

with

```text
D_in  ∈ R^(n×n),       V ∈ {-1,+1}^(r×n),
D_mid ∈ R^(r×r),       U ∈ {-1,+1}^(m×r),
D_out ∈ R^(m×m).
```

All three `D` matrices are diagonal. `V` is the input-side binary factor and `U` is the output-side binary
factor, matching the tensor names used by the implementation. The two binary matrices provide structural
capacity; the three real-valued diagonals recover magnitude information that pure signs cannot carry.

A binary matrix needs one bit per value. We use

```text
0 -> +1
1 -> -1
```

because it follows the sign-bit convention used by IEEE 754 floating-point formats: zero denotes a positive
sign and one denotes a negative sign. It does not make SIGN1 an IEEE 754 format; it merely keeps the sign
semantics unsurprising across serialization and kernels.

Multiplication by a binary weight is then a choice between `x` and `-x`. The binary core needs sign extraction,
conditional negation, and accumulation. The diagonal factors still require real-valued multiplication. Hence
“almost” in the original paper's title.

This representation is structurally different from ordinary scalar quantization. Scalar formats approximate
each weight locally and store shared metadata per block. In uniform formats, the metadata is typically a scale
(and sometimes an offset or nested sub-block scales) applied to fixed integer levels; in non-uniform formats, it
may select or parameterize a small codebook. DBF instead approximates the matrix as a linear operator, using two
learned binary maps and explicit diagonal magnitude channels. It is not merely another way to pack two bits per
weight.

### 1.1 Bit budget

The storage budget is transparent. Ignoring byte alignment for the moment, the two binary factors require

```text
r·n + m·r
```

bits. If the three diagonals are stored in FP16, they add

```text
16·(n + r + m)
```

bits. Relative to the original `m·n` matrix elements, DBF therefore costs

```text
bpe_DBF = [r·(m+n) + 16·(m+n+r)] / (m·n)
```

bits per original matrix element.

Qwen3.6-35B-A3B makes the arithmetic particularly clean. Every routed gate and up projection has shape
`512×2048`; every routed down projection has the transposed shape `2048×512` [[5]](#ref-qwen36). Thus all
30,720 routed expert matrices (40 layers × 256 experts × 3 projections) have the same number of elements and
the same DBF storage cost.

For `r=1024`, one projection requires

```text
binary factors: 512·1024 + 1024·2048       = 2,621,440 bits
FP16 diagonals: 16·(512 + 1024 + 2048)     =    57,344 bits
total:                                      2,678,784 bits
```

Dividing by `512·2048` original elements gives exactly `2.5546875` bits per element. Q2_K costs
`2.625` bits per element, or `2,752,512` bits for the same matrix. DBF is therefore 9,216 bytes smaller per
projection. Across all routed experts, the raw tensor payload is `9.5801 GiB` for DBF versus `9.8438 GiB`
for Q2_K, a 270 MiB margin before container-level alignment and metadata.

So `r=1024` is not vague "approximate parity." It is a slightly smaller representation at the Q2_K budget.
The intermediate dimension also gives us a useful quality knob: changing `r` changes the storage budget in
predictable increments instead of forcing a choice among a few fixed quantization formats.

The choice of 1024 has a second, mechanical advantage. It is a power of two and divisible by the natural
packing, SIMD, subgroup, and matrix-tile widths used by current CPU and GPU hardware. That removes padding
and tail handling from the main contraction and makes specialized kernels substantially cleaner.

### 1.2 What we inherit from DBF

We start from the fixed-dimension factorization core described by Boža and Macko [[4]](#ref-dbf) and
implemented in their public Llama3 notebook at
[`usamec/double_binary@bc00edb`](https://github.com/usamec/double_binary/commit/bc00edb). The only refinement
step we add is one final linear least-squares reprojection of the middle diagonal for the fixed binary signs and
outer scalers; alternating projection can leave stale coefficient and rounding error.

**DBF is not progressive like SVD.** An SVD produces an ordered sequence of components: compute it once,
truncate it at different ranks, and obtain a family of approximations. The core DBF factorization chooses the
intermediate dimension `r` before optimization and solves the complete factorization at that dimension. A
dimension-1024 result is not a dimension-512 result with another 512 components appended.

**Larger middle dimensions are harder to optimize well.** In the paper's unweighted “Properties of DBF”
experiment, DBF beats scalar quantization from 1–3 bits, while scalar quantization wins at 4 bits and above. In
the accepted TMLR revision, Boža and Macko address this with size annealing: they spend 80% of the iterations at
a 2-bit middle dimension, expand it gradually during the final 20%, and report slight improvements at 3, 4, and
6 bits. They interpret the result as evidence that the higher-bit limitation is algorithmic rather than
fundamental to DBF [[4]](#ref-dbf).

The annealing result was added during peer review in direct response to a reviewer's higher-bit concern. The
accepted supplementary material and linked public repository do not include its implementation. Since the
technique targets higher-bit overranking rather than the Q2-scale budget used here, this work does not depend
on it.

**The reference implementation is fast locally and expensive globally.** One expert matrix takes only seconds
on our GPU. Qwen3.6-35B-A3B contains 30,720 routed expert matrices. Even with parallel workers and resumable
checkpoints, a faithful 80-iteration whole-model decomposition takes days. “Fast algorithm” and “fast model
conversion” are different claims when the model contains tens of thousands of independent matrices.

**Its practical strength is that it finds useful structure at extremely low budgets.** DBF jointly learns two
binary maps and three magnitude channels instead of quantizing weights independently. Around the Q2_K storage
regime, this gives substantially better matrix approximations than scalar quantization on the tensors we tested.
The method is compact, reproducible, GPU-friendly, and sufficiently strong to justify building the complete
inference stack around it.

DBF is therefore the foundation of this implementation, not the final algorithm for every rank and tensor. The
representation may outlive this particular optimizer.

---

## 2. Why implement it in llama.cpp?

A Python reconstruction answers an important but narrow question: does the factorization approximate an
individual matrix? We wanted answers about a usable model, in this order:

1. **Can real people actually run and use it?** If the model cannot be loaded into an ordinary inference stack,
   the beauty of the mathematics does not matter.
2. **What is its real quality?** Does the matrix-level advantage survive full-model perplexity, synthetic tests,
   and simply talking to the model?
3. **What is its real speed?** Not an operation count or an isolated microbenchmark, but end-to-end performance
   under real workloads on real hardware.

Answering these questions requires the less glamorous machinery underneath: serialization, model
loading, graph construction, routed execution, intermediate representations, backend kernels, and correctness
tests. llama.cpp is a useful battlefield because it already contains mature quantization formats, backend
abstractions, model loaders, graph allocators, and benchmark tools. A new representation has to coexist with all
of them. There is nowhere for a missing dependency or an accidental dense fallback to hide for long.

More importantly, llama.cpp is a de facto community standard for local inference [[9]](#ref-llamacpp). It
supports a wide range of model families, quantization formats, and hardware backends. A format implemented only
in our experimental runner would demonstrate possibility. A format implemented in llama.cpp can be downloaded,
benchmarked, compared with existing quantizers, and extended by the people who already maintain those backends.

---

## 3. A real one-bit tensor type

llama.cpp already has Q1_0, and our first prototype used it as a sign carrier. We separated SIGN1 for three
reasons. Q1_0 is a quantization format with its own block scaling and execution contract; DBF already carries
its scales explicitly in `D_in`, `D_mid`, and `D_out`; and Q1_0 is evolving independently. Coupling DBF to it
would make two unrelated representations depend on each other's storage and kernel decisions.

SIGN1 therefore stores exactly one thing: signs. Its primitive block contains 64 signs in one `uint64_t`.

Why 64?

- It matches the natural storage width.
- It is portable across CPU and GPU backends.
- Wider SIMD kernels can combine several blocks internally.
- The file format does not overfit one machine’s preferred vector width.

The GGUF therefore contains actual `SIGN1` tensors, not Q1_0 used informally and not a hidden dense
reconstruction. The type is wired through serialization, loading, backend dispatch, and correctness tests. The
implementation details are in the code; or, more realistically, your AI will read them =).

This model requires our SIGN1/DBDBD llama.cpp branch; stock llama.cpp does not understand the tensor type or the
factorized Qwen graph. Reusing Q1_0 as the sign carrier would not remove that requirement. The runtime would still
need the custom `D_in -> V -> D_mid -> U -> D_out` graph, routed scaler handling, and architecture-specific kernel
dispatch. Q1_0 can disguise the storage type; it cannot make DBDBD a stock model architecture.

---

## 4. Teaching the model loader DBDBD

For each routed expert projection, the model may contain five tensors:

```text
projection_din
projection_v
projection_dmid
projection_u
projection_dout
```

Here `v` and `u` are packed `SIGN1` matrices. The diagonal tensors are currently stored in FP16.

The graph computes, in order:

```text
                       packed SIGN1       packed SIGN1
x ── D_in ──────────────── V ── D_mid ─────── U ── D_out ── y

Optimized Vulkan implementation:
MMQ / prompt processing: Q8 bitplanes
MMV / token generation:  floating activations with mix()
```

Equivalently:

```text
x_scaled = D_in · x
h        = V · x_scaled
h_scaled = D_mid · h
y_core   = U · h_scaled
y        = D_out · y_core
```

In exact arithmetic, the factorized expression is simply a matrix product. In finite arithmetic, matrix
multiplication is not associative: rounding and activation quantization make
`U·(D_mid·(V·x))` numerically different from multiplying `x` by a pre-reconstructed dense matrix.
The distinction becomes more important at very low precision because errors enter at every intermediate
boundary. We did not observe a material quality problem from this effect in the routed FFN implementation, but
any new backend or activation format must validate the deployed chain, not only the reconstructed matrix.

For a mixture-of-experts model, every selected `(token, slot)` route also carries an expert identity and a route
weight. That creates two different input topologies:

- gate/up first factors consume one hidden-state vector shared by several selected experts;
- down projections and second factors consume route-specific intermediates already attached to one expert.

Treating these as the same layout is correct algebraically and poor mechanically. Much of the Vulkan work was
teaching the runtime to preserve those distinctions without changing the DBDBD function.

This is why prompt processing is much faster than single-token generation in the table in Section 8. During
prefill, one expert matrix is applied to many vectors, so weight loads, routing setup, and conversion can be
amortized across columns. Generation usually applies that matrix to one vector at a time. The same arithmetic
therefore belongs to two different kernel regimes: MMQ for many vectors and MMV for one.

---

## 5. Vulkan: where the representation became practical

Vulkan is the backend we optimized seriously because it is the fastest llama.cpp backend for this model and
workload on our Strix Halo.

An efficient matrix multiplication kernel is far beyond the high-school “row times column” algorithm. It requires
specialized work partitioning, memory layouts, reductions, and direct use of low-level hardware instructions. The
implementation catalogue below is primarily for readers who are about to write or port a kernel; everyone else
can safely skip to the [measured result](#8-current-measurements).

We tried several ways to execute a binary dot product:

1. **DP4A:** expand packed signs to integer `+1/-1` values and use the conventional four-way signed
   INT8 dot-product-with-INT32-accumulate instruction [[13]](#ref-dp4a) against Q8 activations. This is the
   established BitNet-style GPU baseline. It is fast, but the binary weights become ordinary integers before
   arithmetic and we lose the "addition only" narrative. Our objection is not purely ideological, although it is
   partially ideological =). We implemented and measured DP4A; `mix()` and bitplanes were faster on our
   hardware. That is a hardware-specific result, not an argument to skip the most standard baseline on another GPU.
2. **Conditional negation and addition:** read a sign bit and add either `x` or `-x` directly. This preserves the
   intended arithmetic but relies on the compiler and hardware to implement sign selection efficiently.
3. **GLSL `mix()`:** use the Boolean-selector overload defined by GLSL, so `mix(x, -x, sign)` selects
   component-wise between `x` and `-x` [[14]](#ref-glsl-mix). Folklore says that such a high-level conditional
   should be slow. On RADV and gfx1151 it works remarkably well; measurement beats folklore.
4. **Bitplanes:** floating-point values carry separate exponents, while one integer bitplane reduction needs a
   common scale. We therefore *row-denormalize* the activation: choose one exponent for the reduction row,
   encode one sign plane and seven magnitude planes, combine activation and weight signs with XOR, and evaluate
   each magnitude plane with masked popcounts and shifts.

   The matched Vulkan Q2_K path already quantizes its input activations to Q8_1. Our sign-magnitude Q8 bitplanes
   use a comparable precision budget, but reorganize the activation for binary-weight arithmetic. The resulting
   core is `XOR + AND + POPCOUNT + SHIFT + ADD`, followed by one shared scale. Whether this representation can become
   a general activation contract, rather than a backend-private encoding, remains open.
5. **Split rank:** partition the intermediate dimension, compute independent partial outputs, and sum them
   elementwise. We tested explicit reductions and atomic accumulation. Atomics were much less disastrous than
   generic GPU folklore predicts, although they did not become the final path.
6. **Precomputed signed sums:** for each activation quartet `(x1,x2,x3,x4)`, precompute the possible
   `±x1±x2±x3±x4` combinations and select one using the four packed weight bits. This replaces repeated sign
   application with a small lookup. On Halo, dynamic indexing and register pressure consumed the arithmetic
   saving, but the balance may differ on hardware with native table-lookup instructions.
7. **Weight repacking:** the tensor format in the model file does not have to be the format consumed by the
   kernel. GGUF stores one canonical portable SIGN1 layout; a backend may repack it once during model loading
   into the exact row groups and K-chunks consumed by its lanes. This is effectively free preprocessing when the
   model is loaded once and used many times. The backend API must still return canonical bytes when a tensor is
   read back; violating that boundary gave us a particularly educational false CPU reference. Repacking simplified
   addressing consistently, although end-to-end gains were often marginal because canonical adjacent rows were
   already cache-friendly.
8. **Input repacking:** activations have the same design freedom, but they change every evaluation and therefore
   must repay conversion cost. Globally, route-specific vectors can be compacted into expert-major order so one
   expert kernel consumes contiguous inputs; genuinely shared gate/up inputs should remain shared rather than be
   duplicated eight times. Locally, values can be arranged in the exact words, bytes, or sign/magnitude planes
   consumed by each lane. Both levels matter: global order controls routing locality, while local order controls
   coalescing and instruction shape.
9. **LDS and register caching:** stage activation or weight tiles in shared memory, then optionally retain hot
   fragments in registers. Some caches were load-bearing; others duplicated work that the hardware cache already
   handled and made performance worse through barriers or occupancy loss. "Cache it" is not an optimization
   until the complete kernel proves it.
10. **Scheduling and fusion:** we tested compact expert queues, skipped empty tiles, persistent workgroups,
    gate/up overlap, direct V-to-U carriers, and fusion of adjacent diagonal and SwiGLU operations. The largest
    gains often came from removing null work or an intermediate boundary rather than changing the inner dot
    product.

Another folklore claim is that inference is entirely memory-bound, so memory traffic is everything and compute
is free. Just no. Our kernels often moved essentially the same bytes and still differed materially because of
integer instruction count, dependency chains, subgroup reductions, barriers, register pressure, and exposed
workgroup parallelism. Conversely, reducing activation precision removed arithmetic and traffic but sometimes
changed almost nothing end to end because that part of the kernel was not the bottleneck. Count the bytes, but
also inspect the instructions and measure the complete schedule.

We ran substantially more variants than we can present properly in one post. The intermediate sources,
measurements, and rejected branches are preserved, but turning every experiment into publication-quality prose
would become a second full-time job. If you are implementing one of these paths and need the underlying result,
contact us. We are happy to share the relevant artifacts and discuss what we observed.

The final implementation keeps two different winners. Prompt processing (MMQ) uses bitplanes; single-token
matrix-vector multiplication uses `mix()`. There is no reason to force one arithmetic formulation onto workloads
with different shapes and parallelism.

The measured result is narrower and more useful than the slogan: DBDBD evaluates **two** binary matrix products
plus three explicit diagonal scalers in approximately the time Q2_K evaluates **one** quantized matrix product.
The comparison includes the complete model path, not an operation-count thought experiment. DBDBD performs more
logical stages and still reaches near-parity because its binary cores replace general weight multiplication with
sign selection, integer addition, or popcount arithmetic. The scalers are not free; they are included in the
measured result.

Performance depended as much on execution *regimes* as on arithmetic: how many rows and columns one
workgroup owns, how many workgroups are exposed to the scheduler, which intermediates use expert-major order,
and where conversion or scaling is fused. The retained implementation uses one-pass row encoding, SWAR
bit-matrix transpose, expert-major route-specific layouts, shared routing plans, adaptive tail skipping, and
shape-specific rows2/rows4/rows8 generation kernels.

If you optimize SIGN1 for another architecture, try all the formulations rather than porting our winner blindly.
The final race on Halo was close, and small changes in work ownership repeatedly reversed apparently obvious
conclusions. In particular, test several row/column/workgroup regimes.

---

## 6. CPU: a portable baseline

The CPU path quantizes activations to Q8, applies packed signs during the dot product, and uses AVX2 or
AVX-512 when available. The AVX-512 prompt path repacks eight output rows together and evaluates several input
vectors per call; AVX2 provides the narrower vector path, with a scalar implementation as the correctness
fallback. It is functional and tested, but not the performance focus here.

---

## 7. CUDA/HIP: a starting point, not a polished backend

We wired SIGN1 and routed DBF scalers through llama.cpp's CUDA-source MMV/MMQ implementation and tested that
code through HIP on our AMD machine. The full model runs correctly, although HIP currently retains a bounded
stream synchronization fallback. We did not optimize this backend seriously, and we have no suitable NVIDIA
hardware on which to validate native CUDA performance.

There is a concrete reason to expect more from native CUDA. In the ordinary portable compute model used by
our Vulkan backend, synchronization is local to a subgroup or workgroup. A dependency across the complete
dispatch requires ending the dispatch and synchronizing before another one [[15]](#ref-vulkan-sync). The
pipeline is therefore close to low-level map-reduce: launch homogeneous work, materialize the result, synchronize,
then launch the consumer. Vulkan can dispatch different pipelines, but ordinary indirect dispatch changes the
grid dimensions, not the bound shader itself.

CUDA exposes additional execution mechanisms: cooperative groups can synchronize a cooperatively launched grid,
and dynamic parallelism permits device-side kernel launches [[16]](#ref-cuda-guide). We have not implemented the
following designs, but two directions look especially promising:

1. **Count-specialized MoE kernels.** The number of vectors routed to each expert is known only at runtime. Instead
   of one kernel padded to the largest tile, dispatch specialized 16-, 8-, 4-, 2-, and 1-row kernels according to
   the observed expert counts. Vulkan can approximate this with several dispatches and queues; CUDA offers a more
   natural GPU-driven implementation.
2. **A joint V–D_mid–U kernel.** Compute `V·x`, apply `D_mid`, synchronize the grid, then execute `U·h`
   without returning control to the host-side graph between phases. A global
   barrier does not magically eliminate intermediate storage, but it can remove dispatch boundaries and permit a
   persistent work partition that keeps some data local.

These are hypotheses, not CUDA performance claims. The Vulkan implementation establishes the baseline; a native
CUDA port has a richer synchronization model with which to challenge it.

## 8. Current measurements

The final Vulkan comparison uses the faithful DBF weights and the same tensor shapes, storage layout, and nominal
operation count as the earlier runtime controls. We nevertheless reran speed with the exact release model because
changed hidden states can alter routing and the realized workload.

The matched Vulkan result used drift-balanced Q2_K → DBDBD → DBDBD → Q2_K order, identical non-expert
model tensors, and the same command line. The FP16-expert control was measured separately with that binary and
command shape; it is shown for scale, not included in the ABBA contrast:

| Claim | Original FP16 experts | Q2_K | DBF/SIGN1 | DBF vs Q2_K |
|---|---:|---:|---:|---:|
| Vulkan prompt processing, final DBDBD model, pp512 | 681.31 | 1217.95 | 1198.09 | -1.63% |
| Vulkan token generation, final DBDBD model, tg128 | 54.71 | 82.52 | 82.14 | -0.46% |
| Full WikiText-2 perplexity, final DBDBD model | 6.9935 | 7.7823 | **7.3186** | **58.8% of Q2_K damage removed** |

All three quality models share the same non-expert remainder and differ only in the routed expert representation.
The raw PPL reduction is less informative than the recovered gap to the original FP16-expert model. Q2_K adds
`7.7823 - 6.9935 = 0.7888` PPL; DBF leaves only `7.3186 - 6.9935 = 0.3251`. Thus DBF removes **58.8%** of
Q2_K's additive PPL damage. On the underlying mean-NLL scale, it removes **57.5%** of the Q2_K-to-FP16 gap.
DBF wins 526 of 580 paired chunks; Q2_K wins 54, with no ties. The exact two-sided sign test on per-chunk NLL
gives `p = 3.21e-98`.

Matched Q3_K and Q4_K controls reach `7.2279` and `7.0557` PPL respectively. DBF therefore recovers **83.1%**
of the Q2_K-to-Q3_K improvement on the NLL scale. Interpolating locally between Q2_K (`2.625 bpe`) and
Q3_K (`3.4375 bpe`) places
DBF at approximately **3.30-bpe scalar-equivalent quality**, while its actual routed-expert payload is
**2.555 bpe**. This is an empirical interpolation for this model and dataset, not a universal quantization scale.

The matrix-level result is equally uniform across all 30,720 matched pairs:

| Matrix approximation metric | Q2_K | DBF | Mean relative improvement | DBF wins |
|---|---:|---:|---:|---:|
| Mean relative Frobenius residual | 0.29938 | **0.24184** | **19.10%** | **30,720 / 30,720** |
| Mean relative spectral residual | 0.20146 | **0.15589** | **22.52%** | **30,720 / 30,720** |

We include the spectral metric for a deliberately speculative reason. Let `E = W - W_approx`. Its spectral norm
is exactly the largest output error `||E x||_2` over unit-norm inputs, so it measures the worst-case operator error.
The Frobenius norm is not merely a naive coefficient metric: it aggregates all coefficient errors and, for an
isotropic random input, determines the expected squared output error. Real model activations are not isotropic or
adversarial, so neither metric is automatically the right proxy; an activation-covariance-weighted error may be
more relevant than either. Spectral error may matter especially when one narrow, sensitive direction dominates,
but whether it predicts PPL better than Frobenius error remains an empirical question. Neither DBF nor our current
reproduction optimizes it. More importantly, both unweighted metrics ignore the activation distribution.

Each matrix is normalized by its own norm before aggregation, so large matrices or high-energy experts cannot
silently dominate the average. We report both the mean improvement and the number of matrices on which DBF wins.
Spectral norms use fixed-seed 32-iteration power iteration. The full model-level PPL then tests whether those
approximation gains survive inference.

---

## 9. Open questions and the next episodes

### 9.1 DBF is reported to struggle at larger bit budgets. Can that regime be repaired?

The original method is especially strong at very low budgets. As the intermediate dimension grows, the
alternating problem becomes increasingly overcomplete and difficult to optimize. Is that a fundamental
limitation of the representation, or a limitation of the continuation/projection algorithm?

### 9.2 Can one structural method cover all important model matrices?

Routed FFN experts are only part of an LLM. Attention projections, shared experts, routers, embeddings, and
the language-model head have different shapes and sensitivities. Can we move from “binary experts” to a model
that is genuinely binary in the meaningful weight-storage sense?

### 9.3 Can DBF acquire discrete activations and become “BitNet” in the literal sense?

Classical binary-network work, including BitNet-style systems, combines discrete weights with discrete or highly
constrained activations. Our current implementation has a binary weight core but still enters and leaves through
floating-point values and floating diagonal scales.

Can shared-exponent bitplanes become the natural activation contract rather than a backend-private optimization?
Can the complete linear core operate on packed bits and integer accumulation while preserving model quality?

### 9.4 Which matrix error should we optimize?

The practical answer may be: neither ordinary Frobenius nor ordinary spectral norm. Inference does not feed the
matrix isotropic random vectors or worst-case adversarial vectors; it feeds activations drawn from the model's
actual workload. An importance matrix provides empirical input-channel weights from calibration data, letting the
decomposition optimize errors that the model is likely to exercise rather than treating every direction equally.

DBF already supports row and column weighting naturally. Integrating importance-matrix norms into our
factorization pipeline is therefore mainly a calibrated rerun, not a new representation. We excluded it from
Episode One because we wanted to cut it.

A later episode will compare four views of the same approximation error:

1. ordinary Frobenius norm;
2. ordinary spectral norm;
3. importance-weighted Frobenius norm;
4. importance-weighted spectral norm.

The final judge remains model-level PPL and KLD. The useful question is not which norm has the most attractive
interpretation, but which objective produces the best deployed model at the same storage and runtime budget.

### 9.5 How much evaluation is enough?

This post uses one full WikiText-2 perplexity run as the first model-level quality gate. That is enough for a
pilot comparison, not a complete capability evaluation. A later performance post should widen the test in
three directions:

1. **More perplexity datasets.** Repeat the paired evaluation on corpora with different styles and distributions,
   including C4 and Penn Treebank rather than trusting one WikiText-2 result.
2. **Logit divergence.** Measure token-level KLD against the original FP16-expert model, with matched prompts and
   causal positions. PPL sees only the probability assigned to the observed next token; KLD sees how the complete
   predicted distribution moved.
3. **Functional benchmarks.** Run a compact standard suite through the Language Model Evaluation Harness
   [[17]](#ref-lm-eval), using the exact same prompts, tokenizer, harness revision, and scoring path for every
   model. Representative tasks include LAMBADA, HellaSwag, ARC Challenge, Winogrande, PIQA, BoolQ, MMLU, and
   GSM8K; the final subset should cover several capabilities without turning one quantization comparison into a
   leaderboard.

The purpose is not to manufacture one aggregate score. It is to find whether the representation changes only
average language modeling loss or damages a particular capability that PPL hides.

### 9.6 Does the result transfer to more interesting models?

Qwen3.6-35B-A3B is a practical controlled battlefield, not the final target. The next useful test is whether the
same representation and kernel ideas survive larger expert dimensions, different routing distributions, and newer
frontier MoE architectures.

The most practical next port is **GLM-4.5-Air**: 106B total / 12B active parameters, 46 layers, 128 routed experts,
a 4096-wide hidden state, and 1408-wide expert intermediates [[19]](#ref-glm45air). It fits comfortably on one
128 GB Strix Halo after ordinary low-bit quantization, while its expert geometry and routing differ enough from
Qwen to make the transfer meaningful.

Two larger targets are **MiniMax-M2.5** and **DeepSeek-V4-Flash**. MiniMax-M2.5 is a 230B / 10B-active MoE with
62 layers, 256 experts, 3072-wide hidden states, and 1536-wide expert intermediates [[20]](#ref-minimax-m25).
DeepSeek-V4-Flash is a 284B / 13B-active MoE with 43 layers, 256 routed experts, 4096-wide hidden states,
2048-wide MoE intermediates, and one-million-token context [[18]](#ref-deepseek-v4-flash). Both are close enough
to the one-Halo memory boundary that a Q2-scale structural representation is operationally valuable rather than
merely a matrix experiment.

GLM-5.2 is a less practical immediate target. Its 78-layer, 6144×2048, 256-expert geometry implies hundreds of
billions of expert parameters; even a Q2-scale expert payload alone exceeds one Halo's memory. It belongs to a
multi-node follow-up, not the first transfer test.

These are not promised ports. They are falsification targets. A representation that wins only on one convenient
Qwen geometry is a useful kernel experiment; one that transfers across Qwen, GLM, MiniMax, and DeepSeek begins to
look like a general post-training binary format.

---

## 10. Reproducibility

The public release includes:

- the minimal rebased llama.cpp branch;
- build instructions for CPU, CUDA/HIP, and Vulkan;
- the [model-conversion and analysis pipeline](https://github.com/eodus/dbf-sign1);
- exact tensor and bit accounting;
- backend correctness commands;
- paired full-perplexity scripts;
- benchmark commands and hardware details.

The [release model](https://huggingface.co/eodus/Qwen3.6-35B-A3B-DBF-SIGN1) is available on Hugging Face.
The comparison keeps every non-routed-expert tensor identical, replaces the same routed expert matrices in both
models, and accounts for their exact raw tensor payload. Quality is measured over all 580 WikiText-2 chunks using
paired per-chunk NLL and an exact sign test, not only the aggregate perplexity. The release record identifies the
model and code revisions used for the table and provides the exact commands. The GGUF SHA-256 is
`d4fa165f0bf4752ebd86c8aa5285401b6318e48787d0912d696f05f80b8071ac`; the validated llama.cpp release
commit is `753bd8614d43a03b00583c0e51a9b4fca75f9ab4`. These artifacts establish the authors' fixed-dimension
DBF core, our middle-diagonal reprojection, and our runtime together.

**AI collaboration disclosure.** This work was developed by Sasha Shlemov with Drinkins, his calibrated personal
AI research and engineering assistant. Drinkins contributed implementation, experiment design, debugging, analysis,
source research, and editing across several model engines connected through shared memory and persistent artifacts.
Sasha directed the research, made the final technical and publication decisions, and takes responsibility for the
claims. AI assistance is not incidental here: doing this amount of AI implementation, experimentation, and source
analysis without AI would be a strange methodological choice.

DBF is not a new miracle, but it is a working machine built from an idea worth taking seriously.

---

## References

1. <a id="ref-bitnet"></a>H. Wang et al.,
   “[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453),”
   arXiv:2310.11453, 2023.
2. <a id="ref-bitnet158"></a>S. Ma et al.,
   “[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764),”
   arXiv:2402.17764, 2024.
3. <a id="ref-bitnetcpp"></a>J. Wang et al.,
   “[Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880),”
   ACL 2025; arXiv:2502.11880.
4. <a id="ref-dbf"></a>V. Boža and V. Macko,
   “[Addition Is Almost All You Need: Compressing Large Language Models with Double Binary Factorization](https://openreview.net/forum?id=k5kUKoewdQ),”
   *Transactions on Machine Learning Research*, accepted camera-ready revision, 2026.
5. <a id="ref-qwen36"></a>Qwen Team,
   “[Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),”
   model card and open weights, 2026.
6. <a id="ref-sign1-llama"></a>Sasha Shlemov,
   “[SIGN1/DBDBD support for llama.cpp](https://github.com/eodus/sign1-llama.cpp),”
   public research implementation, 2026.
7. <a id="ref-ds4"></a>S. Sanfilippo (antirez),
   “[DS4 on Strix Halo](https://github.com/antirez/ds4/blob/main/STRIXHALO.md),”
   setup and inference implementation for DeepSeek V4 on Radeon 8060S, 2026.
8. <a id="ref-halo"></a>Advanced Micro Devices,
   “[AMD Ryzen AI Halo](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo.html),”
   platform specifications and measured local-LLM workloads, 2026.
9. <a id="ref-llamacpp"></a>ggml-org and contributors,
   “[llama.cpp](https://github.com/ggml-org/llama.cpp),”
   local LLM inference framework, 2026.
10. <a id="ref-qwen35-35b"></a>Qwen Team,
    “[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B),”
    model card and open weights, 2026.
11. <a id="ref-qwen35-122b"></a>Qwen Team,
    “[Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B),”
    model card and open weights, 2026.
12. <a id="ref-qwen35-397b"></a>Qwen Team,
    “[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B),”
    model card and open weights, 2026.
13. <a id="ref-dp4a"></a>NVIDIA,
    “[CUDA Math API: Integer Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),”
    `__dp4a` intrinsic documentation, 2026.
14. <a id="ref-glsl-mix"></a>Khronos Group,
    “[GLSL 4 Reference: mix](https://registry.khronos.org/OpenGL-Refpages/gl4/html/mix.xhtml),”
    Section 8.3, Boolean-selector overloads of `mix`, 2023.
15. <a id="ref-vulkan-sync"></a>Khronos Group,
    “[Vulkan Guide: Synchronization Examples](https://github.khronos.org/Vulkan-Site/guide/latest/synchronization_examples.html),”
    compute-dispatch and pipeline-barrier synchronization, 2026.
16. <a id="ref-cuda-guide"></a>NVIDIA,
    “[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/),”
    cooperative groups, grid synchronization, and CUDA Dynamic Parallelism, 2026.
17. <a id="ref-lm-eval"></a>L. Gao et al.,
    “[The Language Model Evaluation Harness](https://zenodo.org/records/12608602),”
    EleutherAI, version 0.4.3, Zenodo, 2024, doi:10.5281/zenodo.12608602.
18. <a id="ref-deepseek-v4-flash"></a>DeepSeek-AI,
    “[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash),”
    official model card, configuration, and open weights, 2026.
19. <a id="ref-glm45air"></a>GLM-4.5 Team / Z.ai,
    “[GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air),”
    official model card, configuration, and open weights, 2025.
20. <a id="ref-minimax-m25"></a>MiniMax-AI,
    “[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5),”
    official model card, configuration, and open weights, 2026.

BibTeX metadata is available in [`episode-01-dbf.bib`](episode-01-dbf.bib).
