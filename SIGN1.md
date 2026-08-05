# SIGN1 and DBDBD

This branch is an experimental, buildable llama.cpp implementation of packed one-bit
matrices and Double Binary Factorization (DBF) for routed Qwen MoE expert weights.
It is a research release, not an upstream pull request.

## Credit

The representation and factorization method come from:

> Vladimír Boža and Vladimír Macko,
> “Addition Is Almost All You Need: Compressing Large Language Models with Double
> Binary Factorization,” TMLR, 2026. arXiv:2505.11076.

Please cite the original paper when using DBF. This branch contributes the inference-engine
implementation: storage, serialization, model loading, graph construction, backend kernels,
and integration tests.

## Representation

A dense projection is represented as

```text
W ≈ D_out · B_out · D_mid · B_in · D_in
```

where both `B` factors contain only `+1/-1`, and the three `D` factors are diagonal.
The runtime shorthand is **DBDBD**.

`GGML_TYPE_SIGN1` stores 64 signs in one `uint64_t`:

```text
bit 0 -> +1
bit 1 -> -1
```

The current model integration targets routed `down`, `gate`, and `up` projections in
Qwen3.5/Qwen3.6 MoE models. Binary factors are stored as SIGN1; diagonal factors are
stored as F16.

## Implemented backends

- **CPU:** scalar fallback, AVX2, and AVX-512 paths.
- **CUDA/HIP:** routed SIGN1 matmul and scaler support. The current HIP full-model path
  retains a bounded synchronization fallback for correctness.
- **Vulkan:** optimized routed DBDBD execution, including shared-exponent Q8 activation
  bitplanes for prompt processing and shape-specific generation kernels.

Vulkan on AMD Strix Halo is the primary optimized and validated backend for this release.
The other backends establish portability and correctness; they are not claimed to have
identical performance maturity.

## Build

### CPU

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-backend-ops llama-cli llama-perplexity llama-bench -j
```

### Vulkan

```bash
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target test-backend-ops llama-cli llama-perplexity llama-bench -j
```

### CUDA or HIP

Use llama.cpp's ordinary CUDA build procedure for the target toolchain:

```bash
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target test-backend-ops llama-cli llama-perplexity llama-bench -j
```

On AMD, configure the same source through the supported HIP/ROCm toolchain and set the
appropriate AMD GPU target.

## Test

List devices:

```bash
./build-vulkan/bin/llama-cli --list-devices
```

Run routed-matmul backend tests:

```bash
./build-vulkan/bin/test-backend-ops -o MUL_MAT_ID
```

Run scaler-selection tests:

```bash
./build-vulkan/bin/test-backend-ops -o MUL_ROWS_ID
```

A full model must additionally pass both execution regimes:

```bash
# Generation path: force one-token batches.
./build-vulkan/bin/llama-perplexity \
  -m model-dbdbd.gguf -ngl 999 -f wiki.test.raw -c 512 -b 1 -ub 1

# Prompt-processing path.
./build-vulkan/bin/llama-perplexity \
  -m model-dbdbd.gguf -ngl 999 -f wiki.test.raw -c 512

./build-vulkan/bin/llama-bench \
  -m model-dbdbd.gguf -ngl 999 -p 512 -n 128 -r 10
```

A coherent chat response is not a correctness test. Generation and batched execution use
different kernels and must be validated separately.

## Model artifacts

The release GGUF, model card, and checksum are published at
[`eodus/Qwen3.6-35B-A3B-DBF-SIGN1`](https://huggingface.co/eodus/Qwen3.6-35B-A3B-DBF-SIGN1).
The reproducible conversion, controls, manifests, and measurements are published at
[`eodus/dbf-sign1`](https://github.com/eodus/dbf-sign1). The GGUF SHA-256 is
`d4fa165f0bf4752ebd86c8aa5285401b6318e48787d0912d696f05f80b8071ac`.

The expected routed projection tensor family is:

```text
<projection>_din
<projection>_v
<projection>_dmid
<projection>_u
<projection>_dout
```

`v` and `u` are SIGN1 tensors. The diagonal tensors are F16.

## Current limitations

1. The optimized model integration is specific to routed Qwen MoE expert projections.
2. Attention, embeddings, routers, shared experts, and the LM head remain in their existing
   dense/scalar-quantized forms in the Episode 1 model.
3. HIP uses a correctness synchronization fallback whose nonblocking replacement remains
   unresolved.
4. Vulkan optimization is measured on AMD Strix Halo; NVIDIA CUDA and other GPUs need
   independent validation.
5. The faithful DBF baseline is expensive to produce: Qwen3.6-35B-A3B contains 30,720
   routed expert matrices for the three projections across 40 layers.
6. DBF is known to become difficult to optimize at larger internal ranks/bit budgets;
   rank continuation and revised projectors remain future work.

## Scope of this branch

The intended story is deliberately narrow:

- the code compiles;
- SIGN1 is a real serialized type;
- DBDBD tensors load and execute;
- CPU, CUDA/HIP, and Vulkan contain working paths;
- routed backend tests exist;
- generation and prompt-processing perplexity paths are exercised;
- known correctness fallbacks and performance limits are stated.

The branch does not claim that the current implementation is ready for upstream review or
that every backend is equally optimized. The frozen Episode One quality and speed results are
reported in the model card and accompanying post.

The accompanying implementation story is
[LLM Binarization — Episode One: DBF](docs/blog/llm-binarization-episode-01-dbf.md).
