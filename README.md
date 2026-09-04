# toy neural texture compression trained with Evolution Strategies

A small, self-contained C++ experiment: an RGB image is encoded as a
low-resolution latent texture plus a tiny MLP decoder, and both are trained
**entirely with Evolution Strategies** — no backprop, no derivatives, no
training framework. Dependencies are `stb_image`, `stb_image_write`, and OpenMP.

Write-up: [Fitting a neural texture decoder with ES](https://richg42.blogspot.com/2026/09/fitting-neural-texture-decoder-with-es.html)

Here's the newer version supporting [materials](https://github.com/richgel999/neural_texture_es2).

```
I(u,v) ≈ MLP( bilinear(Z, u, v), phi(u,v) )
```

`Z` is the latent texture, `phi` a small positional encoding. At decode time
each pixel bilinearly samples `Z` at its UV, appends `phi`, and runs the MLP.

## Results

512×512 crop of kodim23, 3000 iterations, latent quantized to 8 bits after
training, MLP weights counted as fp16:

| Latent      | PSNR    | bpp (raw) | bpp (entropy coded) |
|-------------|---------|-----------|---------------------|
| 64×64×4     | 26.9 dB | 0.56      | 0.47                |
| 64×64×8     | 28.2 dB | 1.07      | 0.87                |
| 128×128×4   | 30.3 dB | 2.06      | 1.65                |
| 128×128×8   | 32.2 dB | 4.07      | 3.27                |

The 128×128×8 run uses a 14 → 24 → 24 → 3 MLP (1035 weights, leaky ReLU,
sigmoid output) and trains in about 150 s on a 32-thread CPU. Quantizing the
latent to 8 bits costs 0.04 dB.

Output of that run (`out_128c8/`): target crop, reconstruction after 3000
iterations, and the eight latent channels side by side.

| Target | Reconstruction (32.2 dB) |
|--------|--------------------------|
| ![target](out_128c8/target.png) | ![recon](out_128c8/recon_003000.png) |

![latent](out_128c8/latent_003000.png)

`out_128c8/model.bin` is the trained model; evaluate it with
`ntc kodim23.png --load out_128c8/model.bin --latent 128 128 8 --nfreq 1 --iters 0`.

## How the ES training works

* **MLP:** antithetic ES, 32 perturbation pairs per step, each pair evaluated
  on the same random 4096-pixel minibatch. The estimated gradient is fed to Adam.
* **Latent:** all latent values are perturbed at once and the full image is
  decoded twice per pair, 4 pairs per step. Each pixel's loss change is credited
  only to the 4 texels its bilinear tap reads.

  Ordinary ES already updates every parameter from one antithetic pair, but its
  variance grows with parameter count: the single scalar loss difference is the
  sum of thousands of separate local effects, and each texel's share is
  buried under everyone else's. Here each texel's loss difference is measured
  only over the pixels in its own bilinear footprint, so noise from the other
  ~16k texels is discarded instead of averaged. That credit assignment, not the
  per-evaluation cost, is what makes ES practical on a latent with 131k values
  using only 4 pairs per step.
* The latent stays fp32 during training; quantization is applied afterwards
  and reported at a configurable bit depth.

## Building

CMake generating a Visual Studio solution (MSVC), or any C++17 compiler with
OpenMP on Linux/WSL. Tested with MSVC 2022 and MSVC 2026 on Windows, and
gcc 13 under WSL2. Note that `std::normal_distribution` differs between
standard libraries, so the same seed gives slightly different results on
each platform.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

For Visual Studio 2026 use `-G "Visual Studio 18 2026"`, which needs
CMake 4.2 or newer (the CMake bundled with VS 2026 works).

## Running

```
ntc image.png --out out --latent 128 128 8 --iters 3000
```

Run with no arguments, `ntc` trains on the checked-in `kodim23.png` using the
default 64×64×4 latent. The image is located relative to the executable, so
this works from the build directory as well as the repo root.

Progress is printed to stdout (MSE, PSNR, quantized PSNR and bitrate, latent
stats, throughput). Reconstructions, a latent visualization, and `model.bin`
are written to the output directory periodically.

Useful options (`ntc --help` lists them all):

| Flag | Meaning |
|------|---------|
| `--latent W H C` | latent texture size (default 64 64 4) |
| `--mlp W1,W2,...` | hidden layer widths (default 24,24) |
| `--act leaky\|relu\|tanh\|sine` | hidden activation |
| `--pos SPEC` | positional features: `uv`, `fourier:N`, `dct:N`, `local`, `lfourier:N`, `lquad`, `ldct:N`, `none` |
| `--qbits N` | latent bit depth for the reported quantized PSNR / bitrate |
| `--load model.bin --iters 0` | evaluate a saved model |
| `--mlp-pairs`, `--mlp-sigma`, `--mlp-lr`, `--lat-pairs`, `--lat-sigma`, `--lat-lr` | ES hyperparameters |

Images larger than 512×512 are center-cropped by default (`--crop`).
The Kodak test images are widely available; kodim23 is the parrots.

## Prior art disclosure

Published September 3, 2026 (blog post above and this repository). The
following are disclosed here as public prior art.

Neural texture representations using learned latent grids with small neural
decoders, and Evolution Strategies / simultaneous-perturbation methods for
derivative-free optimization, are established ideas. The technically
distinctive part explored here is their combination with the decoder's known
spatial dependency structure: all latent values are perturbed simultaneously,
antithetic full-image evaluations produce per-pixel loss differences, and each
pixel's loss difference is attributed only to the latent texels actually read
by that pixel's filtering footprint. This yields simultaneous,
support-restricted ES estimates for every latent texel while discarding loss
variation from pixels a given texel cannot affect. Estimates for neighboring
texels still share pixels and the same perturbation draw, so they are
correlated rather than independent.

For a given latent value, the omitted per-pixel loss terms do not depend on
that value's perturbation, so their products with it have zero expectation in
the ordinary Gaussian ES estimator. Footprint attribution therefore removes
them without bias, as a variance-reduction mechanism that follows directly
from the decoder's dependency graph.

Implemented in this repository:

* A low-resolution latent texture plus a small MLP decoder, with **both the
  latent and the decoder optimized entirely by antithetic Evolution
  Strategies**, without backpropagation or derivatives of any kind.
* **Support-restricted footprint attribution for latent ES:** all latent values
  are perturbed simultaneously and the full image is decoded for +ε and −ε.
  Each pixel's loss difference is attributed only to the latent texels in that
  pixel's bilinear sampling footprint, so one antithetic decode pair produces
  simultaneous local ES estimates across the entire latent while excluding loss
  terms that cannot depend on each texel.
* Separate ES schedules matched to parameter support: minibatched, many-pair ES
  for the globally acting decoder weights, and full-image, few-pair
  footprint-attributed ES for the spatially local latent, interleaved every
  iteration, with the estimates fed through Adam.
* Post-training scalar quantization of the latent with per-channel scale, and
  reported bitrate at arbitrary latent bit depth.
* Pluggable positional encodings for the decoder, including cell-periodic
  cosine features of the bilinear cell offset (`ldct:N`), found to improve
  quality at fine latent resolution.
* Configurable decoder depth, width, and activation; saved models record the
  full configuration.

Described, not yet implemented:

* **Quantization-aware training under ES:** quantize (or block-compress) the
  latent inside the decode used for every ES evaluation. Because ES only
  observes loss values, any non-differentiable quantizer or codec can sit in
  the loop with no straight-through estimator or differentiable surrogate.
* **Block-compressed latents (BC1/BC2/BC3/BC4/BC5/BC6H/BC7/ASTC LDR/ASTC HDR) in the training loop**, with
  loss attribution per compressed block rather than per texel.
* **Search directly in the encoded or GPU compressed texture domain:** drop the float latent and run ES
  (or stochastic coordinate descent) over block endpoints and indices (weights)
  themselves, **so the trainer and the GPU texture compressor are the same program**.
* **Non-overlapping perturbation phases:** perturb only texels or blocks on
  one phase of a 2×2 or 3×3 grid per evaluation so footprints never overlap,
  giving exact per-parameter loss differences in one decode; cycle the phase
  to cover all parameters.
* Multiresolution latent pyramids trained with the same footprint attribution,
  and learned interpolation kernels expressed as a few global parameters
  rather than as decoder inputs.

## Status

This is a deliberately simple research toy for learning and experimentation,
not a codec. Nothing is tuned. Obvious next steps: quantization-aware
training, block-compressed (BC/ASTC) latents inside the training loop,
multiresolution latents, alternative losses.
