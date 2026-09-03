# ntc — toy neural texture compression trained with Evolution Strategies

A small, self-contained C++ experiment: an RGB image is encoded as a
low-resolution latent texture plus a tiny MLP decoder, and both are trained
**entirely with Evolution Strategies** — no backprop, no derivatives, no
training framework. Dependencies are `stb_image`, `stb_image_write`, and OpenMP.

Write-up: [Fitting a neural texture decoder with ES](https://richg42.blogspot.com/2026/09/fitting-neural-texture-decoder-with-es.html)

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
  only to the 4 texels its bilinear tap reads, so a single decode pair yields a
  local gradient estimate for every texel simultaneously. This locality trick is
  what makes ES practical on tens of thousands of latent parameters.
* The latent stays fp32 during training; quantization is applied afterwards
  and reported at a configurable bit depth.

## Building

CMake generating a Visual Studio solution (MSVC), or any C++17 compiler with
OpenMP on Linux/WSL:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Running

```
ntc image.png --out out --latent 128 128 8 --iters 3000
```

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

## Status

This is a deliberately simple research toy for learning and experimentation,
not a codec. Nothing is tuned. Obvious next steps: quantization-aware
training, block-compressed (BC/ASTC) latents inside the training loop,
multiresolution latents, alternative losses.
