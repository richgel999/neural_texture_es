// ntc - toy neural texture compressor trained with Evolution Strategies.
//
//   I_hat(u,v) = MLP( bilinear(Z, u, v), phi(u,v) )
//
// Z   : low-res latent texture (LW x LH x LC floats)
// MLP : tiny fully connected net, < ~2000 weights
// phi : (u,v) in [-1,1] plus a few Fourier features
//
// Both Z and the MLP are trained with antithetic ES (no backprop).
//   MLP    : global ES on a random pixel minibatch shared across all pairs.
//   Latent : all texels perturbed at once; each texel's loss change is measured
//            only over the pixels its bilinear footprint touches, so one
//            full-image decode pair yields a gradient estimate for every texel.

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static const float PI = 3.14159265358979f;

// ---------------------------------------------------------------- options
struct Options {
    std::string input = "kodim23.png";  // default test image (checked in)
    std::string outdir = "out";
    int crop = 512;            // center-crop input to crop x crop (0 = none)
    int LW = 64, LH = 64, LC = 4;
    std::vector<int> hidden = { 24, 24 };  // MLP hidden layer widths
    std::string act = "leaky";             // leaky | relu | tanh | sine
    std::string pos = "uv";    // positional encoding spec, see PosEnc
    bool clamp_out = false;    // hard clamp instead of sigmoid
    int iters = 3000;
    int save_every = 100;
    int print_every = 10;
    unsigned seed = 1;
    // MLP ES
    int mlp_pairs = 32;
    int mlp_batch = 4096;
    float mlp_sigma = 0.02f;
    float mlp_lr = 0.005f;
    int mlp_every = 1;
    // latent ES
    int lat_pairs = 4;
    float lat_sigma = 0.05f;
    float lat_lr = 0.02f;
    float lat_init = 0.1f;
    int threads = 0;
    int qbits = 8;             // latent bit depth used for the reported quantized stats
    std::string load;          // load model.bin instead of random init
};

static void usage() {
    printf(
        "ntc [options] [input.png]   (default input: kodim23.png)\n"
        "  --out DIR            output directory (default out)\n"
        "  --crop N             center-crop to NxN, 0 = none (512)\n"
        "  --latent W H C       latent texture size (64 64 4)\n"
        "  --mlp W1,W2,...      MLP hidden layer widths (24,24); e.g. --mlp 32 or --mlp 16,16,16\n"
        "  --hidden N           shorthand for --mlp N,N\n"
        "  --act NAME           hidden activation: leaky | relu | tanh | sine (leaky)\n"
        "  --pos SPEC           positional features, comma list (uv). Kinds:\n"
        "                         uv | fourier:N | dct:N | local | lfourier:N | lquad | ldct:N | none\n"
        "                         e.g. --pos uv,local  or  --pos local,lfourier:2\n"
        "  --nfreq N            shorthand: --pos uv,fourier:N (N=0 -> uv only)\n"
        "  --clamp              hard-clamp output instead of sigmoid\n"
        "  --iters N            training iterations (3000)\n"
        "  --save-every N       write PNGs every N iters (100)\n"
        "  --print-every N      print stats every N iters (10)\n"
        "  --seed N\n"
        "  --mlp-pairs N --mlp-batch N --mlp-sigma F --mlp-lr F --mlp-every N\n"
        "  --lat-pairs N --lat-sigma F --lat-lr F --lat-init F\n"
        "  --threads N\n"
        "  --qbits N            latent bit depth for reported quantized bitrate/psnr (8)\n"
        "  --load model.bin     start from a saved model (use --iters 0 to just evaluate)\n");
}

// ---------------------------------------------------------------- image
struct Image {
    int w = 0, h = 0;
    std::vector<float> rgb; // w*h*3, [0,1]
    float& at(int x, int y, int c) { return rgb[(size_t)(y * w + x) * 3 + c]; }
    float at(int x, int y, int c) const { return rgb[(size_t)(y * w + x) * 3 + c]; }
};

static bool load_png(const std::string& path, Image& img, int crop) {
    int w, h, n;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!data) return false;
    int cw = w, ch = h, ox = 0, oy = 0;
    if (crop > 0 && (w > crop || h > crop)) {
        cw = std::min(w, crop); ch = std::min(h, crop);
        ox = (w - cw) / 2; oy = (h - ch) / 2;
    }
    img.w = cw; img.h = ch; img.rgb.resize((size_t)cw * ch * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++)
            for (int c = 0; c < 3; c++)
                img.at(x, y, c) = data[((size_t)(y + oy) * w + (x + ox)) * 3 + c] / 255.0f;
    stbi_image_free(data);
    return true;
}

static void save_png(const std::string& path, const Image& img) {
    std::vector<unsigned char> buf(img.rgb.size());
    for (size_t i = 0; i < buf.size(); i++)
        buf[i] = (unsigned char)std::lround(std::min(1.0f, std::max(0.0f, img.rgb[i])) * 255.0f);
    stbi_write_png(path.c_str(), img.w, img.h, 3, buf.data(), img.w * 3);
}

// Synthetic fallback so the program runs without an input image.
static void make_synthetic(Image& img, int n) {
    img.w = img.h = n; img.rgb.resize((size_t)n * n * 3);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float u = x / (float)n, v = y / (float)n;
            float r = 0.5f + 0.5f * std::sin(u * 12.0f + 3.0f * std::sin(v * 7.0f));
            float g = v;
            float b = ((x / 32 + y / 32) & 1) ? 0.8f : 0.2f;
            float dx = u - 0.5f, dy = v - 0.5f;
            if (dx * dx + dy * dy < 0.05f) { r = 1.0f; g = 0.9f; b = 0.1f; }
            img.at(x, y, 0) = r; img.at(x, y, 1) = g; img.at(x, y, 2) = b;
        }
}

// ---------------------------------------------------------------- latent
struct Latent {
    int W = 0, H = 0, C = 0;
    std::vector<float> z; // W*H*C
    const float* texel(int x, int y) const { return &z[((size_t)y * W + x) * C]; }
    size_t size() const { return z.size(); }
};

// Map a pixel (px,py) of a W x H image to latent-space texel coordinates.
// UV = pixel center in [0,1]; texel centers sit at (i+0.5)/LW.
struct BilinearTap {
    int x0, x1, y0, y1;
    float fx, fy;
};

static inline BilinearTap bilinear_tap(const Latent& L, float u, float v) {
    float x = u * L.W - 0.5f, y = v * L.H - 0.5f;
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    BilinearTap t;
    t.fx = x - x0; t.fy = y - y0;
    t.x0 = std::max(0, std::min(L.W - 1, x0));
    t.x1 = std::max(0, std::min(L.W - 1, x0 + 1));
    t.y0 = std::max(0, std::min(L.H - 1, y0));
    t.y1 = std::max(0, std::min(L.H - 1, y0 + 1));
    return t;
}

static inline void sample_latent(const Latent& L, const float* z, const BilinearTap& t, float* out) {
    const float* a = &z[((size_t)t.y0 * L.W + t.x0) * L.C];
    const float* b = &z[((size_t)t.y0 * L.W + t.x1) * L.C];
    const float* c = &z[((size_t)t.y1 * L.W + t.x0) * L.C];
    const float* d = &z[((size_t)t.y1 * L.W + t.x1) * L.C];
    float w00 = (1 - t.fx) * (1 - t.fy), w10 = t.fx * (1 - t.fy);
    float w01 = (1 - t.fx) * t.fy, w11 = t.fx * t.fy;
    for (int k = 0; k < L.C; k++)
        out[k] = w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
}

// ---------------------------------------------------------------- MLP
// Fully connected net with an arbitrary list of hidden widths, e.g. {24,24}.
// Flat parameter layout, layer by layer: W[out*in] (row-major, one row per
// output unit) followed by b[out].
static const int MAXH = 128;      // max units in any layer (incl. the input)
static const int MAXL = 8;        // max hidden layers

enum Act { ACT_LEAKY, ACT_RELU, ACT_TANH, ACT_SINE };

static inline float activate(Act a, float x) {
    switch (a) {
    case ACT_RELU:  return x > 0 ? x : 0.0f;
    case ACT_TANH:  return std::tanh(x);
    case ACT_SINE:  return std::sin(x);
    default:        return x > 0 ? x : 0.01f * x;   // leaky ReLU
    }
}

static bool parse_act(const std::string& s, Act& a) {
    if (s == "leaky") a = ACT_LEAKY;
    else if (s == "relu") a = ACT_RELU;
    else if (s == "tanh") a = ACT_TANH;
    else if (s == "sine") a = ACT_SINE;
    else return false;
    return true;
}
static const char* act_name(Act a) {
    static const char* n[] = { "leaky", "relu", "tanh", "sine" };
    return n[a];
}

struct MLP {
    int nin = 0, nout = 3;
    std::vector<int> hidden;       // hidden layer widths
    Act act = ACT_LEAKY;
    std::vector<float> p;
    size_t size() const { return p.size(); }

    // Widths of every layer, input first, output last.
    std::vector<int> widths() const {
        std::vector<int> w; w.push_back(nin);
        for (int h : hidden) w.push_back(h);
        w.push_back(nout);
        return w;
    }
    static size_t count(const std::vector<int>& w) {
        size_t n = 0;
        for (size_t l = 1; l < w.size(); l++) n += (size_t)w[l] * w[l - 1] + w[l];
        return n;
    }
    std::string describe() const {
        std::string s = std::to_string(nin);
        for (int h : hidden) s += " -> " + std::to_string(h);
        return s + " -> " + std::to_string(nout);
    }
    void init(int nin_, const std::vector<int>& hidden_, Act act_, std::mt19937& rng) {
        nin = nin_; hidden = hidden_; act = act_;
        std::vector<int> w = widths();
        p.assign(count(w), 0.0f);
        std::normal_distribution<float> N(0.0f, 1.0f);
        size_t o = 0;
        for (size_t l = 1; l < w.size(); l++) {
            bool last = (l + 1 == w.size());
            // He init for hidden layers, smaller for the output so it starts near mid-gray.
            float s = last ? std::sqrt(1.0f / w[l - 1]) : std::sqrt(2.0f / w[l - 1]);
            if (act == ACT_SINE && !last) s = (l == 1) ? 1.0f / w[l - 1] * 30.0f : std::sqrt(6.0f / w[l - 1]);
            for (int i = 0; i < w[l] * w[l - 1]; i++) p[o++] = N(rng) * s;
            o += w[l]; // biases stay zero
        }
    }
};

// Forward pass with an explicit parameter pointer so perturbed copies can be used.
static inline void mlp_forward(const MLP& m, const float* p, const float* in, float* out, bool clamp_out) {
    float bufA[MAXH], bufB[MAXH];
    const float* cur = in;
    int ncur = m.nin;
    float* nxt = bufA;
    for (size_t l = 0; l < m.hidden.size(); l++) {
        int nh = m.hidden[l];
        const float* W = p; const float* b = W + (size_t)nh * ncur;
        for (int j = 0; j < nh; j++) {
            float s = b[j]; const float* w = W + (size_t)j * ncur;
            for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
            nxt[j] = activate(m.act, s);
        }
        p = b + nh;
        cur = nxt; ncur = nh;
        nxt = (nxt == bufA) ? bufB : bufA;
    }
    const float* W = p; const float* b = W + (size_t)m.nout * ncur;
    for (int j = 0; j < m.nout; j++) {
        float s = b[j]; const float* w = W + (size_t)j * ncur;
        for (int i = 0; i < ncur; i++) s += w[i] * cur[i];
        if (clamp_out) out[j] = std::min(1.0f, std::max(0.0f, s + 0.5f));
        else out[j] = 1.0f / (1.0f + std::exp(-s));
    }
}

// ---------------------------------------------------------------- positional encoding
// phi(u,v): a list of named feature generators, composed from --pos "a,b:N,...".
// To add an experiment: add an enum value, a name in POS_NAMES, its output
// count in PosFeature::count(), and its evaluation in PosEnc::encode().
//
//   uv           global u,v mapped to [-1,1]                               (2)
//   fourier:N    sin/cos(2^k * 2*pi*u), same for v, k = 0..N-1              (4N)
//   local        fractional offset inside the bilinear cell, fx,fy -> [-1,1] (2)
//   lfourier:N   sin/cos(2^k * 2*pi*fx), same for fy: periodic per texel    (4N)
//   lquad        fx*fy, fx^2, fy^2 of the cell offset: cheap 2nd-order kernel (3)
//   dct:N        cos(pi*k*u), cos(pi*k*v), k = 1..N: the DCT (k,0),(0,k) bases   (2N)
//   ldct:N       same on the cell offset fx,fy                                    (2N)
enum PosKind { POS_UV, POS_FOURIER, POS_LOCAL, POS_LFOURIER, POS_LQUAD, POS_DCT, POS_LDCT, POS_COUNT };
static const char* POS_NAMES[POS_COUNT] = { "uv", "fourier", "local", "lfourier", "lquad", "dct", "ldct" };

struct PosFeature {
    PosKind kind;
    int n = 0;   // octave count for the fourier kinds
    int count() const {
        switch (kind) {
        case POS_UV:       return 2;
        case POS_FOURIER:  return 4 * n;
        case POS_LOCAL:    return 2;
        case POS_LFOURIER: return 4 * n;
        case POS_LQUAD:    return 3;
        case POS_DCT:      return 2 * n;
        case POS_LDCT:     return 2 * n;
        default:           return 0;
        }
    }
};

struct PosEnc {
    std::vector<PosFeature> feats;
    std::string spec;   // canonical text form, stored in the model file

    int count() const { int c = 0; for (auto& f : feats) c += f.count(); return c; }

    // Parse "uv,fourier:2,local". "none" or "" gives no positional input.
    bool parse(const std::string& s) {
        feats.clear(); spec.clear();
        if (s == "none" || s.empty()) { spec = "none"; return true; }
        for (size_t a = 0; a < s.size();) {
            size_t e = s.find(',', a);
            if (e == std::string::npos) e = s.size();
            std::string item = s.substr(a, e - a);
            a = e + 1;
            if (item.empty()) continue;
            PosFeature f;
            size_t colon = item.find(':');
            std::string name = item.substr(0, colon);
            f.n = (colon == std::string::npos) ? 1 : atoi(item.substr(colon + 1).c_str());
            int k = 0;
            while (k < POS_COUNT && name != POS_NAMES[k]) k++;
            if (k == POS_COUNT || f.n < 0) return false;
            f.kind = (PosKind)k;
            if (f.count() == 0) continue;
            feats.push_back(f);
            if (!spec.empty()) spec += ",";
            spec += name;
            if (f.kind == POS_FOURIER || f.kind == POS_LFOURIER || f.kind == POS_DCT || f.kind == POS_LDCT) spec += ":" + std::to_string(f.n);
        }
        if (spec.empty()) spec = "none";
        return true;
    }

    // u,v: global pixel-center UV in [0,1]. t: the bilinear tap for that UV.
    inline int encode(float u, float v, const BilinearTap& t, float* f) const {
        int k = 0;
        for (const PosFeature& pf : feats) {
            switch (pf.kind) {
            case POS_UV:
                f[k++] = u * 2.0f - 1.0f;
                f[k++] = v * 2.0f - 1.0f;
                break;
            case POS_FOURIER: {
                float fr = 2.0f * PI;
                for (int o = 0; o < pf.n; o++, fr *= 2.0f) {
                    f[k++] = std::sin(fr * u); f[k++] = std::cos(fr * u);
                    f[k++] = std::sin(fr * v); f[k++] = std::cos(fr * v);
                }
                break;
            }
            case POS_LOCAL:
                f[k++] = t.fx * 2.0f - 1.0f;
                f[k++] = t.fy * 2.0f - 1.0f;
                break;
            case POS_LFOURIER: {
                float fr = 2.0f * PI;
                for (int o = 0; o < pf.n; o++, fr *= 2.0f) {
                    f[k++] = std::sin(fr * t.fx); f[k++] = std::cos(fr * t.fx);
                    f[k++] = std::sin(fr * t.fy); f[k++] = std::cos(fr * t.fy);
                }
                break;
            }
            case POS_LQUAD: {
                float x = t.fx * 2.0f - 1.0f, y = t.fy * 2.0f - 1.0f;
                f[k++] = x * y; f[k++] = x * x; f[k++] = y * y;
                break;
            }
            case POS_DCT:
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * u); f[k++] = std::cos(PI * o * v); }
                break;
            case POS_LDCT:
                for (int o = 1; o <= pf.n; o++) { f[k++] = std::cos(PI * o * t.fx); f[k++] = std::cos(PI * o * t.fy); }
                break;
            default: break;
            }
        }
        return k;
    }
};

// ---------------------------------------------------------------- decoder
struct Decoder {
    const Options* opt;
    Latent lat;
    MLP mlp;
    PosEnc pos;
    int W = 0, H = 0; // output image size

    int nin() const { return lat.C + pos.count(); }

    // Build the MLP input for pixel (px,py) from a given latent array:
    // the bilinear latent sample followed by the positional features.
    inline void features(const float* z, int px, int py, float* f) const {
        float u = (px + 0.5f) / W, v = (py + 0.5f) / H;
        BilinearTap t = bilinear_tap(lat, u, v);
        sample_latent(lat, z, t, f);
        pos.encode(u, v, t, f + lat.C);
    }

    inline void pixel(const float* p, const float* z, int px, int py, float* rgb) const {
        float f[MAXH];
        features(z, px, py, f);
        mlp_forward(mlp, p, f, rgb, opt->clamp_out);
    }

    // Full-image decode into img (parallel over rows).
    void decode_full(const float* p, const float* z, Image& img) const {
        img.w = W; img.h = H; img.rgb.resize((size_t)W * H * 3);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixel(p, z, x, y, &img.rgb[((size_t)y * W + x) * 3]);
    }

    // Per-pixel squared error (summed over channels) of a full decode.
    void decode_err(const float* p, const float* z, const Image& target, std::vector<float>& err) const {
        err.resize((size_t)W * H);
#pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float rgb[3];
                pixel(p, z, x, y, rgb);
                const float* t = &target.rgb[((size_t)y * W + x) * 3];
                float e = 0;
                for (int c = 0; c < 3; c++) { float d = rgb[c] - t[c]; e += d * d; }
                err[(size_t)y * W + x] = e;
            }
    }

    // Mean squared error on a pixel index subset (serial; called per-thread).
    double loss_subset(const float* p, const float* z, const Image& target, const std::vector<int>& idx) const {
        double s = 0;
        for (int i : idx) {
            int px = i % W, py = i / W;
            float rgb[3];
            pixel(p, z, px, py, rgb);
            const float* t = &target.rgb[(size_t)i * 3];
            for (int c = 0; c < 3; c++) { float d = rgb[c] - t[c]; s += d * d; }
        }
        return s / (3.0 * idx.size());
    }
};

static double mse_of(const Image& a, const Image& b) {
    double s = 0;
    for (size_t i = 0; i < a.rgb.size(); i++) { double d = a.rgb[i] - b.rgb[i]; s += d * d; }
    return s / a.rgb.size();
}
static double psnr_of(double mse) { return mse > 0 ? 10.0 * std::log10(1.0 / mse) : 99.0; }

// ---------------------------------------------------------------- Adam
struct Adam {
    std::vector<float> m, v;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int t = 0;
    void init(size_t n) { m.assign(n, 0); v.assign(n, 0); t = 0; }
    // Gradient-descent step: theta -= lr * adam(g)
    void step(std::vector<float>& theta, const std::vector<float>& g, float lr) {
        t++;
        float c1 = 1.0f - std::pow(b1, (float)t), c2 = 1.0f - std::pow(b2, (float)t);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)theta.size(); i++) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            float mh = m[i] / c1, vh = v[i] / c2;
            theta[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
};

// ---------------------------------------------------------------- ES: MLP
// Antithetic ES on the MLP weights, evaluated on one shared random minibatch.
// g = 1/(2 N sigma) * sum_i [L(p + s e_i) - L(p - s e_i)] e_i
struct MlpTrainer {
    Adam adam;
    std::vector<float> grad;
    std::vector<float> eps;   // N x P
    std::vector<int> batch;

    void init(size_t P) { adam.init(P); grad.assign(P, 0); }

    // Returns the mean minibatch loss across all evaluations (for stats).
    double step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o, double& diff_std) {
        const size_t P = D.mlp.size();
        const int N = o.mlp_pairs;
        eps.resize((size_t)N * P);
        std::normal_distribution<float> Nd(0.0f, 1.0f);
        for (auto& e : eps) e = Nd(rng);

        batch.resize(o.mlp_batch);
        std::uniform_int_distribution<int> U(0, D.W * D.H - 1);
        for (auto& b : batch) b = U(rng);

        std::vector<double> dl(N), lsum(N);
        const float* p0 = D.mlp.p.data();
#pragma omp parallel
        {
            std::vector<float> pp(P), pm(P);
#pragma omp for schedule(dynamic)
            for (int i = 0; i < N; i++) {
                const float* e = &eps[(size_t)i * P];
                for (size_t k = 0; k < P; k++) { pp[k] = p0[k] + o.mlp_sigma * e[k]; pm[k] = p0[k] - o.mlp_sigma * e[k]; }
                double lp = D.loss_subset(pp.data(), D.lat.z.data(), target, batch);
                double lm = D.loss_subset(pm.data(), D.lat.z.data(), target, batch);
                dl[i] = lp - lm; lsum[i] = 0.5 * (lp + lm);
            }
        }
        double mean = 0, mean2 = 0, lmean = 0;
        for (int i = 0; i < N; i++) { mean += dl[i]; mean2 += dl[i] * dl[i]; lmean += lsum[i]; }
        mean /= N; mean2 /= N; lmean /= N;
        diff_std = std::sqrt(std::max(0.0, mean2 - mean * mean));

        std::fill(grad.begin(), grad.end(), 0.0f);
        float scale = 1.0f / (2.0f * N * o.mlp_sigma);
        for (int i = 0; i < N; i++) {
            float w = (float)dl[i] * scale;
            const float* e = &eps[(size_t)i * P];
            for (size_t k = 0; k < P; k++) grad[k] += w * e[k];
        }
        adam.step(D.mlp.p, grad, o.mlp_lr);
        return lmean;
    }
};

// ---------------------------------------------------------------- ES: latent
// All texels are perturbed simultaneously. For each antithetic pair we decode
// the full image twice and get per-pixel squared errors e+ and e-. Each pixel's
// (e+ - e-) is attributed to the (up to) 4 texels its bilinear tap reads, so
// texel (x,y) accumulates the loss change over exactly its footprint.
//   g[x,y,c] = 1/(2 K sigma) * sum_pairs [ sum_{footprint} (e+ - e-) / (3 W H) ] * eps[x,y,c]
struct LatentTrainer {
    Adam adam;
    std::vector<float> grad, eps, zp, zm;
    std::vector<float> ep, em;   // per-pixel errors
    std::vector<float> dtex;     // per-texel footprint loss difference

    void init(size_t n) { adam.init(n); grad.assign(n, 0); }

    void step(Decoder& D, const Image& target, std::mt19937& rng, const Options& o) {
        const Latent& L = D.lat;
        const size_t n = L.size();
        const int K = o.lat_pairs;
        eps.resize(n); zp.resize(n); zm.resize(n);
        dtex.resize((size_t)L.W * L.H);
        std::fill(grad.begin(), grad.end(), 0.0f);
        std::normal_distribution<float> Nd(0.0f, 1.0f);
        const float inv_px = 1.0f / (3.0f * D.W * D.H);
        const float scale = 1.0f / (2.0f * K * o.lat_sigma);

        for (int k = 0; k < K; k++) {
            for (size_t i = 0; i < n; i++) {
                eps[i] = Nd(rng);
                zp[i] = L.z[i] + o.lat_sigma * eps[i];
                zm[i] = L.z[i] - o.lat_sigma * eps[i];
            }
            D.decode_err(D.mlp.p.data(), zp.data(), target, ep);
            D.decode_err(D.mlp.p.data(), zm.data(), target, em);

            // Scatter per-pixel loss differences into the texels each pixel reads.
            std::fill(dtex.begin(), dtex.end(), 0.0f);
            for (int py = 0; py < D.H; py++) {
                float v = (py + 0.5f) / D.H;
                for (int px = 0; px < D.W; px++) {
                    float u = (px + 0.5f) / D.W;
                    BilinearTap t = bilinear_tap(L, u, v);
                    float d = (ep[(size_t)py * D.W + px] - em[(size_t)py * D.W + px]) * inv_px;
                    dtex[(size_t)t.y0 * L.W + t.x0] += d;
                    if (t.x1 != t.x0) dtex[(size_t)t.y0 * L.W + t.x1] += d;
                    if (t.y1 != t.y0) {
                        dtex[(size_t)t.y1 * L.W + t.x0] += d;
                        if (t.x1 != t.x0) dtex[(size_t)t.y1 * L.W + t.x1] += d;
                    }
                }
            }
            for (int ty = 0; ty < L.H; ty++)
                for (int tx = 0; tx < L.W; tx++) {
                    float w = dtex[(size_t)ty * L.W + tx] * scale;
                    size_t base = ((size_t)ty * L.W + tx) * L.C;
                    for (int c = 0; c < L.C; c++) grad[base + c] += w * eps[base + c];
                }
        }
        adam.step(D.lat.z, grad, o.lat_lr);
    }
};

// ---------------------------------------------------------------- I/O helpers
static void save_latent_png(const std::string& path, const Latent& L) {
    // Channels laid out side by side, each mapped from [-1.5,1.5] to [0,255].
    int w = L.W * L.C, h = L.H;
    std::vector<unsigned char> buf((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int c = 0; c < L.C; c++)
            for (int x = 0; x < L.W; x++) {
                float v = L.texel(x, y)[c];
                float m = std::min(1.0f, std::max(0.0f, v / 3.0f + 0.5f));
                buf[(size_t)y * w + c * L.W + x] = (unsigned char)std::lround(m * 255.0f);
            }
    stbi_write_png(path.c_str(), w, h, 1, buf.data(), w);
}

static void save_model(const std::string& path, const Decoder& D) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    // Header: magic, LW, LH, LC, nin, poslen, act, nlayers, then the hidden widths,
    // then poslen bytes of positional spec text.
    int hdr[8] = { 0x4E544333, D.lat.W, D.lat.H, D.lat.C, D.mlp.nin, (int)D.pos.spec.size(), (int)D.mlp.act, (int)D.mlp.hidden.size() };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(D.mlp.hidden.data(), sizeof(int), D.mlp.hidden.size(), f);
    fwrite(D.pos.spec.data(), 1, D.pos.spec.size(), f);
    fwrite(D.lat.z.data(), sizeof(float), D.lat.z.size(), f);
    fwrite(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f);
    fclose(f);
}

static void latent_stats(const Latent& L, float& mean, float& sd, float& mx) {
    double s = 0, s2 = 0; mx = 0;
    for (float v : L.z) { s += v; s2 += (double)v * v; mx = std::max(mx, std::fabs(v)); }
    mean = (float)(s / L.size());
    sd = (float)std::sqrt(std::max(0.0, s2 / L.size() - (s / L.size()) * (s / L.size())));
}


// ---------------------------------------------------------------- bitrate
// Effective compressed size estimate. The latent is quantized to 8 bits per
// channel with a per-channel min/max scale (2 floats per channel overhead), and
// the MLP weights are counted as fp16. Two latent numbers are reported: raw
// 8 bits/texel/channel, and the zeroth-order entropy of the quantized symbols.
// The dequantized latent is also returned so the caller can measure the PSNR
// the codec would actually achieve at that bitrate.
struct BitrateStats {
    double bpp_fp32;      // everything stored as fp32
    double bpp_q8;        // 8-bit latent (raw) + fp16 MLP
    double bpp_q8_ent;    // entropy-coded 8-bit latent + fp16 MLP
    double bits_mlp;      // fp16 MLP bits
};

static BitrateStats bitrate_stats(const Latent& L, const MLP& m, int W, int H, int qbits, std::vector<float>& zq) {
    const int levels = (1 << qbits) - 1;
    BitrateStats s;
    const double npix = (double)W * H;
    s.bits_mlp = m.size() * 16.0;
    s.bpp_fp32 = (L.size() * 32.0 + m.size() * 32.0) / npix;
    zq.resize(L.size());
    double ent_bits = 0;
    for (int c = 0; c < L.C; c++) {
        float lo = 1e30f, hi = -1e30f;
        for (size_t i = c; i < L.size(); i += L.C) { lo = std::min(lo, L.z[i]); hi = std::max(hi, L.z[i]); }
        float range = std::max(hi - lo, 1e-6f);
        std::vector<int> hist(levels + 1, 0);
        for (size_t i = c; i < L.size(); i += L.C) {
            int q = (int)std::lround((L.z[i] - lo) / range * levels);
            q = std::max(0, std::min(levels, q));
            hist[q]++;
            zq[i] = lo + q / (float)levels * range;
        }
        double n = (double)L.W * L.H;
        for (int q = 0; q <= levels; q++)
            if (hist[q]) ent_bits -= hist[q] * std::log2(hist[q] / n);
    }
    const double header_bits = L.C * 2 * 32.0; // per-channel min/max
    s.bpp_q8 = (L.size() * (double)qbits + header_bits + s.bits_mlp) / npix;
    s.bpp_q8_ent = (ent_bits + header_bits + s.bits_mlp) / npix;
    return s;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int k = 1) { if (i + k >= argc) { usage(); exit(1); } return argv[i + k]; };
        if (a == "--out") { o.outdir = next(); i++; }
        else if (a == "--crop") { o.crop = atoi(next()); i++; }
        else if (a == "--latent") { o.LW = atoi(next(1)); o.LH = atoi(next(2)); o.LC = atoi(next(3)); i += 3; }
        else if (a == "--hidden") { int n = atoi(next()); o.hidden = { n, n }; i++; }
        else if (a == "--mlp") {
            o.hidden.clear();
            std::string list = next(); i++;
            for (size_t s = 0; s < list.size();) {
                size_t e = list.find(',', s);
                if (e == std::string::npos) e = list.size();
                if (e > s) o.hidden.push_back(atoi(list.substr(s, e - s).c_str()));
                s = e + 1;
            }
        }
        else if (a == "--act") { o.act = next(); i++; }
        else if (a == "--nfreq") { int n = atoi(next()); i++; o.pos = n > 0 ? "uv,fourier:" + std::to_string(n) : "uv"; }
        else if (a == "--pos") { o.pos = next(); i++; }
        else if (a == "--clamp") { o.clamp_out = true; }
        else if (a == "--iters") { o.iters = atoi(next()); i++; }
        else if (a == "--save-every") { o.save_every = atoi(next()); i++; }
        else if (a == "--print-every") { o.print_every = atoi(next()); i++; }
        else if (a == "--seed") { o.seed = (unsigned)atoi(next()); i++; }
        else if (a == "--mlp-pairs") { o.mlp_pairs = atoi(next()); i++; }
        else if (a == "--mlp-batch") { o.mlp_batch = atoi(next()); i++; }
        else if (a == "--mlp-sigma") { o.mlp_sigma = (float)atof(next()); i++; }
        else if (a == "--mlp-lr") { o.mlp_lr = (float)atof(next()); i++; }
        else if (a == "--mlp-every") { o.mlp_every = atoi(next()); i++; }
        else if (a == "--lat-pairs") { o.lat_pairs = atoi(next()); i++; }
        else if (a == "--lat-sigma") { o.lat_sigma = (float)atof(next()); i++; }
        else if (a == "--lat-lr") { o.lat_lr = (float)atof(next()); i++; }
        else if (a == "--lat-init") { o.lat_init = (float)atof(next()); i++; }
        else if (a == "--threads") { o.threads = atoi(next()); i++; }
        else if (a == "--qbits") { o.qbits = atoi(next()); i++; }
        else if (a == "--load") { o.load = next(); i++; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] == '-') { printf("unknown option %s\n", a.c_str()); usage(); return 1; }
        else o.input = a;
    }
    Act act;
    if (!parse_act(o.act, act)) { printf("unknown activation %s\n", o.act.c_str()); return 1; }
    if (o.hidden.empty() || (int)o.hidden.size() > MAXL) { printf("need 1..%d hidden layers\n", MAXL); return 1; }
    for (int h : o.hidden) if (h < 1 || h > MAXH) { printf("hidden width must be 1..%d\n", MAXH); return 1; }
#ifdef _OPENMP
    if (o.threads > 0) omp_set_num_threads(o.threads);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif

#ifdef _WIN32
    _mkdir(o.outdir.c_str());
#else
    mkdir(o.outdir.c_str(), 0755);
#endif

    Image target;
    if (!load_png(o.input, target, o.crop)) {
        printf("could not load %s, using synthetic %dx%d image\n", o.input.c_str(), o.crop ? o.crop : 512, o.crop ? o.crop : 512);
        make_synthetic(target, o.crop ? o.crop : 512);
    }
    save_png(o.outdir + "/target.png", target);

    std::mt19937 rng(o.seed);
    Decoder D;
    D.opt = &o;
    if (!D.pos.parse(o.pos)) { printf("bad --pos spec: %s\n", o.pos.c_str()); return 1; }
    D.W = target.w; D.H = target.h;
    D.lat.W = o.LW; D.lat.H = o.LH; D.lat.C = o.LC;
    D.lat.z.resize((size_t)o.LW * o.LH * o.LC);
    {
        std::normal_distribution<float> N(0.0f, o.lat_init);
        for (auto& z : D.lat.z) z = N(rng);
    }
    if (D.nin() > MAXH) { printf("too many MLP inputs\n"); return 1; }
    D.mlp.init(D.nin(), o.hidden, act, rng);
    if (!o.load.empty()) {
        FILE* f = fopen(o.load.c_str(), "rb");
        int hdr[8];
        if (!f || fread(hdr, sizeof(int), 6, f) != 6) { printf("cannot read %s\n", o.load.c_str()); return 1; }
        std::vector<int> saved_hidden;
        std::string saved_pos;
        bool hdr_ok;
        auto nfreq_spec = [](int n) { return n > 0 ? "uv,fourier:" + std::to_string(n) : std::string("uv"); };
        if (hdr[0] == 0x4E544333 || hdr[0] == 0x4E544332) {
            bool v3 = hdr[0] == 0x4E544333;
            hdr_ok = fread(hdr + 6, sizeof(int), 2, f) == 2 && hdr[7] >= 1 && hdr[7] <= MAXL;
            if (hdr_ok) {
                saved_hidden.resize(hdr[7]);
                hdr_ok = fread(saved_hidden.data(), sizeof(int), saved_hidden.size(), f) == saved_hidden.size();
            }
            if (hdr_ok) {
                if (v3) {
                    // hdr[5] is the positional spec length, followed by the spec text.
                    hdr_ok = hdr[5] >= 0 && hdr[5] < 4096;
                    if (hdr_ok) {
                        saved_pos.resize(hdr[5]);
                        hdr_ok = fread(&saved_pos[0], 1, saved_pos.size(), f) == saved_pos.size();
                    }
                } else {
                    saved_pos = nfreq_spec(hdr[5]);   // v2 stored nfreq
                }
            }
        } else {
            // Legacy header: LW, LH, LC, nin, nh, nfreq; always two leaky layers of width nh.
            saved_hidden = { hdr[4], hdr[4] };
            saved_pos = nfreq_spec(hdr[5]);
            int lw = hdr[0], lh = hdr[1], lc = hdr[2], nin = hdr[3];
            hdr[1] = lw; hdr[2] = lh; hdr[3] = lc; hdr[4] = nin; hdr[6] = ACT_LEAKY; hdr[7] = 2;
            hdr_ok = true;
        }
        if (!hdr_ok || hdr[1] != D.lat.W || hdr[2] != D.lat.H || hdr[3] != D.lat.C || hdr[4] != D.mlp.nin
            || saved_pos != D.pos.spec || hdr[6] != (int)act || saved_hidden != o.hidden) {
            std::string mlp;
            for (size_t k = 0; k < saved_hidden.size(); k++) mlp += (k ? "," : "") + std::to_string(saved_hidden[k]);
            printf("%s was saved with --latent %d %d %d --mlp %s --act %s --pos %s; pass the same options\n",
                o.load.c_str(), hdr[1], hdr[2], hdr[3], mlp.c_str(), act_name((Act)std::max(0, std::min(3, hdr[6]))), saved_pos.c_str());
            fclose(f);
            return 1;
        }
        bool ok = fread(D.lat.z.data(), sizeof(float), D.lat.z.size(), f) == D.lat.z.size()
               && fread(D.mlp.p.data(), sizeof(float), D.mlp.p.size(), f) == D.mlp.p.size();
        fclose(f);
        if (!ok) { printf("truncated %s\n", o.load.c_str()); return 1; }
        printf("loaded   : %s\n", o.load.c_str());
    }

    const double bits_latent = D.lat.size() * 32.0;
    const double bits_mlp = D.mlp.size() * 32.0;
    printf("image    : %dx%d (%s)\n", D.W, D.H, o.input.c_str());
    printf("latent   : %dx%dx%d = %zu floats\n", o.LW, o.LH, o.LC, D.lat.size());
    printf("mlp      : %s = %zu params, %s hidden, %s output\n", D.mlp.describe().c_str(), D.mlp.size(), act_name(D.mlp.act), o.clamp_out ? "clamp" : "sigmoid");
    printf("inputs   : %d latent + %d positional (%s)\n", D.lat.C, D.pos.count(), D.pos.spec.c_str());
    printf("bitrate  : fp32 %.3f bpp (latent %.3f + mlp %.3f); %d-bit latent + fp16 mlp %.3f bpp raw\n",
        (bits_latent + bits_mlp) / (D.W * D.H), bits_latent / (D.W * D.H), bits_mlp / (D.W * D.H),
        o.qbits, (D.lat.size() * (double)o.qbits + D.lat.C * 64.0 + D.mlp.size() * 16.0) / (D.W * D.H));
    printf("mlp ES   : %d pairs, batch %d, sigma %g, lr %g, every %d\n", o.mlp_pairs, o.mlp_batch, o.mlp_sigma, o.mlp_lr, o.mlp_every);
    printf("latent ES: %d pairs (full image), sigma %g, lr %g\n", o.lat_pairs, o.lat_sigma, o.lat_lr);
    fflush(stdout);

    MlpTrainer mt; mt.init(D.mlp.size());
    LatentTrainer lt; lt.init(D.lat.size());

    Image recon, recon_q;
    std::vector<float> zq;
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    D.decode_full(D.mlp.p.data(), D.lat.z.data(), recon);
    double mse = mse_of(target, recon);
    printf("iter %6d  mse %.6f  psnr %6.2f dB  (initial)\n", 0, mse, psnr_of(mse));
    save_png(o.outdir + "/recon_000000.png", recon);
    save_latent_png(o.outdir + "/latent_000000.png", D.lat);
    fflush(stdout);

    double best_psnr = -1;
    double batch_loss = 0, diff_std = 0;
    for (int it = 1; it <= o.iters; it++) {
        if (it % o.mlp_every == 0) batch_loss = mt.step(D, target, rng, o, diff_std);
        lt.step(D, target, rng, o);

        bool do_print = (it % o.print_every == 0) || it == o.iters;
        bool do_save = (it % o.save_every == 0) || it == o.iters;
        if (do_print || do_save) {
            D.decode_full(D.mlp.p.data(), D.lat.z.data(), recon);
            mse = mse_of(target, recon);
            double ps = psnr_of(mse);
            best_psnr = std::max(best_psnr, ps);
            float lm, lsd, lmx; latent_stats(D.lat, lm, lsd, lmx);
            // Bitrate and quality when the latent is actually quantized to 8 bits.
            BitrateStats bs = bitrate_stats(D.lat, D.mlp, D.W, D.H, o.qbits, zq);
            D.decode_full(D.mlp.p.data(), zq.data(), recon_q);
            double ps_q = psnr_of(mse_of(target, recon_q));
            double sec = elapsed();
            printf("iter %6d  mse %.6f  psnr %6.2f dB  best %6.2f | q%d psnr %6.2f @ %.3f bpp (ent %.3f) | mlp batch %.5f dstd %.2e | lat mean %+.3f sd %.3f max %.2f | %.1fs (%.2f it/s)\n",
                it, mse, ps, best_psnr, o.qbits, ps_q, bs.bpp_q8, bs.bpp_q8_ent, batch_loss, diff_std, lm, lsd, lmx, sec, it / sec);
            fflush(stdout);
        }
        if (do_save) {
            char name[64];
            snprintf(name, sizeof(name), "/recon_%06d.png", it);
            save_png(o.outdir + name, recon);
            snprintf(name, sizeof(name), "/latent_%06d.png", it);
            save_latent_png(o.outdir + name, D.lat);
            save_model(o.outdir + "/model.bin", D);
        }
    }
    {
        BitrateStats bs = bitrate_stats(D.lat, D.mlp, D.W, D.H, o.qbits, zq);
        D.decode_full(D.mlp.p.data(), zq.data(), recon_q);
        double ps_q = psnr_of(mse_of(target, recon_q));
        save_png(o.outdir + "/recon_q_final.png", recon_q);
        printf("done: final psnr %.2f dB (best %.2f) at fp32 %.3f bpp | %d-bit latent + fp16 mlp: psnr %.2f dB at %.3f bpp raw, %.3f bpp entropy-coded | %.1fs\n",
            psnr_of(mse), best_psnr, bs.bpp_fp32, o.qbits, ps_q, bs.bpp_q8, bs.bpp_q8_ent, elapsed());
    }
    return 0;
}
