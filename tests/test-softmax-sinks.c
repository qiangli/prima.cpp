// Standalone numerical self-test for gpt-oss "attention sinks" in soft_max.
//
// The CPU forward kernel ggml_compute_forward_soft_max_f32 is static inside
// ggml.c, so (unlike P1's dequantize_row_mxfp4) we cannot call it directly.
// Instead we drive it through the public API + a one-node graph:
//     a  = ggml_soft_max_ext(ctx, scores, NULL, 1.0f, 0.0f);
//     ggml_soft_max_add_sinks(a, sinks);     // optional
//     ggml_graph_compute_with_ctx(ctx, gf, 1);
//
// We assert BOTH:
//   (a) NULL sinks  -> bit-identical to the standard softmax (regression guard
//       proving existing Qwen/Llama models are untouched); and
//   (b) with a per-head sink k -> equals the hand-computed
//       out_i = exp(s_i - m) / ( Σ_j exp(s_j - m) + exp(k - m) ),  m = max(max_j s_j, k)
//
// Rows are sized to 3 (< 4) so ggml's NEON soft_max takes its *scalar* tail
// (plain libm expf), which lets the reference below be bit-for-bit identical.
// (The NEON 4-wide path uses a polynomial ggml_v_expf; for larger rows a tiny
// tolerance would apply — the math is the same.)
//
// Build (from repo root, after `make`) — link with the c++ driver because the
// ggml objects are compiled as C++:
//   c++ -O2 -Iggml/include -Iggml/src \
//       -x c tests/test-softmax-sinks.c -x none \
//       ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o \
//       ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o ggml/src/ggml-blas.o \
//       ggml/src/llamafile/sgemm.o -o test-softmax-sinks \
//       -pthread -framework Accelerate
//   ./test-softmax-sinks

#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define NC 3   // scores per row (kept < 4 to force ggml's scalar expf tail)

// Independent reference, mirroring ggml's exact accumulation order:
//   dp[i] = expf(s[i]-m)  (float);  sum = Σ (double)dp[i] [+ (double)expf(sink-m)];
//   inv = (float)(1.0/sum);  dp[i] *= inv.
// sink == NAN means "no sink" (standard softmax).
static void ref_softmax(const float * s, int n, float sink, float * out) {
    float m = -INFINITY;
    for (int i = 0; i < n; i++) if (s[i] > m) m = s[i];
    if (!isnan(sink) && sink > m) m = sink;

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float v = expf(s[i] - m);
        out[i]  = v;
        sum    += (double) v;
    }
    if (!isnan(sink)) {
        sum += (double) expf(sink - m);
    }
    float inv = (float) (1.0 / sum);
    for (int i = 0; i < n; i++) out[i] *= inv;
}

// Run ggml soft_max over a [NC, 1, n_head] input, optionally with per-head sinks.
// `use_sinks`: if non-zero, attach `sink_vals[n_head]`. Output written to `out`
// (n_head rows of NC).
static void run_ggml(const float * scores, int n_head,
                     int use_sinks, const float * sink_vals, float * out) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 16u*1024u*1024u,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ip);

    // scores: [NC, 1, n_head]  -> ne0=NC (kv), ne1=1 (query), ne2=n_head
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NC, 1, n_head);
    memcpy(a->data, scores, sizeof(float) * NC * n_head);

    struct ggml_tensor * sm = ggml_soft_max_ext(ctx, a, NULL, 1.0f, 0.0f);

    if (use_sinks) {
        struct ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
        memcpy(sinks->data, sink_vals, sizeof(float) * n_head);
        ggml_soft_max_add_sinks(sm, sinks);
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sm);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    memcpy(out, sm->data, sizeof(float) * NC * n_head);
    ggml_free(ctx);
}

static int cmp_rows(const char * label, const float * got, const float * ref,
                    int n_head, int exact) {
    int fails = 0;
    for (int h = 0; h < n_head; h++) {
        for (int i = 0; i < NC; i++) {
            float g = got[h*NC + i];
            float r = ref[h*NC + i];
            int ok = exact ? (g == r) : (fabsf(g - r) <= 1e-6f);
            if (!ok) fails++;
            printf("  %s head %d [%d]  ggml=%.9g  ref=%.9g  diff=%.3g  %s\n",
                   label, h, i, g, r, (double)(g - r), ok ? "OK" : "FAIL");
        }
    }
    return fails;
}

int main(void) {
    // Two heads, distinct scores + distinct sinks -> exercises per-head sinks[h].
    const int n_head = 2;
    const float scores[2*NC] = {
        2.0f,  1.0f,  0.5f,   // head 0
       -1.0f,  3.0f,  0.0f,   // head 1
    };
    const float sink_vals[2] = { 0.7f, -0.5f };

    int fails = 0;

    // ---- Case (a): NULL sinks == standard softmax (regression guard) ----
    printf("== Case A: NULL sinks -> standard softmax (bit-identical) ==\n");
    {
        float got[2*NC];
        float ref[2*NC];
        run_ggml(scores, n_head, /*use_sinks=*/0, NULL, got);
        for (int h = 0; h < n_head; h++) {
            ref_softmax(scores + h*NC, NC, /*sink=*/NAN, ref + h*NC);
        }
        fails += cmp_rows("A", got, ref, n_head, /*exact=*/1);
        // sanity: each row sums to 1
        for (int h = 0; h < n_head; h++) {
            double s = 0; for (int i = 0; i < NC; i++) s += got[h*NC+i];
            printf("  A head %d row-sum = %.9g (expect 1)\n", h, s);
        }
    }

    // ---- Case (b): per-head sink k -> hand-computed formula ----
    printf("== Case B: with per-head sink k -> exp(s_i-m)/(Sum exp(s_j-m)+exp(k-m)) ==\n");
    {
        float got[2*NC];
        float ref[2*NC];
        run_ggml(scores, n_head, /*use_sinks=*/1, sink_vals, got);
        for (int h = 0; h < n_head; h++) {
            ref_softmax(scores + h*NC, NC, /*sink=*/sink_vals[h], ref + h*NC);
        }
        fails += cmp_rows("B", got, ref, n_head, /*exact=*/1);
        // with a sink, each real-score row sums to < 1 (mass stolen by the sink)
        for (int h = 0; h < n_head; h++) {
            double s = 0; for (int i = 0; i < NC; i++) s += got[h*NC+i];
            double m = -INFINITY;
            for (int i = 0; i < NC; i++) if (scores[h*NC+i] > m) m = scores[h*NC+i];
            if (sink_vals[h] > m) m = sink_vals[h];
            double denom = 0; for (int i = 0; i < NC; i++) denom += exp(scores[h*NC+i]-m);
            denom += exp(sink_vals[h]-m);
            double sink_mass = exp(sink_vals[h]-m)/denom;
            printf("  B head %d row-sum = %.9g, sink-mass = %.9g (sum -> %.9g)\n",
                   h, s, sink_mass, s + sink_mass);
        }
    }

    printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
