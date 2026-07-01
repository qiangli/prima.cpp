// Standalone dequant self-test for GGML_TYPE_MXFP4.
// Hand-constructs block_mxfp4 values and verifies dequantize_row_mxfp4 against
// an independent reference: element = kvalues_mxfp4[nibble] * 2^(e-127)/...
// (the E8M0 "half" scale used by MXFP4).
//
// Build (from repo root, after `make`):
//   cc -O2 -Iggml/include -Iggml/src tests/test-mxfp4-dequant.c \
//      ggml/src/ggml.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o \
//      ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-blas.o \
//      ggml/src/llamafile/sgemm.o -o test-mxfp4-dequant \
//      -pthread -framework Accelerate
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// Mirror the real block layout (ggml-common.h).
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;

// Symbol exported from ggml-quants.o (C linkage).
extern void dequantize_row_mxfp4(const block_mxfp4 * x, float * y, int64_t k);

// Independent reference copies (must match ggml-impl.h / ggml-quants.c).
static const int8_t ref_kvalues[16] =
    {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static float ref_e8m0_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) bits = 0x00200000u << x;
    else       bits = (uint32_t)(x - 1) << 23;
    float f; __builtin_memcpy(&f, &bits, sizeof(f)); return f;
}

static int check_block(uint8_t e, const char * label) {
    block_mxfp4 b;
    b.e = e;
    // low nibble j, high nibble j -> exercises all 16 kvalues (twice per block).
    for (int j = 0; j < 16; j++) b.qs[j] = (uint8_t)(j | (j << 4));

    float y[32];
    dequantize_row_mxfp4(&b, y, 32);

    const float d = ref_e8m0_half(e);
    printf("== %s: e=%u  half-scale d=%.9g ==\n", label, e, d);

    int fails = 0;
    for (int j = 0; j < 16; j++) {
        float expect = ref_kvalues[j] * d;   // element j  (low nibble)
        float got0 = y[j];                    // y[j]
        float got1 = y[j + 16];               // y[j+16] (high nibble, same nibble j)
        int ok = (got0 == expect) && (got1 == expect);
        if (!ok) fails++;
        printf("  nibble %2d -> kval %4d  expect %10.5f | y[%2d]=%10.5f  y[%2d]=%10.5f  %s\n",
               j, ref_kvalues[j], expect, j, got0, j + 16, got1, ok ? "OK" : "FAIL");
    }
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_block(127, "e=127 -> d=0.5 (true E2M1 values)");
    fails += check_block(129, "e=129 -> d=2.0 (E2M1 x4 magnitude)");
    fails += check_block(128, "e=128 -> d=1.0");
    if (fails == 0) { printf("\nALL DEQUANT CHECKS PASSED\n"); return 0; }
    printf("\n%d CHECKS FAILED\n", fails); return 1;
}
