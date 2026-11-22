/*
 * Portable WebAssembly SIMD128 kernels.
 *
 * No x86-specific 256-bit lane or pointer-width assumptions are allowed here.
 * The RTCD selection is opt-in and --asm c still selects the C reference.
 */
#include "aom_dsp_rtcd.h"
#include <wasm_simd128.h>
#include <stdint.h>
#if defined(SVT_WASM_SIMD_TRACE)
#include <stdio.h>
#endif

#ifndef __wasm_simd128__
#error "SVT_AV1_WASM_SIMD requires -msimd128"
#endif

void svt_copy_mi_map_grid_wasm_simd(MbModeInfo** mi_grid_ptr, uint32_t mi_stride,
                                    uint8_t num_rows, uint8_t num_cols) {
#if defined(SVT_WASM_SIMD_TRACE)
    static unsigned int call_count;
    if (__atomic_fetch_add(&call_count, 1, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "SVT_WASM_SIMD_COPY_MI_USED ptr_bytes=%zu\n", sizeof(void*));
#endif
    if (!num_rows || !num_cols) return;
    MbModeInfo* target = mi_grid_ptr[0];
#if __SIZEOF_POINTER__ == 4
    v128_t broadcast = wasm_i32x4_splat((int32_t)(uintptr_t)target);
#elif __SIZEOF_POINTER__ == 8
    v128_t broadcast = wasm_i64x2_splat((int64_t)(uintptr_t)target);
#else
#error "Unsupported WebAssembly pointer size"
#endif
    for (uint8_t y = 0; y < num_rows; ++y) {
        MbModeInfo** row = mi_grid_ptr + (size_t)y * mi_stride;
        if (num_cols == 1) {
            row[0] = target;
        } else {
#if __SIZEOF_POINTER__ == 4
            if (num_cols == 2) {
                wasm_v128_store64_lane(row, broadcast, 0);
            } else {
                for (uint8_t x = 0; x < num_cols; x += 4)
                    wasm_v128_store(row + x, broadcast);
            }
#else
            for (uint8_t x = 0; x < num_cols; x += 2)
                wasm_v128_store(row + x, broadcast);
#endif
        }
    }
}

/* Sum of absolute transform coefficients; all 32-bit lane totals fit int32
 * for the documented 1024 samples of [-32640, 32640]. */
int svt_aom_satd_wasm_simd(const TranLow* coeff, int length) {
#if defined(SVT_WASM_SIMD_TRACE)
    static unsigned int called;
    if (__atomic_fetch_add(&called, 1, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "SVT_WASM_SIMD_SATD_USED\n");
#endif
    int i = 0;
    v128_t accumulator = wasm_i32x4_splat(0);
    for (; i + 4 <= length; i += 4) {
        const v128_t samples = wasm_v128_load(coeff + i);
        accumulator = wasm_i32x4_add(accumulator, wasm_i32x4_abs(samples));
    }
    int total = wasm_i32x4_extract_lane(accumulator, 0) + wasm_i32x4_extract_lane(accumulator, 1)
              + wasm_i32x4_extract_lane(accumulator, 2) + wasm_i32x4_extract_lane(accumulator, 3);
    for (; i < length; ++i) total += coeff[i] < 0 ? -coeff[i] : coeff[i];
    return total;
}


/* Horizontal sum of four signed 32-bit lanes. */
static inline int64_t wasm_sse_hsum(v128_t v) {
    return (int64_t)wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1)
         + wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

/* 8-bit squared prediction error; accumulate for up to 256 samples
 * before widening to int64 to avoid lane overflow. */
int64_t svt_aom_sse_wasm_simd(const uint8_t* a, int a_stride,
                              const uint8_t* b, int b_stride, int width, int height) {
#if defined(SVT_WASM_SIMD_TRACE)
    static unsigned int called;
    if (__atomic_fetch_add(&called, 1, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "SVT_WASM_SIMD_SSE8_USED\n");
#endif
    int64_t total = 0;
    for (int y = 0; y < height; ++y) {
        int x = 0;
        v128_t accumulator = wasm_i32x4_splat(0);
        for (; x + 16 <= width; x += 16) {
            const v128_t av = wasm_v128_load(a + x);
            const v128_t bv = wasm_v128_load(b + x);
            const v128_t lo = wasm_i16x8_sub(wasm_u16x8_extend_low_u8x16(av), wasm_u16x8_extend_low_u8x16(bv));
            const v128_t hi = wasm_i16x8_sub(wasm_u16x8_extend_high_u8x16(av), wasm_u16x8_extend_high_u8x16(bv));
            accumulator = wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(lo, lo));
            accumulator = wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(hi, hi));
            if (((x + 16) & 255) == 0) {
                total += wasm_sse_hsum(accumulator);
                accumulator = wasm_i32x4_splat(0);
            }
        }
        total += wasm_sse_hsum(accumulator);
        for (; x < width; ++x) {
            int diff = (int)a[x] - (int)b[x];
            total += diff * diff;
        }
        a += a_stride;
        b += b_stride;
    }
    return total;
}

/* The high-bit-depth input buffers contain uint16_t samples, and the stride
 * is counted in samples rather than bytes (matching the SVT C reference).
 * SVT-AV1 supports 8/10-bit input; the dot-product lanes fit these values.
 */
int64_t svt_aom_highbd_sse_wasm_simd(const uint8_t* a8, int a_stride,
                                     const uint8_t* b8, int b_stride, int width, int height) {
#if defined(SVT_WASM_SIMD_TRACE)
    static unsigned int called;
    if (__atomic_fetch_add(&called, 1, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "SVT_WASM_SIMD_SSE10_USED\n");
#endif
    const uint16_t* a = (const uint16_t*)a8;
    const uint16_t* b = (const uint16_t*)b8;
    int64_t total = 0;
    for (int y = 0; y < height; ++y) {
        int x = 0;
        v128_t accumulator = wasm_i32x4_splat(0);
        for (; x + 16 <= width; x += 16) {
            const v128_t a0 = wasm_v128_load(a + x);
            const v128_t a1 = wasm_v128_load(a + x + 8);
            const v128_t b0 = wasm_v128_load(b + x);
            const v128_t b1 = wasm_v128_load(b + x + 8);
            const v128_t d0 = wasm_i16x8_sub(a0, b0);
            const v128_t d1 = wasm_i16x8_sub(a1, b1);
            accumulator = wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(d0, d0));
            accumulator = wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(d1, d1));
            if (((x + 16) & 255) == 0) {
                total += wasm_sse_hsum(accumulator);
                accumulator = wasm_i32x4_splat(0);
            }
        }
        total += wasm_sse_hsum(accumulator);
        for (; x < width; ++x) {
            int diff = (int)a[x] - (int)b[x];
            total += diff * diff;
        }
        a += a_stride;
        b += b_stride;
    }
    return total;
}
