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
