#include <stdint.h>
#include <stddef.h>
#include <emscripten/emscripten.h>
#include "src/webp/encode.h"
#include "src/webp/decode.h"

EMSCRIPTEN_KEEPALIVE
int webp_encode_rgba(const uint8_t *rgba, int width, int height, int stride,
                     float quality, uint8_t **out_buf, size_t *out_size)
{
    if (!rgba || width <= 0 || height <= 0 || stride <= 0 || !out_buf || !out_size)
        return 0;
    uint8_t *out = NULL;
    size_t sz = WebPEncodeRGBA(rgba, width, height, stride, quality, &out);
    if (sz == 0 || !out)
        return 0;
    *out_buf = out;
    *out_size = sz;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
int webp_encode_lossless_rgba(const uint8_t *rgba, int width, int height, int stride,
                              uint8_t **out_buf, size_t *out_size)
{
    if (!rgba || width <= 0 || height <= 0 || stride <= 0 || !out_buf || !out_size)
        return 0;
    uint8_t *out = NULL;
    size_t sz = WebPEncodeLosslessRGBA(rgba, width, height, stride, &out);
    if (sz == 0 || !out)
        return 0;
    *out_buf = out;
    *out_size = sz;
    return 1;
}

// Simple wrapper that decodes a WebP (lossy or lossless) into RGBA32.
// Returns 1 on success, 0 on failure.
// out_rgba: pointer to buffer pointer that will receive allocated image data (must be freed with webp_free)
// width/height: output dimensions
EMSCRIPTEN_KEEPALIVE
int webp_decode_rgba(const uint8_t *webp_data, size_t data_size, uint8_t **out_rgba, int *width, int *height)
{
    if (!webp_data || data_size == 0 || !out_rgba || !width || !height)
        return 0;
    int w = 0, h = 0;
    if (!WebPGetInfo(webp_data, data_size, &w, &h) || w <= 0 || h <= 0)
        return 0;
    uint8_t *rgba = WebPDecodeRGBA(webp_data, data_size, &w, &h);
    if (!rgba)
        return 0;
    *out_rgba = rgba;
    *width = w;
    *height = h;
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void webp_free(void *p)
{
    WebPFree(p);
}
