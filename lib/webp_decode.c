#include <stdint.h>
#include <stddef.h>
#include <emscripten/emscripten.h>
#include "src/webp/decode.h"

// Simple wrapper that decodes a WebP (lossy or lossless) into RGBA32.
// Returns 1 on success, 0 on failure.
// out_rgba: pointer to buffer pointer that will receive allocated image data (must be freed with webp_free)
// width/height: output dimensions
EMSCRIPTEN_KEEPALIVE
int webp_decode_rgba(const uint8_t* webp_data, size_t data_size, uint8_t** out_rgba, int* width, int* height) {
  if (!webp_data || data_size == 0 || !out_rgba || !width || !height) return 0;
  int w = 0, h = 0;
  if (!WebPGetInfo(webp_data, data_size, &w, &h) || w <= 0 || h <= 0) return 0;
  uint8_t* rgba = WebPDecodeRGBA(webp_data, data_size, &w, &h);
  if (!rgba) return 0;
  *out_rgba = rgba;
  *width = w;
  *height = h;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
void webp_free(void* p) { WebPFree(p); }
