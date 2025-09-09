#include <stdint.h>
#include <stddef.h>
#include <emscripten/emscripten.h>
#include "src/webp/encode.h"

EMSCRIPTEN_KEEPALIVE
int webp_encode_rgba(const uint8_t* rgba, int width, int height, int stride,
                     float quality, uint8_t** out_buf, size_t* out_size) {
  if (!rgba || width <= 0 || height <= 0 || stride <= 0 || !out_buf || !out_size) return 0;
  uint8_t* out = NULL;
  size_t sz = WebPEncodeRGBA(rgba, width, height, stride, quality, &out);
  if (sz == 0 || !out) return 0;
  *out_buf = out;
  *out_size = sz;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int webp_encode_lossless_rgba(const uint8_t* rgba, int width, int height, int stride,
                              uint8_t** out_buf, size_t* out_size) {
  if (!rgba || width <= 0 || height <= 0 || stride <= 0 || !out_buf || !out_size) return 0;
  uint8_t* out = NULL;
  size_t sz = WebPEncodeLosslessRGBA(rgba, width, height, stride, &out);
  if (sz == 0 || !out) return 0;
  *out_buf = out;
  *out_size = sz;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
void webp_free(void* p) { WebPFree(p); }
