## Steps for building webp_encode / webp_decode wasm modules

1. Install emsdk and activate it

2. Clone the webp and build wasm library:
```
git clone https://github.com/webmproject/libwebp.git
cd libwebp
emcmake cmake -S . -B build -DBUILD_SHARED_LIBS=OFF
emmake make -C build

# build the encoder (produces webp_encode.mjs + webp_encode.wasm)
emcc -O3 webp_encode.c build/libwebp.a build/libsharpyuv.a \
  -sENVIRONMENT=node -sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS='["_webp_encode_rgba","_webp_encode_lossless_rgba","_webp_free","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["cwrap","HEAPU8","HEAPU32"]' \
  -o webp_encode.mjs

# build the decoder (produces webp_decode.mjs + webp_decode.wasm)
emcc -O3 webp_decode.c build/libwebp.a build/libsharpyuv.a \
  -sENVIRONMENT=node -sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS='["_webp_decode_rgba","_webp_free","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["cwrap","HEAPU8","HEAPU32"]' \
  -o webp_decode.mjs
```

