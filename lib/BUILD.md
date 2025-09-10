## Steps for building webp_encode wasm module

1. Install emsdk and activate it

2. Clone the webp and build wasm library:
```
git clone https://github.com/webmproject/libwebp.git
cd libwebp
emcmake cmake -S . -B build -DBUILD_SHARED_LIBS=OFF
emmake make -C build

// build the app
emcc -O3 webp_encode.c build/libwebp.a build/libsharpyuv.a \
  -sENVIRONMENT=node -sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS='["_webp_encode_rgba","_webp_encode_lossless_rgba","_webp_free","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["cwrap","HEAPU8","HEAPU32"]' \
  -o webp_encode.mjs
```

