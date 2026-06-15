import createModule from '../../../lib/webp.mjs';

class WebPCodec {
    /**
     * URL to the webp.wasm file. Set this before any SOG read/write operations
     * in browser environments where the default path resolution doesn't work.
     * Must be set before the first `create()` call: the compiled module is
     * cached, so later changes have no effect.
     *
     * @example
     * import { WebPCodec } from '@playcanvas/splat-transform';
     * import wasmUrl from '@playcanvas/splat-transform/lib/webp.wasm?url';
     * WebPCodec.wasmUrl = wasmUrl;
     */
    static wasmUrl: string | null = null;

    private static modulePromise: Promise<any> | null = null;

    Module: any;

    /**
     * The effective webp.wasm location to hand to worker threads (which can't
     * resolve it from their own module URL). Returns `wasmUrl` verbatim when
     * set - exactly the value `locateFile` uses, so a Windows file path or a
     * URL both pass through unchanged - otherwise the default resolution
     * relative to this module.
     *
     * @returns The configured `wasmUrl`, or the default webp.wasm URL.
     * @ignore
     */
    static resolveWasmUrl() {
        return WebPCodec.wasmUrl ?? new URL('../lib/webp.wasm', import.meta.url).toString();
    }

    static async create() {
        // Compile/instantiate the wasm module once and share it across all
        // instances; per-call instantiation pays a fresh Emscripten heap each
        // time (readers like readLcc2 call create() once per chunk). Memoize
        // the promise so concurrent first calls share a single instantiation,
        // but reset on rejection so a failed load (e.g. wasmUrl set late in a
        // browser) can be retried.
        if (!WebPCodec.modulePromise) {
            const promise = createModule({
                locateFile: (path: string) => {
                    if (path.endsWith('.wasm') && WebPCodec.wasmUrl) {
                        return WebPCodec.wasmUrl;
                    }
                    return new URL(`../lib/${path}`, import.meta.url).toString();
                }
            });
            promise.catch(() => {
                if (WebPCodec.modulePromise === promise) {
                    WebPCodec.modulePromise = null;
                }
            });
            WebPCodec.modulePromise = promise;
        }
        const instance = new WebPCodec();
        instance.Module = await WebPCodec.modulePromise;
        return instance;
    }

    encodeLosslessRGBA(rgba: Uint8Array, width: number, height: number, stride = width * 4) {
        const { Module } = this;

        const inPtr = Module._malloc(rgba.length);
        const outPtrPtr = Module._malloc(4);
        const outSizePtr = Module._malloc(4);

        Module.HEAPU8.set(rgba, inPtr);

        const ok = Module._webp_encode_lossless_rgba(inPtr, width, height, stride, outPtrPtr, outSizePtr);
        if (!ok) {
            throw new Error('WebP lossless encode failed');
        }

        const outPtr = Module.HEAPU32[outPtrPtr >> 2];
        const outSize = Module.HEAPU32[outSizePtr >> 2];
        const bytes = Module.HEAPU8.slice(outPtr, outPtr + outSize);

        Module._webp_free(outPtr);
        Module._free(inPtr); Module._free(outPtrPtr); Module._free(outSizePtr);

        return bytes;
    }

    decodeRGBA(webp: Uint8Array): { rgba: Uint8Array, width: number, height: number } {
        const { Module } = this;

        const input = webp;

        const inPtr = Module._malloc(input.length);
        const outPtrPtr = Module._malloc(4);
        const widthPtr = Module._malloc(4);
        const heightPtr = Module._malloc(4);

        Module.HEAPU8.set(input, inPtr);

        const ok = Module._webp_decode_rgba(inPtr, input.length, outPtrPtr, widthPtr, heightPtr);
        if (!ok) {
            Module._free(inPtr); Module._free(outPtrPtr); Module._free(widthPtr); Module._free(heightPtr);
            throw new Error('WebP decode failed');
        }

        const outPtr = Module.HEAPU32[outPtrPtr >> 2];
        const width = Module.HEAPU32[widthPtr >> 2];
        const height = Module.HEAPU32[heightPtr >> 2];
        const size = width * height * 4;
        const bytes = Module.HEAPU8.slice(outPtr, outPtr + size);

        Module._webp_free(outPtr);
        Module._free(inPtr); Module._free(outPtrPtr); Module._free(widthPtr); Module._free(heightPtr);

        return { rgba: bytes, width, height };
    }
}

export { WebPCodec };
