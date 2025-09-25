/* eslint-disable import/no-unresolved */
// @ts-ignore - .mjs is provided at runtime, declared in global.d.ts
import createModule from '../../lib/webp_decode.mjs';

class WebpDecoder {
    Module: any;

    static async create() {
        const instance = new WebpDecoder();
        instance.Module = await createModule({
            locateFile: (path: string) => {
                if (path.endsWith('.wasm')) {
                    return new URL(`../lib/${path}`, import.meta.url).toString();
                }
                return path;
            }
        });
        return instance;
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

export { WebpDecoder };
