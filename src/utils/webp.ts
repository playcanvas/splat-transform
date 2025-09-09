import createModule from '../../lib/webp_encode.mjs';

class WebpEncoder {
    Module: any;

    static async create() {
        const instance = new WebpEncoder();
        instance.Module = await createModule({
            locateFile: (path: string) => {
                if (path.endsWith('.wasm')) {
                    return new URL('../lib/' + path, import.meta.url).toString();
                }
                return path;
            }
        });
        return instance;
    }

    async encodeLosslessRGBA(rgba: Uint8Array, width: number, height: number, stride = width * 4) {
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

        return Buffer.from(bytes);
    }
}

export { WebpEncoder };
