declare module '*.mjs' {
    type Module = {
        HEAPU8: Uint8Array;
        HEAPU32: Uint32Array;
        _malloc: (size: number) => number;
        _free: (ptr: number) => void;
        _webp_free: (ptr: number) => void;
        _webp_encode_lossless_rgba: (
            input: number,
            width: number,
            height: number,
            stride: number,
            output: number,
            size: number
        ) => number;
        _webp_decode_rgba: (input: number, size: number, output: number, width: number, height: number) => number;
    };
    const mod: (options: { locateFile: (path: string) => string }) => Promise<Module>;
    export default mod;
}
