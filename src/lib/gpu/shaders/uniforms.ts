import {
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    UniformFormat
} from 'playcanvas';

/**
 * Shared WGSL declaration of the `struct Uniforms` block consumed by
 * every per-render compute shader (project, prefix-sum, emit-pairs,
 * init-tile-offsets, find-boundaries, rasterize-binned, finalize). The
 * matching JS layout produced by {@link uniformFormatEntries} sits
 * immediately below — when one changes the other must change in
 * lock-step. The pad fields keep f32 triples 16-byte aligned per
 * WGSL's uniform storage rules.
 *
 * Included into each consuming shader via the engine's WGSL preprocessor:
 * `#include "uniformsStruct"` at the top of the shader source replaces
 * the directive with the full struct declaration. See
 * `playcanvas/src/platform/graphics/shader.js` for the preprocessor
 * wiring (the `cincludes` Map on the `Shader` constructor's definition).
 */
const uniformsStruct = /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    focusDistance: f32, apertureScale: f32, _p5: f32, _p6: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}
`;

/**
 * Build the {@link UniformFormat} entries describing the WGSL `struct
 * Uniforms` block above, in declaration order. The PlayCanvas
 * UniformBuffer machinery expects a JS-side description that matches the
 * shader's uniform layout exactly; this function is the single source of
 * truth for that JS-side description and is consumed by every compute
 * shader that binds the shared Uniforms struct.
 *
 * @returns Array of UniformFormat entries in declaration order.
 */
const uniformFormatEntries = (): UniformFormat[] => [
    new UniformFormat('rightX', UNIFORMTYPE_FLOAT),
    new UniformFormat('rightY', UNIFORMTYPE_FLOAT),
    new UniformFormat('rightZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p0', UNIFORMTYPE_FLOAT),
    new UniformFormat('downX', UNIFORMTYPE_FLOAT),
    new UniformFormat('downY', UNIFORMTYPE_FLOAT),
    new UniformFormat('downZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p1', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardX', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardY', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p2', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeX', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeY', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p3', UNIFORMTYPE_FLOAT),
    new UniformFormat('focalX', UNIFORMTYPE_FLOAT),
    new UniformFormat('focalY', UNIFORMTYPE_FLOAT),
    new UniformFormat('near', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p4', UNIFORMTYPE_FLOAT),
    new UniformFormat('focusDistance', UNIFORMTYPE_FLOAT),
    new UniformFormat('apertureScale', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p5', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p6', UNIFORMTYPE_FLOAT),
    new UniformFormat('imageWidth', UNIFORMTYPE_UINT),
    new UniformFormat('imageHeight', UNIFORMTYPE_UINT),
    new UniformFormat('splatStride', UNIFORMTYPE_UINT),
    new UniformFormat('chunkSize', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMinX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMinY', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMaxX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMaxY', UNIFORMTYPE_UINT),
    new UniformFormat('groupTilesX', UNIFORMTYPE_UINT),
    new UniformFormat('groupTilesY', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelOriginX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelOriginY', UNIFORMTYPE_UINT),
    new UniformFormat('bgR', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgG', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgB', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgA', UNIFORMTYPE_FLOAT)
];

export { uniformsStruct, uniformFormatEntries };
