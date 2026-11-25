# Rendering: state and remaining work

Notes for the image renderer (`render/`, `gpu/gpu-scene-rasterizer.ts`, `gpu/gpu-splat-rasterizer.ts`, `writers/write-image.ts`). Written 2026-09-18 at the end of the speed round and updated 2026-09-19 as the depth sort, slice accumulation, encode, source-native upload, streamed tiers and equirect motion blur landed.

## How a frame renders today

The image writer is chunk-native: `writeSource` hands it the `ChunkSource` and a pool, nothing is materialized, and the camera is moved into the source's space instead of baking the scene into the camera's. `SceneRenderer` picks one of three tiers at `upload()`, the first that fits the device's storage-binding limit and the caller's budget (`--resident-budget`), falling to the next if the device refuses the allocation:

- **Resident.** The source's LOD 0 layers are uploaded once, as stored (position 12 B, geometric 32 B, colour 12 B plus SH per gaussian): each chunk is a raw copy into three layer buffers, no CPU de-interleave. A render sends only the camera. Per pass the GPU keys every gaussian by view depth (`shaders/depth-keys.ts`: camera z, radius squared for equirect, the mean of the two shutter poses under motion blur; gaussians the project shader will reject get the largest key), radix sorts all 32 key bits with an identity payload (stable, so ties break by row exactly as the CPU sort does), and the project shader gathers attributes in that order. Then: two-level block scan of tile coverage → read back the block prefixes and total → pair buffers sized exactly → the sorted list cut into ranges at 2048-splat blocks so each radix sort stays inside the pair budget → emit, sort by tile, find tile boundaries, rasterize → finalize, read back.
- **Streamed, GPU sort.** Only the position layer stays resident. Per pass the GPU sorts as above and the sorted order comes back to the CPU with the visible count; the visible gaussians' records are then gathered from the source in that order, a pool chunk at a time (`source.read({ indices })`, which the PLY reader coalesces into file-order reads internally), uploaded into range-sized layer buffers with an identity order, and rasterized by the same machinery. The next range is gathered while the current one rasterizes. Attribute memory is bounded by the range, so any scene whose positions fit a binding renders.
- **Streamed, CPU sort.** Past that limit the positions live in RAM, the cull and sort run on the CPU (`sortCandidatesByDepth`, with the same tests as the key pass), and the ranges stream as above. Only the range buffers are on the GPU.

A motion-blurred frame renders its shutter slices one after another, each slice's composited colour added into an f32 accumulator on the GPU (`shaders/accumulate.ts`) and the mean packed on the last slice, so the frame quantizes once and reads back once. Images over 4096² render as a grid of groups, slices inside groups, so the accumulator stays group-sized; each group repeats the depth sort and, when streaming, the gathers.

Motion blur is exact per gaussian: each pass integrates the truncated gaussian along its screen-space path between the shutter-open and shutter-close poses (closed form in `erf`), and a frame averages `--motion-samples` such shutter slices. Both projections support it; under equirect the displacement takes the short way round the ±π seam and the midpoint is wrapped back into the image, matching the rasterizer's per-pixel wrap, and gaussians order by their mean squared distance. Camera tracks (`--camera-track`) keep the scene on the GPU across frames and encode each frame on a worker thread (`runEncodeWebp`, shared with the SOG writer) while the next renders, with at most four frames waiting for a thread.

The chunked `GpuSplatRasterizer` and `renderRasterPass` remain as the reference path (CPU cull and sort from a `DataTable`, 8192-splat chunks) and as the LOD error pass's renderer. `test/render-resident.test.mjs` renders a 1/64-grid scene through all three tiers and the reference and requires byte-exact agreement for static, defocus, motion blur, equirect and equirect blur, with gaussians behind the camera, on the near plane and with a NaN position.

## Where the time goes

Bowes Museum, 17.5M splats, SH0, 1080p, M4 Max. Frame times from a script driving `SceneRenderer` directly; the CLI adds readback handoff and encode overlap.

|                         | resident | streamed, GPU sort | streamed, CPU sort |
| ----------------------- | -------- | ------------------ | ------------------ |
| GPU bytes               | 2.24 GB  | 0.61 GB            | 0.12 GB            |
| upload from the PLY     | 0.46 s   | 0.19 s             | 0.22 s             |
| static frame            | 39 ms    | 4.8 s              | 5.0 s              |
| blurred frame, 4 slices | 175 ms   | 19.2 s             | 20.3 s             |

The streamed tiers agree with the resident one: byte-identical with the GPU sort, near-tie swaps only with the CPU sort (89–91 dB). Their time is the source: each pass gathers every visible gaussian's record, about 4 GB for this scene, so they are the fallback for scenes that do not fit, not a fast path.

For the round as a whole, the 101-frame blurred 1080p track went from 112 s (CPU sort, CPU accumulation, inline encode, materialized table) to 41 s (GPU sort), 22.5 s (GPU accumulation, threaded encode) and 20.7 s (layer-native upload from the source), with peak CPU memory down from 1.5 GB to 0.5 GB. A pass on the resident tier is now under 45 ms of GPU time end to end.

## Limits today

- Each resident buffer must fit one storage binding (4 GB on Apple silicon, 128 MiB on the WebGPU baseline). The largest per-gaussian buffers are the projection records (48 B) and the colour layer (12 B for SH0, 192 B for SH3), so the resident tier caps SH0 near 89M splats and SH3 near 22M on Apple silicon. The GPU-sort streamed tier needs 12 B per gaussian in one binding (357M on Apple silicon); the CPU-sort tier has no per-scene GPU limit.
- Only LOD 0 renders. A multi-LOD source should be viewed through `selectLod` upstream, which the CLI does for single-scene outputs.
- Equirect groups are never split along X, so on a 128 MiB baseline binding the panorama width caps near 4096.
- The reference path's CPU accumulation of 8-bit slices carries a +0.125 level tie bias (an integer mean lands on a half level a quarter of the time and rounds up); the GPU accumulation has no such tie.

## Remaining work

- **Splitting the colour layer per channel** would triple the SH3 resident cap. Not needed for any scene seen so far.
- **Streamed tiers for very large images**: the ranges are gathered once per group, so a multi-group image of a scene too large to be resident gathers the scene once per group.
- **Output formats** (raw, PNG, video), further **camera formats** (COLMAP, transforms.json, nerfstudio camera paths) and **shutter in degrees** are still open from the round's brief.
