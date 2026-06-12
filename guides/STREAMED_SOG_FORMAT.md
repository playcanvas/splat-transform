# Streamed SOG Format Specification

This document specifies version 1 of the PlayCanvas **streamed SOG** format. It describes the on-disk structure written by `splat-transform` when the output path ends in `lod-meta.json` and consumed by streaming viewers such as the PlayCanvas engine.

Streamed SOG splits a Gaussian splat scene into spatial chunks at multiple levels of detail (LOD). A viewer walks a spatial tree to decide which chunks and detail levels to load for the current camera, allowing very large scenes to load progressively and stay interactive.

For a step-by-step guide to generating this format, see [STREAMED_SOG.md](STREAMED_SOG.md). For the format of the individual chunks, see the [SOG format specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/).

## File set

A streamed SOG dataset is a directory containing a single index file plus one subdirectory per chunk:

```
scene/
├── lod-meta.json        # index: scene info + spatial tree (this spec)
├── 0_0/                 # LOD 0, chunk 0 — unbundled SOG (meta.json + .webp textures)
│   ├── meta.json
│   └── *.webp
├── 0_1/                 # LOD 0, chunk 1
├── 1_0/                 # LOD 1, chunk 0
├── …                    # one directory per {lod}_{chunk}
└── env/                 # optional environment splats — unbundled SOG
```

- The index file is always named `lod-meta.json`. Loaders identify the format by this filename.
- Each chunk is a standard **unbundled** SOG dataset (a `meta.json` plus WebP texture files). Bundled (single-archive) SOG chunks are not part of this format.
- The `{lod}_{chunk}/` directory naming is a convention of the writer. Readers must resolve chunk locations through the `filenames` array, not the naming pattern.
- All paths in `lod-meta.json` are relative to the directory containing `lod-meta.json`.

## `lod-meta.json`

```ts
interface LodMeta {
    version: 1;                  // file format version (integer)
    asset?: {
        generator?: string;      // tool/version that produced the file, e.g. "splat-transform v2.5.2"
    };
    count: number;               // total gaussians across all LOD levels (excludes environment)
    counts: number[];            // gaussians per LOD level; index = LOD level, length = lodLevels
    lodLevels: number;           // number of LOD levels
    environment?: string;        // relative path to the environment SOG's meta.json; omitted if none
    filenames: string[];         // relative paths to chunk SOG meta.json files, referenced by index
    tree: Node;                  // root of the spatial tree
}

interface Node {
    bound: {
        min: number[];           // AABB minimum [x, y, z]
        max: number[];           // AABB maximum [x, y, z]
    };
    children?: Node[];           // interior node: array of child nodes
    lods?: {
        [lodLevel: string]: {    // leaf node: map of LOD level → splat range
            file: number;        // index into filenames
            offset: number;      // index of the first splat within the chunk
            count: number;       // number of consecutive splats
        };
    };
}
```

## The spatial tree

`tree` is a binary spatial subdivision of the scene. Every node carries an axis-aligned bounding box and is either an **interior node** (has `children`) or a **leaf node** (has `lods`) — never both.

- A leaf's `bound` encloses the full extents of every Gaussian assigned to it — each Gaussian's position expanded by its rotated, scaled ellipsoid — not just the Gaussian centers.
- An interior node's `bound` is the union of its children's bounds.
- Bounds are expressed in the same coordinate frame as the splat positions stored in the chunk SOG files.

### LOD levels

LOD level `0` is the highest detail; higher levels are progressively coarser. A leaf's `lods` object is keyed by the decimal string form of the LOD level (`"0"` … `"lodLevels - 1"`). A missing key means the leaf has no splats at that level. All LOD levels of a leaf cover the same spatial region — a viewer selects exactly one level per leaf based on, for example, distance to camera.

### Chunk references

Each `lods` entry addresses a contiguous run of splats within one chunk:

- `file` is an index into the top-level `filenames` array.
- `offset` and `count` select splats `[offset, offset + count)` in the chunk's storage order (splat indices, not bytes). In the chunk's SOG textures, storage order is row-major pixel order, so splat `i` lives at pixel `(i % width, floor(i / width))`.

A chunk file's contents are exactly the concatenation of the leaf runs that reference it: the runs are non-overlapping and cover the chunk completely. Within each run, splats are sorted in Morton order for spatial locality; no ordering holds across run boundaries. A chunk only ever contains splats of a single LOD level.

## Environment

The optional `environment` field points to a standard unbundled SOG dataset containing splats that are not part of the LOD/chunk streaming scheme — typically far-field background such as sky. A viewer should load and render the environment unconditionally, independent of the spatial tree.

## Precision

Non-integer numbers in `lod-meta.json` are quantized to 7 significant digits (approximately 32-bit float precision).

## Example `lod-meta.json`

A two-level scene with an environment, split into one interior node with two leaves:

```json
{
    "version": 1,
    "asset": {
        "generator": "splat-transform v2.5.2"
    },
    "count": 1500000,
    "counts": [1000000, 500000],
    "lodLevels": 2,
    "environment": "env/meta.json",
    "filenames": [
        "0_0/meta.json",
        "1_0/meta.json"
    ],
    "tree": {
        "bound": { "min": [-10, 0, -10], "max": [10, 5, 10] },
        "children": [
            {
                "bound": { "min": [-10, 0, -10], "max": [0.5, 5, 10] },
                "lods": {
                    "0": { "file": 0, "offset": 0, "count": 600000 },
                    "1": { "file": 1, "offset": 0, "count": 300000 }
                }
            },
            {
                "bound": { "min": [0.5, 0, -10], "max": [10, 4.5, 10] },
                "lods": {
                    "0": { "file": 0, "offset": 600000, "count": 400000 },
                    "1": { "file": 1, "offset": 300000, "count": 200000 }
                }
            }
        ]
    }
}
```

## Versioning & compatibility

- Files conforming to this specification have `version: 1`. Readers should reject files with a greater major version.
- Files produced before the format was versioned omit `version`, `asset`, `count` and `counts`, and may contain `"environment": null`. Readers that wish to support them should treat a missing `version` as pre-release and `environment: null` as no environment.
- Unknown fields should be ignored, allowing minor additive evolution without a version bump.
