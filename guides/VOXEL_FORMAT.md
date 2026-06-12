# Voxel Format Specification

This document specifies version 1.1 of the **voxel** format. It describes the on-disk structure written by `splat-transform` when the output path ends in `.voxel.json` and consumed by runtimes such as **supersplat-viewer** for collision detection and raycasts.

The format stores a solid/empty voxelization of a Gaussian splat scene as a sparse voxel octree (SVO) in the Laine–Karras node layout: a JSON header describing the grid, plus a compact binary file holding the tree.

For a step-by-step guide to generating this format (voxelization, exterior fill, nav carving, collision meshes), see [COLLISION.md](COLLISION.md).

## File set

A voxel dataset is a pair of files with a shared stem, plus an optional collision mesh:

```
scene.voxel.json       # header: bounds, resolution, tree statistics (this spec)
scene.voxel.bin        # binary octree data (nodes + leafData)
scene.collision.glb    # optional triangle mesh extracted from the voxels
```

- The header is always named `*.voxel.json`. Writers and readers derive the binary filename by replacing the `.voxel.json` suffix with `.voxel.bin`; the header does not reference it explicitly.
- The `.collision.glb` (written only when collision mesh generation is requested) is a standard glTF binary and is not described further here.

## `*.voxel.json`

```ts
interface VoxelMeta {
    version: string;             // file format version, decimal string, e.g. "1.1"
    asset?: {
        generator?: string;      // tool/version that produced the file, e.g. "splat-transform v2.5.2"
    };
    gridBounds: {                // AABB of the voxel grid, aligned to 4-voxel block boundaries
        min: number[];           // [x, y, z] — world position of the min corner of voxel (0, 0, 0)
        max: number[];           // [x, y, z]
    };
    sceneBounds: {               // AABB of the source Gaussians (informational)
        min: number[];
        max: number[];
    };
    voxelResolution: number;     // edge length of one voxel, world units
    leafSize: number;            // voxels per leaf-block edge; always 4
    treeDepth: number;           // subdivision levels from the root cube down to leaf blocks (>= 1)
    numInteriorNodes: number;    // number of interior nodes in `nodes`
    numMixedLeaves: number;      // number of mixed leaves in `nodes`
    nodeCount: number;           // total uint32 entries in the `nodes` array
    leafDataCount: number;       // total uint32 entries in the `leafData` array (= 2 * numMixedLeaves)
}
```

- `gridBounds` is the authoritative frame for the voxel data: voxel `(vx, vy, vz)` occupies the world-space box from `gridBounds.min + (vx, vy, vz) · voxelResolution` to `gridBounds.min + (vx+1, vy+1, vz+1) · voxelResolution`. Each axis spans a whole number of 4×4×4-voxel blocks: `nb = round((max − min) / (4 · voxelResolution))`.
- `sceneBounds` records the bounds of the source Gaussian scene (each Gaussian's position expanded by its rotated, scaled extents at the renderer's sigma cutoff). It is informational; depending on fill/carve/crop options, `gridBounds` may be larger or smaller than `sceneBounds`.
- `nodeCount` = `numInteriorNodes` + `numMixedLeaves` + the number of solid leaves (not stored separately).

### Coordinate space

All bounds are expressed in the PlayCanvas engine's coordinate frame — the frame the splat scene renders in. This differs from the source PLY's coordinate frame by the engine's standard splat import transform (a 180° rotation about the Z axis).

## `*.voxel.bin`

The binary file is two concatenated little-endian `uint32` arrays with no header or padding:

| Offset (bytes) | Length (bytes) | Contents |
| --- | --- | --- |
| `0` | `nodeCount * 4` | `nodes` array |
| `nodeCount * 4` | `leafDataCount * 4` | `leafData` array |

The total file size is exactly `(nodeCount + leafDataCount) * 4` bytes. A scene with no solid voxels has `nodeCount = 0`, `leafDataCount = 0` and an empty binary file.

## The octree

The tree subdivides a cube of `2^treeDepth` blocks per axis (each block being `leafSize`³ = 4×4×4 voxels), anchored at `gridBounds.min`. Where the per-axis block counts are not equal powers of two, the root cube extends beyond `gridBounds.max`; that excess region is always empty (such octants are simply absent from their parents' child masks).

Nodes at depth `treeDepth` are **leaf blocks** covering 4×4×4 voxels. `nodes[0]` is the root.

### Octant numbering

An interior node's eight octants are numbered by axis halves, X least significant:

```
oct = x | (y << 1) | (z << 2)    // x, y, z = 0 for the lower half, 1 for the upper half
```

This matches Morton order: the path from the root to a leaf block at block coordinates `(bx, by, bz)` consumes the coordinates' bits from most to least significant — at depth `d` (root = 0), `oct = ((bx >> s) & 1) | (((by >> s) & 1) << 1) | (((bz >> s) & 1) << 2)` where `s = treeDepth − 1 − d`.

### Node words

Each entry of `nodes` is one `uint32`, decoded as follows (test in order):

| Condition | Node kind | Meaning |
| --- | --- | --- |
| `word == 0xFF000000` | **Solid leaf** | The node's entire volume is solid. No children, no leaf data. May appear at any depth (a collapsed fully-solid subtree). |
| `(word >>> 24) == 0` | **Mixed leaf** | `word` is an index `i` into `leafData` pairs: the node's voxel mask is `leafData[2i]` (lo) and `leafData[2i + 1]` (hi). Only appears at depth `treeDepth`. |
| otherwise | **Interior** | `childMask = word >>> 24` (8 bits, one per octant); `firstChild = word & 0xFFFFFF` (index into `nodes`). |

The two leaf encodings are unambiguous:

- An interior node always has at least one child, so its `childMask` is never `0` — a zero top byte uniquely identifies a mixed leaf.
- Nodes are emitted in breadth-first order with children always following their parent, so an interior node's `firstChild` is never `0` — `0xFF000000` uniquely identifies a solid leaf.

### Children

An interior node's present children (set bits of `childMask`) are stored contiguously starting at `nodes[firstChild]`, ordered by ascending octant index. The child for octant `oct` (when `childMask & (1 << oct)` is set) is at:

```
nodes[firstChild + popcount(childMask & ((1 << oct) - 1))]
```

A clear bit in `childMask` means that octant's entire volume is empty; no node is stored for it.

### Mixed-leaf voxel masks

A mixed leaf's 4×4×4 = 64 voxels are stored as a 64-bit occupancy mask split across two `uint32`s. For the voxel at local coordinates `(lx, ly, lz)`, each in `[0, 4)`:

```
bit = lx + (ly << 2) + (lz << 4)
solid = bit < 32 ? (lo >>> bit) & 1 : (hi >>> (bit - 32)) & 1
```

`leafData[2i]` holds bits 0–31 (`lo`), `leafData[2i + 1]` holds bits 32–63 (`hi`). A set bit means the voxel is solid. A mixed leaf's mask is never all-zero or all-one — such blocks are encoded as an absent octant or a solid leaf instead.

### Query domain

Queries outside `gridBounds` are outside the format's domain, and the appropriate convention depends on the use case. Navigation/collision consumers (e.g. supersplat-viewer) treat out-of-grid space as **solid**; outputs produced with the nav options (exterior fill, floor fill, carve) rely on this — the writer crops away fully-solid blocks beyond the navigable region precisely because the runtime treats everything outside the grid as solid anyway.

## Limits

`firstChild` and the mixed-leaf `leafData` index are 24-bit values, capping the format at 16,777,216 node entries and 16,777,216 mixed leaves. The writer fails with an error rather than exceed either limit.

## Example `*.voxel.json`

A 32 × 14 × 32-block grid at 5 cm resolution (block size 0.2):

```json
{
    "version": "1.1",
    "asset": {
        "generator": "splat-transform v2.5.2"
    },
    "gridBounds": {
        "min": [-3.2, -0.2, -3.2],
        "max": [3.2, 2.6, 3.2]
    },
    "sceneBounds": {
        "min": [-3.13, -0.08, -3.07],
        "max": [3.11, 2.49, 3.08]
    },
    "voxelResolution": 0.05,
    "leafSize": 4,
    "treeDepth": 5,
    "numInteriorNodes": 1201,
    "numMixedLeaves": 5678,
    "nodeCount": 9232,
    "leafDataCount": 11356
}
```

## Versioning & compatibility

- `version` is a decimal `"major.minor"` string. Files conforming to this specification have `version: "1.1"`. Readers should reject files with a greater major version.
- **1.0 → 1.1**: version 1.0 files store bounds and voxel data in the source PLY coordinate frame; 1.1 voxelizes in the PlayCanvas engine frame (see [Coordinate space](#coordinate-space)). The binary layout is unchanged.
- The `asset` block was added after 1.1 shipped and may be absent from older 1.1 files.
- Unknown fields should be ignored, allowing minor additive evolution without a version bump.
