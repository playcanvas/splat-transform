# Collision Mesh Guide

This guide explains how to generate collision data from a Gaussian splat scene using `splat-transform`. The output is a sparse voxel octree (`.voxel.json` + `.voxel.bin`) and an optional mesh (`.collision.glb`) suitable for runtime collision detection.

## Overview

Two outputs are produced from the same voxelization pass:

- **`.voxel.json` / `.voxel.bin`** — sparse voxel octree for raycasts and broad-phase collision queries.
- **`.collision.glb`** — triangulated mesh built from the voxel grid (only when `-K` / `--collision-mesh` is passed).

A typical pipeline runs four stages, with the latter two being optional depending on the scene type:

```
input splat ──► filter-cluster ──► voxelize ──► fill ──► carve ──► [collision mesh]
```

<!-- TODO: pipeline diagram -->

## Step 1: Isolating the scene with `--filter-cluster`

Splats often contain stray floaters and disconnected geometry far from the scene of interest. `--filter-cluster` GPU-voxelizes the input at a coarse resolution and keeps only the connected component containing `--seed-pos`.

```bash
splat-transform input.ply --filter-cluster --seed-pos 0,1,0 output.voxel.json
```

Run this *before* fine-grained voxelization. Without it, fill and carve passes can leak through gaps or seed the wrong cluster.

![filter-cluster: before/after showing the central cluster selected](images/filter-cluster.png)
<!-- TODO: illustration -->

### Tuning

```none
-D, --filter-cluster [res,op,min]
```

- `res` — voxel size in world units used for clustering (default `1.0`). Coarser = faster, more permissive about gaps.
- `op` — opacity threshold (default `0.999`).
- `min` — minimum contribution per voxel (default `0.1`).

## Step 2: Voxelization (`--voxel-params`)

Controls the grid that all subsequent passes operate on.

```none
--voxel-params [size,opacity]   Default: 0.05,0.1
```

- `size` — voxel edge length in world units. Smaller = higher fidelity, larger files, slower fills.
- `opacity` — minimum splat opacity required to mark a voxel solid.

## Step 3: Sealing the shell

After voxelization, the surface is typically a thin shell with holes. Filling closes those holes so carve has a watertight volume to flood. Choose the fill that matches your scene type — they are not normally combined.

### Interior scenes — `--voxel-external-fill`

For room scans, where you want a closed interior volume that carve can flood. The pass:

1. Dilates the solid grid by `[size]` world units (converted internally to a voxel half-extent) to bridge small holes in the walls.
2. Flood-fills empty space inward from the bounding-box boundary — every voxel reachable from outside is marked as exterior.
3. Marks the exterior region as solid in the output, leaving only the enclosed interior as empty space for carve.

`--seed-pos` is used as a sanity check: if the seed ends up reachable from outside (i.e. the volume isn't actually enclosed at the seed), the fill is skipped and the original grid is returned.

```none
--voxel-external-fill [size]    Default size: 1.6
```

The optional `size` (world units) controls the dilation distance used to seal small wall gaps — increase if walls have noisy holes the fill leaks through.

![external-fill: cross-section of a room before/after, showing void around the room marked solid](images/external-fill.png)
<!-- TODO: illustration -->

### Exterior scenes — `--voxel-floor-fill`

For outdoor scans, terrain, or objects on a ground plane. The pass walks each XZ column upward from the bottom of the bounding box until it hits a solid voxel, marking everything below as solid. This produces a ground volume even when the scan only captured the surface.

```none
--voxel-floor-fill [radius]     Default radius: 1.6
```

The optional `radius` (world units) restricts patching to XZ areas surrounded by floor within `2*radius` — large empty exterior areas are left alone, so this won't accidentally fill the sky.

![floor-fill: cross-section of a terrain before/after, showing solid mass below the surface](images/floor-fill.png)
<!-- TODO: illustration -->

### Choosing

| Scene type             | Fill                     |
| ---------------------- | ------------------------ |
| Indoor room scan       | `--voxel-external-fill`  |
| Outdoor terrain/object | `--voxel-floor-fill`     |
| Single object in space | (skip both)              |

## Step 4: Carving navigable space (`--voxel-carve`)

Flood-fills a capsule volume from `--seed-pos`, marking voxels the capsule can reach as *navigable*. This produces the actual walkable region used at runtime.

```none
--voxel-carve [h,r]             Default: height=1.6, radius=0.2
```

- `h` — capsule height (world units), roughly the agent height.
- `r` — capsule radius, roughly the agent radius.

The capsule must fit at the seed position. If carve produces no output, the seed is likely inside solid geometry or the capsule is too large to fit.

![carve: room before/after the capsule flood](images/carve.png)
<!-- TODO: illustration -->

## Step 5: Generating the collision mesh (`-K` / `--collision-mesh`)

```none
-K, --collision-mesh [smooth|faces]   Default: smooth
```

Two output shapes:

### `smooth` (default)

A smoothed mesh fitted to the voxel surface. Lower triangle count, more natural contours, suitable for character collision.

![smooth collision mesh output](images/mesh-smooth.png)
<!-- TODO: illustration -->

### `faces`

A watertight mesh built from the exposed voxel faces — every face is axis-aligned to the voxel grid. Higher triangle count but exactly matches the voxel volume; useful when collision must agree with raycasts against the voxel data.

![voxel-face collision mesh output](images/mesh-faces.png)
<!-- TODO: illustration -->

## Full examples

### Interior room scan

```bash
splat-transform room.ply \
    --filter-cluster --seed-pos 0,1,0 \
    --voxel-external-fill --voxel-carve \
    -K room.voxel.json
```

### Exterior terrain

```bash
splat-transform terrain.ply \
    --filter-cluster --seed-pos 0,0,0 \
    --voxel-floor-fill \
    -K terrain.voxel.json
```

### High-fidelity voxel-face mesh

```bash
splat-transform input.ply \
    --voxel-params 0.025,0.1 \
    -K faces output.voxel.json
```

## Troubleshooting

- **Carve produces nothing.** `--seed-pos` is inside solid geometry, or the capsule (`h,r`) doesn't fit at the seed. Move the seed or shrink the capsule.
- **Fill leaks through walls.** Increase `--voxel-external-fill` size, or lower the voxel `opacity` so thin walls are marked solid.
- **Carve leaks into adjacent rooms.** Walls are too thin or have gaps. Lower voxel `size` for higher resolution, or increase fill size.
- **Collision mesh is too dense.** Use `-K smooth` (default), or coarsen `--voxel-params` size.
- **`--filter-cluster` selects the wrong cluster.** Move `--seed-pos` into the cluster you want, or increase its `res` to bridge intentional gaps.
