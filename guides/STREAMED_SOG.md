# Generating Streamed SOG Format

This guide explains how to turn a single PLY scene into the PlayCanvas **streamed SOG** format.

Streamed SOG splits the scene into SOG chunks at several levels of detail. The viewer can then stream in only the chunks and detail levels it needs for the current camera. This lets large scenes (tens of millions of Gaussians) load progressively and stay interactive.

For a description of the generated files, see the [Streamed SOG Format Specification](STREAMED_SOG_FORMAT.md).

## Overview

The pipeline has two stages:
1. Generate multiple versions of the scene at increasingly lower LOD
2. Combine the files into streamed SOG format

## Prerequisites

- `splat-transform` installed — see [Installation](../README.md#installation).
- A single full-resolution `.ply` scene.

## Step 1: Generate the LOD chain

Treat your full-resolution scene as **LOD 0**. Each subsequent level is produced by decimating the previous one to 50% — `--decimate 50%` keeps half the Gaussians:

```bash
splat-transform lod0.ply --decimate 50% lod1.ply
splat-transform lod1.ply --decimate 50% lod2.ply
splat-transform lod2.ply --decimate 50% lod3.ply
# …repeat for as many levels as you need
```
Keep halving until the coarsest level has around 1M Gaussians, which represents the lowest renderable LOD.

## Step 2: Combine into streamed SOG

Pass every level as an input, tagging each with `--lod <n>`, and write to a path ending in `lod-meta.json`. The `--lod` option applies to the **preceding** input file:

```bash
splat-transform \
    lod0.ply --lod 0 \
    lod1.ply --lod 1 \
    lod2.ply --lod 2 \
    lod3.ply --lod 3 \
    ./scene/lod-meta.json
```

This generates the `./scene/` folder of streamed SOG:

```
scene/
├── lod-meta.json        # spatial tree + per-chunk LOD index
├── 0_0/                 # LOD 0, chunk 0 — unbundled SOG (meta.json + .webp textures)
│   ├── meta.json
│   └── *.webp
├── 0_1/                 # LOD 0, chunk 1
├── 1_0/                 # LOD 1, chunk 0
└── …                    # one folder per {lod}_{chunk}
```

### Tuning the chunking

| Option | Default | Description |
| --- | --- | --- |
| `-C, --lod-chunk-count <n>` | `512` | Approximate Gaussians per chunk, in thousands (K). |
| `-X, --lod-chunk-extent <n>` | `16` | Approximate chunk size in world units (m). |

Smaller chunks mean finer-grained streaming (more, smaller files); larger chunks mean fewer requests but coarser granularity.

The `scene/` folder is now ready to serve to a viewer that supports streamed SOG format.
