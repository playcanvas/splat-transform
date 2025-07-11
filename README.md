# Splat Transform - 3D Gaussian Splat Converter

Splat Transform is an open source CLI tool for reading gaussian splat PLY files and writing them to PLY, Compressed PLY, CSV, and SOGS format.

Multiple files may be combined and transformed before being written to the output.

## Installation

First install the package globally:

```bash
npm install -g @playcanvas/splat-transform
```

## Usage

```bash
splat-transform [GLOBAL]  <input.{ply|splat}> [ACTIONS]  ...  <output.{ply|compressed.ply|meta.json|csv}> [ACTIONS]
```

**Key points:**
- Every time an `*.ply*` appears, it becomes the current working set; the following ACTIONS are applied in the order listed
- The last file on the command line is treated as the output; anything after it is interpreted as actions that modify the final result

## Supported Formats

**Input:**
- `.ply` - Standard PLY format
- `.splat` - Binary splat format (antimatter15 format)

**Output:**
- `.ply` - Standard PLY format
- `.compressed.ply` - Compressed PLY format
- `meta.json` - SOGS format (JSON + WebP images)
- `.csv` - Comma-separated values

## Actions

Actions can be repeated and applied in any order:

```bash
-t, --translate  x,y,z                  Translate splats by (x, y, z)
-r, --rotate     x,y,z                  Rotate splats by Euler angles (deg)
-s, --scale      x                      Uniformly scale splats by factor x
-n, --filterNaN                         Remove any Gaussian containing NaN/Inf
-c, --filterByValue name,cmp,value      Keep splats where <name> <cmp> <value>
                                        cmp ∈ {lt,lte,gt,gte,eq,neq}
-b, --filterBands  {0|1|2|3}            Strip spherical-harmonic bands > N
```

## Global Options

```bash
-w, --overwrite                         Overwrite output file if it already exists
-h, --help                              Show help and exit
-v, --version                           Show version and exit
```

## Examples

### Basic Operations

```bash
# Simple format conversion
splat-transform input.ply output.csv

# Convert from .splat format
splat-transform input.splat output.ply

# Convert to compressed PLY
splat-transform input.ply output.compressed.ply

# Convert to SOGS format
splat-transform input.ply output/meta.json
```

### Transformations

```bash
# Scale and translate
splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny_scaled.ply

# Rotate by 90 degrees around Y axis
splat-transform input.ply -r 0,90,0 output.ply

# Chain multiple transformations
splat-transform input.ply -s 2 -t 1,0,0 -r 0,0,45 output.ply
```

### Filtering

```bash
# Remove entries containing NaN and Inf
splat-transform input.ply --filterNaN output.ply

# Filter by opacity values (keep only splats with opacity > 0.5)
splat-transform input.ply -c opacity,gt,0.5 output.ply

# Strip spherical harmonic bands higher than 2
splat-transform input.ply --filterBands 2 output.ply
```

### Advanced Usage

```bash
# Combine multiple files with different transforms
splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

# Apply final transformations to combined result
splat-transform input1.ply input2.ply output.ply -t 0,0,10 -s 0.5
```

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```
