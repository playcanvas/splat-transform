# SplatTransform - 3D Gaussian Splat Converter

[![NPM Version](https://img.shields.io/npm/v/@playcanvas/splat-transform.svg)](https://www.npmjs.com/package/@playcanvas/splat-transform)
[![NPM Downloads](https://img.shields.io/npm/dw/@playcanvas/splat-transform)](https://npmtrends.com/@playcanvas/splat-transform)
[![License](https://img.shields.io/npm/l/@playcanvas/splat-transform.svg)](https://github.com/playcanvas/splat-transform/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white&color=black)](https://discord.gg/RSaMRzg)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat&logo=reddit&logoColor=white&color=black)](https://www.reddit.com/r/PlayCanvas)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white&color=black)](https://x.com/intent/follow?screen_name=playcanvas)

| [User Guide](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/) | [API Reference](https://api.playcanvas.com/splat-transform/) | [Blog](https://blog.playcanvas.com/) | [Forum](https://forum.playcanvas.com/) |

SplatTransform is an open source library and CLI tool for converting and editing Gaussian splats. It can:

📥 Read PLY, Compressed PLY, SOG, SPLAT, KSPLAT, SPZ and LCC formats  
📤 Write PLY, Compressed PLY, SOG, GLB, CSV, HTML Viewer, LOD and Voxel formats  
📊 Generate statistical summaries for data analysis  
🔗 Merge multiple splats  
🔄 Apply transformations to input splats  
🎛️ Filter out Gaussians or spherical harmonic bands  
🔀 Reorder splats for improved spatial locality  
⚙️ Procedurally generate splats using JavaScript generators

The library is platform-agnostic and can be used in both Node.js and browser environments.

## Installation

Install or update to the latest version:

```bash
npm install -g @playcanvas/splat-transform
```

For library usage, install as a dependency:

```bash
npm install @playcanvas/splat-transform
```

## CLI Usage

```bash
splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]
```

**Key points:**
- Input files become the working set; ACTIONS are applied in order
- The last file is the output; actions after it modify the final result
- Use `null` as output to discard file output

## Supported Formats

| Format | Input | Output | Description |
| ------ | ----- | ------ | ----------- |
| `.ply` | ✅ | ✅ | Standard PLY format |
| `.sog` | ✅ | ✅ | Bundled super-compressed format (recommended) |
| `meta.json` | ✅ | ✅ | Unbundled super-compressed format (accompanied by `.webp` textures) |
| `.compressed.ply` | ✅ | ✅ | Compressed PLY format (auto-detected and decompressed on read) |
| `.lcc` | ✅ | ❌ | LCC file format (XGRIDS) |
| `.ksplat` | ✅ | ❌ | Compressed splat format (mkkellogg format) |
| `.splat` | ✅ | ❌ | Compressed splat format (antimatter15 format) |
| `.spz` | ✅ | ❌ | Compressed splat format (Niantic format) |
| `.mjs` | ✅ | ❌ | Generate a scene using an mjs script (Beta) |
| `.glb` | ❌ | ✅ | Binary glTF with [KHR_gaussian_splatting](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) extension |
| `.csv` | ❌ | ✅ | Comma-separated values spreadsheet |
| `.html` | ❌ | ✅ | HTML viewer app (single-page or unbundled) based on SOG |
| `.voxel.json` | ❌ | ✅ | Sparse voxel octree for collision detection |
| `lod-meta.json` | ❌ | ✅ | Multi-level-of-detail SOG bundle (accompanied by per-LOD `.sog` chunks) |
| `null` | ❌ | ✅ | Discard output (useful with `--summary` for analysis-only runs) |

## Actions

Actions can be repeated and applied in any order:

```none
-t, --translate        <x,y,z>          Translate Gaussians by (x, y, z)
-r, --rotate           <x,y,z>          Rotate Gaussians by Euler angles (x, y, z), in degrees
-s, --scale            <factor>         Uniformly scale Gaussians by factor
-H, --filter-harmonics <0|1|2|3>        Remove spherical harmonic bands > n
-N, --filter-nan                        Remove Gaussians with NaN values and most Inf values;
                                          retains +Infinity in opacity and -Infinity in scale_*
-B, --filter-box       <x,y,z,X,Y,Z>    Remove Gaussians outside box (min, max corners)
-S, --filter-sphere    <x,y,z,radius>   Remove Gaussians outside sphere (center, radius)
-V, --filter-value     <name,cmp,value> Keep Gaussians where <name> <cmp> <value>
                                          cmp ∈ {lt,lte,gt,gte,eq,neq}
                                          opacity, scale_*, f_dc_* use transformed values
                                          (linear opacity 0-1, linear scale, linear color 0-1).
                                          Append _raw for raw PLY values (e.g. opacity_raw).
-F, --decimate         <n|n%>           Simplify to n Gaussians via progressive pairwise merging
                                          Use n% to keep a percentage of Gaussians
-G, --filter-floaters  [size,op,min]    Remove Gaussians not contributing to any solid voxel.
                                          Evaluates each Gaussian at occupied voxel centers.
                                          Default: size=0.05, opacity=0.1, min=0.004 (1/255)
-D, --filter-cluster   [res,op,min]     Keep only the connected cluster at --seed-pos.
                                          GPU-voxelizes at coarse resolution (res world units/voxel).
                                          Default: res=1.0, opacity=0.999, min=0.1
-p, --params           <key=val,...>    Pass parameters to .mjs generator script
-l, --lod              <n>              Specify the level of detail, n >= 0
-m, --summary                           Print per-column statistics to stdout
-M, --morton-order                      Reorder Gaussians by Morton code (Z-order curve)
```

## Global Options

```none
-h, --help                              Show this help and exit
-v, --version                           Show version and exit
-q, --quiet                             Suppress non-error output
    --verbose                           Show debug-level diagnostics
    --mem                               Show memory usage in progress output
    --no-tty                            Force non-interactive output (auto when stderr is not a TTY)
-w, --overwrite                         Overwrite output file if it exists
-i, --iterations       <n>              Iterations for SOG SH compression (more=better). Default: 10
-L, --list-gpus                         List available GPU adapters and exit
-g, --gpu              <n|cpu>          Select device for SOG compression: GPU adapter index | 'cpu'
-E, --viewer-settings  <settings.json>  HTML viewer settings JSON file
-U, --unbundled                         Generate unbundled HTML viewer with separate files
-O, --lod-select       <n,n,...>        Comma-separated LOD levels to read from LCC input
-C, --lod-chunk-count  <n>              Approximate number of Gaussians per LOD chunk in K. Default: 512
-X, --lod-chunk-extent <n>              Approximate size of an LOD chunk in world units (m). Default: 16
    --voxel-params     [size,opacity]   Voxel size and opacity threshold for .voxel.json. Default: 0.05,0.1
    --voxel-external-fill [size]        Fill exterior voxels by dilation from seed. Default size: 1.6
    --voxel-floor-fill [radius]         Fill each column upward from bottom until hitting solid (runs before carve).
                                          Optional radius (world units): only patch XZ areas surrounded by floor
                                          within 2*radius; large empty exterior areas are left alone.
                                          Default radius: 1.6
    --voxel-carve [h,r]                 Carve navigable space using capsule flood fill from seed.
                                          Default: height=1.6, radius=0.2
    --seed-pos         <x,y,z>          Seed position for voxel processing and --filter-cluster. Default: 0,0,0
-K, --collision-mesh                    Generate collision mesh (.collision.glb) from the voxel output.
```

> [!NOTE]
> See the [SuperSplat Viewer Settings Schema](https://github.com/playcanvas/supersplat-viewer?tab=readme-ov-file#settings-schema) for details on how to pass data to the `-E` option.

## Examples

### Basic Operations

```bash
# Simple format conversion
splat-transform input.ply output.csv

# Convert from .splat format
splat-transform input.splat output.ply

# Convert from .ksplat format
splat-transform input.ksplat output.ply

# Convert to compressed PLY
splat-transform input.ply output.compressed.ply

# Uncompress a compressed PLY back to standard PLY
# (compressed .ply is detected automatically on read)
splat-transform input.compressed.ply output.ply

# Convert to SOG bundled format
splat-transform input.ply output.sog

# Convert to SOG unbundled format
splat-transform input.ply output/meta.json

# Convert from SOG (bundled) back to PLY
splat-transform scene.sog restored.ply

# Convert from SOG (unbundled folder) back to PLY
splat-transform output/meta.json restored.ply

# Convert to standalone HTML viewer (bundled, single file)
splat-transform input.ply output.html

# Convert to unbundled HTML viewer (separate CSS, JS, and SOG files)
splat-transform -U input.ply output.html

# Convert to HTML viewer with custom settings
splat-transform -E settings.json input.ply output.html
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
splat-transform input.ply --filter-nan output.ply

# Filter by opacity values (keep only splats with opacity > 0.5)
splat-transform input.ply -V opacity,gt,0.5 output.ply

# Strip spherical harmonic bands higher than 2
splat-transform input.ply --filter-harmonics 2 output.ply

# Simplify to 50000 splats via progressive pairwise merging
splat-transform input.ply --decimate 50000 output.ply

# Simplify to 25% of original splat count
splat-transform input.ply -F 25% output.ply
```

### Advanced Usage

```bash
# Combine multiple files with different transforms
splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

# Apply final transformations to combined result
splat-transform input1.ply input2.ply output.ply -t 0,0,10 -s 0.5
```

### Statistical Summary

Generate per-column statistics for data analysis or test validation:

```bash
# Print summary, then write output
splat-transform input.ply --summary output.ply

# Print summary without writing a file (discard output)
splat-transform input.ply -m null

# Print summary before and after a transform
splat-transform input.ply --summary -s 0.5 --summary output.ply
```

The summary includes min, max, median, mean, stdDev, nanCount and infCount for each column in the data.

### Generators (Beta)

Generator scripts can be used to synthesize gaussian splat data. See [gen-grid.mjs](generators/gen-grid.mjs) for an example.

```bash
splat-transform gen-grid.mjs -p width=10,height=10,scale=10,color=0.1 scenes/grid.ply -w
```

### Voxel Format

The voxel format stores sparse voxel octree data for collision detection. It consists of two files: `.voxel.json` (metadata) and `.voxel.bin` (binary octree data).

```bash
# Generate voxel collision data from a splat file
splat-transform input.ply output.voxel.json

# Generate voxel data with custom resolution and opacity threshold
splat-transform --voxel-params 0.1,0.3 input.ply output.voxel.json

# Generate voxel data with exterior fill and carve
splat-transform --voxel-external-fill --voxel-carve input.ply output.voxel.json

# Generate voxel data with custom seed position and carve parameters
splat-transform --seed-pos 1,0,0 --voxel-carve 2.0,0.3 input.ply output.voxel.json
```

### Device Selection for SOG Compression

When compressing to SOG format, you can control which device (GPU or CPU) performs the compression:

```bash
# List available GPU adapters
splat-transform --list-gpus

# Let WebGPU automatically choose the best GPU (default behavior)
splat-transform input.ply output.sog

# Explicitly select a GPU adapter by index
splat-transform -g 0 input.ply output.sog  # Use first listed adapter
splat-transform -g 1 input.ply output.sog  # Use second listed adapter

# Use CPU for compression instead (much slower but always available)
splat-transform -g cpu input.ply output.sog
```

> [!NOTE]
> When `-g` is not specified, WebGPU automatically selects the best available GPU. Use `-L` to list available adapters with their indices and names. The order and availability of adapters depends on your system and GPU drivers. Use `-g <index>` to select a specific adapter, or `-g cpu` to force CPU computation.

> [!WARNING]
> CPU compression can be significantly slower than GPU compression (often 5-10x slower). Use CPU mode only if GPU drivers are unavailable or problematic.

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```

---

## Library Usage

SplatTransform exposes a programmatic API for reading, processing, and writing Gaussian splat data.

### Basic Import

```typescript
import {
    readFile,
    writeFile,
    getInputFormat,
    getOutputFormat,
    DataTable,
    processDataTable
} from '@playcanvas/splat-transform';
```

### Key Exports

| Export | Description |
| ------ | ----------- |
| `readFile` | Read splat data from various formats |
| `writeFile` | Write splat data to various formats |
| `getInputFormat` | Detect input format from filename |
| `getOutputFormat` | Detect output format from filename |
| `DataTable`, `Column` | Core data structures for splat data |
| `combine` | Merge multiple DataTables into one |
| `convertToSpace` | Convert a DataTable between coordinate spaces |
| `processDataTable` | Apply a sequence of processing actions |
| `computeSummary` | Generate statistical summary of data |
| `sortMortonOrder` | Sort indices by Morton code for spatial locality |
| `sortByVisibility` | Sort indices by visibility score for filtering |
| `writeVoxel` | Write sparse voxel octree files |

### File System Abstractions

The library uses abstract file system interfaces for maximum flexibility:

**Reading:**
- `UrlReadFileSystem` - Read from URLs (browser/Node.js)
- `MemoryReadFileSystem` - Read from in-memory buffers
- `ZipReadFileSystem` - Read from ZIP archives

**Writing:**
- `MemoryFileSystem` - Write to in-memory buffers
- `ZipFileSystem` - Write to ZIP archives

### Example: Reading and Processing

```typescript
import { Vec3 } from 'playcanvas';
import {
    readFile,
    writeFile,
    getInputFormat,
    getOutputFormat,
    processDataTable,
    UrlReadFileSystem,
    MemoryFileSystem
} from '@playcanvas/splat-transform';

// Read a PLY file from URL
const fileSystem = new UrlReadFileSystem();
const inputFormat = getInputFormat('scene.ply');

const dataTables = await readFile({
    filename: 'https://example.com/scene.ply',
    inputFormat,
    options: { iterations: 10 },
    params: [],
    fileSystem
});

// Apply transformations
const processed = processDataTable(dataTables[0], [
    { kind: 'scale', value: 0.5 },
    { kind: 'translate', value: new Vec3(0, 1, 0) },
    { kind: 'filterNaN' }
]);

// Write to in-memory buffer
const memFs = new MemoryFileSystem();
const outputFormat = getOutputFormat('output.ply', {});

await writeFile({
    filename: 'output.ply',
    outputFormat,
    dataTable: processed,
    options: {}
}, memFs);

// Get the output data
const outputBuffer = memFs.files.get('output.ply');
```

### Processing Actions

The `processDataTable` function accepts an array of actions:

```typescript
type ProcessAction =
    | { kind: 'translate'; value: Vec3 }
    | { kind: 'rotate'; value: Vec3 }       // Euler angles in degrees
    | { kind: 'scale'; value: number }
    | { kind: 'filterNaN' }
    | { kind: 'filterByValue'; columnName: string; comparator: 'lt'|'lte'|'gt'|'gte'|'eq'|'neq'; value: number }
    | { kind: 'filterBands'; value: 0|1|2|3 }
    | { kind: 'filterBox'; min: Vec3; max: Vec3 }
    | { kind: 'filterSphere'; center: Vec3; radius: number }
    | { kind: 'filterFloaters'; voxelResolution?: number; opacityCutoff?: number; minContribution?: number } // GPU
    | { kind: 'filterCluster'; voxelResolution?: number; seed?: Vec3; opacityCutoff?: number; minContribution?: number } // GPU
    | { kind: 'decimate'; count: number | null; percent: number | null }
    | { kind: 'param'; name: string; value: string }
    | { kind: 'lod'; value: number }
    | { kind: 'summary' }
    | { kind: 'mortonOrder' };
```

> [!NOTE]
> `filterFloaters` and `filterCluster` require a GPU device — pass `createDevice` via the `ProcessOptions` argument to `processDataTable`.

### Custom Logging

Configure the logger for your environment:

```typescript
import { logger } from '@playcanvas/splat-transform';

logger.setLogger({
    log: console.log,
    warn: console.warn,
    error: console.error,
    debug: console.debug,
    progress: (text) => process.stdout.write(text),
    output: console.log
});

logger.setQuiet(true); // Suppress non-error output
```
