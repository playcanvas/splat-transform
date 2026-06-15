# Agent Guidelines for splat-transform

This document contains rules, conventions, and best practices for AI agents working on the splat-transform codebase.

## Project Overview

splat-transform is a library and CLI tool for 3D Gaussian splat format conversion and transformation. It runs both in the browser (as a library) and on Node.js (as a CLI).

- **Language**: TypeScript (ES2022)
- **Module System**: ES Modules (`"type": "module"`)
- **Node Version**: >=18.0.0 (per `package.json` `engines`)
- **Build System**: Rollup
- **Testing**: Node.js built-in test runner (`node:test`)
- **Linting**: ESLint with `@playcanvas/eslint-config`
- **API Docs**: Typedoc
- **License**: MIT

## Code Style and Formatting

### Linting

- Run `npm run lint` before committing
- Only fix lint issues in code you are actively modifying
- Auto-fix available via `npm run lint:fix`

### ESLint Configuration

Base: `@playcanvas/eslint-config` with TypeScript overrides:

- **Relaxed rules**: `@typescript-eslint/ban-ts-comment`, `@typescript-eslint/no-explicit-any`, `@typescript-eslint/no-unused-vars` are off
- **JSDoc types relaxed**: `jsdoc/require-param-type`, `jsdoc/require-returns-type` off (TypeScript provides types)
- **Other relaxations**: `lines-between-class-members`, `no-await-in-loop`, `require-atomic-updates` off
- **Globals**: Both `node` and `browser` for `.ts` files; `node` only for `.mjs` files

### Naming Conventions

- Classes: PascalCase (`DataTable`, `Column`, `NodeFileSystem`)
- Functions: camelCase (`readFile`, `processDataTable`, `getInputFormat`)
- Types: PascalCase (`InputFormat`, `ReadFileOptions`, `TypedArray`)
- Constants: UPPER_SNAKE_CASE

### Import Order

1. Node built-in modules (`node:fs/promises`, `node:path`, etc.)
2. External packages (`playcanvas`)
3. Internal modules (relative paths)

### TypeScript

- Target: `es2022`
- `noImplicitAny: true`
- Generates declarations (`declaration: true`)
- Use TypeScript `type` and `interface` keywords for type definitions
- JSDoc comments are used for API documentation (Typedoc), not for type definitions

## File Organization

### Directory Structure

```
src/
├── lib/                    # Platform-agnostic library (browser + Node)
│   ├── index.ts            # Public API exports
│   ├── read.ts             # High-level read orchestration
│   ├── write.ts            # High-level write orchestration
│   ├── process.ts          # processDataTable and action types
│   ├── types.ts            # Options, Param types
│   ├── data-table/         # Core data model
│   │   ├── data-table.ts   # DataTable and Column classes
│   │   ├── combine.ts      # Merge multiple DataTables
│   │   ├── transform.ts    # Geometric transforms
│   │   ├── summary.ts      # Statistical summary
│   │   ├── morton-order.ts  # Morton code sorting
│   │   └── decimate.ts
│   ├── io/
│   │   ├── read/           # Read abstractions (FileSystem, streams)
│   │   └── write/          # Write abstractions (FileSystem, helpers)
│   ├── readers/            # Format-specific readers (one per file)
│   │   ├── read-ply.ts
│   │   ├── read-sog.ts     # (read-sog-v1.ts handles the legacy SOG layout)
│   │   ├── read-splat.ts
│   │   ├── read-ksplat.ts
│   │   ├── read-spz.ts
│   │   ├── read-lcc.ts
│   │   ├── read-lcc2.ts
│   │   └── read-mjs.ts
│   ├── writers/            # Format-specific writers (one per file)
│   │   ├── write-ply.ts
│   │   ├── write-compressed-ply.ts
│   │   ├── write-sog.ts
│   │   ├── write-csv.ts
│   │   ├── write-html.ts
│   │   ├── write-lod.ts
│   │   ├── write-spz.ts
│   │   ├── write-glb.ts
│   │   ├── write-image.ts
│   │   └── write-voxel.ts
│   ├── workers/            # Cross-platform worker pool (WorkerQueue, tasks)
│   ├── spatial/            # Spatial algorithms (k-means, kd-tree, b-tree, quantize-1d)
│   ├── voxel/              # Voxel generation (BVH, octree, GPU voxelization)
│   ├── mesh/               # Mesh generation (collision/marching cubes)
│   ├── render/             # GPU splat rasterizer (for image output)
│   ├── gpu/                # WebGPU compute (clustering)
│   └── utils/              # Logger, math, SH rotation, WebP codec
└── cli/                    # Node.js CLI (NOT platform-agnostic)
    ├── index.ts            # CLI entry, argument parsing
    ├── node-device.ts      # WebGPU device creation for Node
    └── node-file-system.ts # Node.js file system implementations
```

### Critical Architecture Rule: lib/ vs cli/ Separation

**`src/lib/` must remain platform-agnostic.** It must not import Node.js built-in modules (`node:fs`, `node:path`, etc.) or any Node-specific APIs. It runs in both browsers and Node.

**`src/cli/` is Node-specific.** It imports from `src/lib/` and adds Node.js file system access, WebGPU device creation, and CLI argument parsing.

Do not break this separation. If you need platform-specific behavior in `lib/`, use the file system abstraction interfaces (`ReadFileSystem`, `FileSystem`).

**Sole exception: `src/lib/workers/`.** The worker queue uses runtime-guarded dynamic imports (`import('node:worker_threads')`, `import('node:os')`) behind the same environment check the emscripten glue in `lib/webp.mjs` ships with; browser bundles never execute those branches. Do not add `node:` imports anywhere else in `lib/`, and do not convert these to static imports.

### Build Outputs

Four Rollup targets, all emitting to `dist/`:

| Output | Input | Format | External |
|--------|-------|--------|----------|
| `dist/worker.mjs` | `src/lib/workers/worker-entry.ts` | ESM | `node:*` (guarded dynamic imports) |
| `dist/index.mjs` | `src/lib/index.ts` | ESM | `playcanvas`, `node:*` |
| `dist/index.cjs` | `src/lib/index.ts` | CJS | `playcanvas`, `node:*` |
| `dist/cli.mjs` | `src/cli/index.ts` | ESM | `webgpu`, `node:*` |

`dist/worker.mjs` is the self-contained worker entry, shipped and exported as `@playcanvas/splat-transform/worker`. `WorkerQueue` spawns it from a URL: Node and bundlers that rewrite `new Worker(new URL('./worker.mjs', import.meta.url))` (Vite, webpack 5) resolve it automatically; other bundlers (e.g. plain Rollup, as SuperSplat uses) set `WorkerQueue.workerUrl` explicitly to a copied/vendored asset, mirroring how `WebPCodec.wasmUrl` is set. The `mark-worker-bundled` plugin flips the `workerBundled` flag true in the library/CLI builds; from source via tsx (dev, tests) it stays false and all worker tasks run inline on the calling thread. The worker bundle must stay lean: it must not pull in `DataTable` (whose `Transform` member drags in the playcanvas engine) - see `src/lib/spatial/quantize-1d-core.ts`.

The pool defaults to `min(4, cores - 1)` workers (`WorkerQueue.maxWorkers`); peak memory scales with worker count since each worker holds its own WebP WASM heap. The CLI exposes `--max-workers <n>` (0 = inline/serial) to trade speed for peak.

Type declarations go to `dist/lib/`. A post-build step copies `index.d.ts` to `index.d.cts` for CJS consumers.

### CLI Binary

`bin/cli.mjs` is the CLI entry point (`splat-transform` command). It imports `dist/cli.mjs`.

## Core Data Model

### DataTable and Column

The central data structure is `DataTable` -- a columnar store of typed arrays:

```typescript
class Column {
    name: string;
    data: TypedArray;  // Int8Array | Uint8Array | ... | Float32Array | Float64Array
}

class DataTable {
    columns: Column[];
    get numRows(): number;
    get numColumns(): number;

    addColumn(column: Column): void;
    getColumn(index: number): Column;            // by index
    getColumnByName(name: string): Column | null; // by name
    removeColumn(name: string): void;
    getRow(index: number, row?: Row, columns?: Column[]): Row;
    clone(options?: { rows?: Uint32Array | number[]; columns?: string[] }): DataTable;
}
```

Standard Gaussian splat columns:
- **Position**: `x`, `y`, `z` (Float32, always)
- **Rotation**: `rot_0`, `rot_1`, `rot_2`, `rot_3` (Float32, quaternion)
- **Scale**: `scale_0`, `scale_1`, `scale_2` (Float32, log scale)
- **Color**: `f_dc_0`, `f_dc_1`, `f_dc_2` (Float32, SH DC coefficients)
- **Opacity**: `opacity` (Float32, logit)
- **Spherical Harmonics**: `f_rest_0` through `f_rest_44`

### File System Abstractions

Read and write operations use abstract interfaces so the same code works in browsers and Node:

- **Read**: `ReadFileSystem` interface with implementations `UrlReadFileSystem` (browser), `NodeReadFileSystem` (CLI), `MemoryReadFileSystem`, `ZipReadFileSystem`
- **Write**: `FileSystem` interface with implementations `NodeFileSystem` (CLI), `MemoryFileSystem`, `ZipFileSystem`

### Supported Formats

**Input**: PLY, splat, KSplat, SOG, SPZ, LCC, LCC2, MJS

**Output**: PLY, compressed PLY, SOG, SOG-bundle, SPZ, GLB, CSV, HTML, HTML-bundle, LOD, voxel, image (WebP)

Each reader lives in `src/lib/readers/read-<format>.ts` and each writer in `src/lib/writers/write-<format>.ts`.

## Testing

### Framework

Uses Node.js built-in test runner with `describe`/`it`/`before` from `node:test` and `assert` from `node:assert`.

### Running Tests

```bash
npm test                 # Run all tests
npm run test:fixtures    # Regenerate test fixtures
```

### Test Structure

```
test/
├── *.test.mjs           # Test files
├── helpers/
│   ├── test-utils.mjs   # createMinimalTestData() and other helpers
│   └── summary-compare.mjs  # assertClose() for floating-point comparison
├── fixtures/
│   ├── splat/            # Test fixture files
│   └── create-fixtures.mjs   # Script to regenerate fixtures
└── static/              # Static test resources
```

### Writing Tests

```javascript
import { describe, it, before } from 'node:test';
import assert from 'node:assert';
import { processDataTable, DataTable, Column } from '../src/lib/index.js';
import { createMinimalTestData } from './helpers/test-utils.mjs';

describe('Feature Name', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should do something specific', async () => {
        const result = await processDataTable(testData.clone(), [/* actions */]);
        assert.strictEqual(result.numRows, testData.numRows);
    });
});
```

Key patterns:
- Clone test data before mutating (`testData.clone()`)
- Use `assertClose()` from helpers for floating-point comparisons
- Import from `../src/lib/index.js` (note `.js` extension for ESM resolution)
- Test files use `.test.mjs` extension

## Documentation

### JSDoc and Typedoc

Public API functions and classes must have JSDoc documentation including:
- Description
- `@param` with description (type comes from TypeScript)
- `@returns` with description
- `@throws` when applicable
- `@example` for common usage

Use `@ignore` for internal types that should not appear in generated docs.

Generate docs with `npm run docs`. Published at https://api.playcanvas.com/splat-transform/.

## NPM Scripts

| Script | Purpose |
|--------|---------|
| `npm run build` | Build library (ESM + CJS) and CLI |
| `npm run watch` | Watch mode build |
| `npm run lint` | Lint `src/` |
| `npm run lint:fix` | Auto-fix lint issues |
| `npm test` | Run all tests |
| `npm run test:fixtures` | Regenerate test fixtures |
| `npm run docs` | Generate Typedoc API docs |
| `npm run publint` | Check package publishing correctness |

## Dependencies

- **Runtime**: `webgpu` (for CLI GPU operations), `@adobe/spz` (SPZ format codec)
- **Peer**: `playcanvas` (>=2.0.0) -- used for Vec3, GraphicsDevice, etc.
- **Dev**: Rollup, TypeScript, ESLint, Typedoc, tsx (for running TypeScript tests)

## Common Patterns

### Adding a New File Format Reader

1. Create `src/lib/readers/read-<format>.ts`
2. Implement a function that takes a `ReadFileSystem` and returns a `DataTable`
3. Register the format in `src/lib/read.ts` (`getInputFormat` and `readFile`)
4. Export from `src/lib/index.ts`
5. Add CLI support in `src/cli/index.ts`
6. Add tests in `test/`

### Adding a New File Format Writer

1. Create `src/lib/writers/write-<format>.ts`
2. Implement a function that takes a `DataTable` + `FileSystem` and writes files
3. Register the format in `src/lib/write.ts` (`getOutputFormat` and `writeFile`)
4. Export from `src/lib/index.ts`
5. Add CLI support in `src/cli/index.ts`
6. Add tests in `test/`

### Processing Actions

Transformations are applied via `await processDataTable(dataTable, actions)` (it is async):

```typescript
const result = await processDataTable(dataTable, [
    { kind: 'translate', value: new Vec3(10, 0, 0) },
    { kind: 'scale', value: new Vec3(2, 2, 2) },
    { kind: 'filterNaN' }
]);
```

Action `kind` values (camelCase): `translate`, `rotate`, `scale`, `filterNaN`, `filterByValue`, `filterBands`, `filterBox`, `filterSphere`, `filterFloaters`, `filterCluster`, `param`, `lod`, `summary`, `mortonOrder`, `decimate`.

## Things to Avoid

- **Don't import Node APIs in `src/lib/`**: This breaks browser usage
- **Don't use `var`**: Use `const` or `let`
- **Don't mutate input DataTables without cloning**: Tests and callers expect non-destructive operations unless documented otherwise
- **Don't skip the FileSystem abstraction**: Direct file access belongs in `src/cli/` only
- **Don't commit `dist/` or `docs/`**: These are gitignored build artifacts

## AI Agent Guidelines

### When Making Changes

1. **Read existing code first**: Understand the lib/cli separation and DataTable patterns
2. **Follow existing style**: Match surrounding code
3. **Run `npm run lint`** before committing
4. **Run `npm test`** after changes to readers, writers, or data-table logic
5. **Update exports**: If adding public API, export from `src/lib/index.ts`
6. **Update JSDoc**: Public APIs need documentation for Typedoc

### When Adding Format Support

Follow the reader/writer patterns described above. Each format gets its own file. Register in the orchestration modules (`read.ts`/`write.ts`), export from `index.ts`, and add tests.

### Commit Messages

Use conventional commits:
- `feat: Add feature description`
- `fix: Bug fix description`
- `refactor: Code refactoring`
- `test: Test updates`
- `docs: Documentation update`

## Resources

- **NPM**: https://www.npmjs.com/package/@playcanvas/splat-transform
- **API Docs**: https://api.playcanvas.com/splat-transform/
- **User Manual**: https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/
- **GitHub**: https://github.com/playcanvas/splat-transform
