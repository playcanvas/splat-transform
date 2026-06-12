import { Column, combine, DataTable } from '../data-table';
import { readSog } from './read-sog';
import { readSpz } from './read-spz';
import { basename, dirname, join, readFile, ReadFileSystem, ReadSource, ZipReadFileSystem } from '../io/read';
import { Options } from '../types';
import { logger, Transform } from '../utils';

// Bounded concurrency for chunk decoding. SOG/SPZ decoding is heavier than
// LCC v1's range reads (WebP decode / WASM calls), so we stay conservative.
const LOAD_CONCURRENCY = 4;

// LCC2 coordinate transform (matches readLcc).
const LCC2_TRANSFORM = () => new Transform().fromEulers(90, 0, 180);

// --- Raw meta.lcc2 shapes (pre-normalization, mixed new/old protocol) -------

// Raw octree node. `child` may be an array or a string-keyed object map.
// Old protocol uses files / child_num / datatype; new protocol uses
// splatFiles / childNum / dataType.
type RawLcc2Node = {
    // Node data: '3dgs' points to this node's main chunk file index; env
    // points to the optional environment chunk.
    data?: {
        '3dgs'?: { name: number };
        env?: { name: number };
        [key: string]: any;
    };
    // Children: array or { "0": node, "1": node, ... } object map.
    child?: RawLcc2Node[] | Record<string, RawLcc2Node>;

    // Old protocol fields (need mapping).
    files?: string[];
    child_num?: number;
    datatype?: any;

    // New protocol fields (used directly).
    splatFiles?: string[];
    childNum?: number;
    dataType?: any;

    [key: string]: any;
};

// Top-level meta.lcc2 raw shape.
type RawLcc2Meta = {
    // Old protocol trio (all three must exist to trigger the old branch).
    total_splats?: number;
    lod_3dgs_info?: number[];
    lod_level?: number;

    // New protocol fields.
    totalSplats?: number;
    lodSplats?: number[];
    totalLevels?: number;

    // Unified chunk encoding type; missing is treated as '.sog'.
    splatType?: '.sog' | '.spz';

    // Bounding box (passed through for downstream/debug use).
    boundingBox?: any;

    // Octree root node.
    root: RawLcc2Node;

    [key: string]: any;
};

// --- Normalized model (post-parse) ------------------------------------------

// Normalized octree node. collectFileIndicesForLod only relies on
// data['3dgs'].name and child.
type Lcc2TreeNode = {
    data?: {
        '3dgs'?: { name: number };
        env?: { name: number };
        [key: string]: any;
    };
    child?: Lcc2TreeNode[] | Record<string, Lcc2TreeNode>;
    splatFiles?: string[];
    childNum?: number;
    dataType?: any;
    [key: string]: any;
};

// Return value of parseLcc2Meta.
type Lcc2Meta = {
    /** Total splat count for the whole scene (old protocol total_splats). */
    totalSplats: number;
    /** Per-LOD splat counts (old protocol lod_3dgs_info reversed). */
    lodSplats: number[];
    /** Total LOD level count (old protocol lod_level). Basis for LOD_Level = totalLevels - depth. */
    totalLevels: number;
    /** Chunk file paths, addressed by file index (root.splatFiles). */
    splatFiles: string[];
    /** Unified chunk encoding type, defaults to '.sog'. */
    splatType: '.sog' | '.spz';
    /** Bounding box (passed through). */
    boundingBox: any;
    /** Optional environment chunk file index (root.data.env.name); undefined if absent. */
    envFileIndex: number | undefined;
    /** Octree root node (normalized). */
    tree: Lcc2TreeNode;
};

/**
 * Return the children of an octree node, tolerating both array and
 * string-keyed object map shapes (and null/undefined).
 *
 * @param node - The octree node.
 * @returns Child nodes as an array (empty when there are none).
 * @ignore
 */
const getChildren = (node: Lcc2TreeNode): Lcc2TreeNode[] => {
    const c = node.child;
    if (c === null || c === undefined) {
        return [];
    }
    if (Array.isArray(c)) {
        return c;
    }
    return Object.values(c);
};

/**
 * Parse meta.lcc2 text into a normalized {@link Lcc2Meta}.
 *
 * Tolerates trailing commas before `}` (non-strict JSON). Handles both the
 * old protocol (total_splats / lod_3dgs_info / lod_level + files / child_num /
 * datatype) and the new protocol (canonical field names) field naming.
 *
 * @param metaText - Raw meta.lcc2 file contents.
 * @param filename - Source filename, used to build descriptive parse errors.
 * @returns Normalized LCC2 meta model.
 * @ignore
 */
const parseLcc2Meta = (metaText: string, filename: string): Lcc2Meta => {
    // 1) Strip trailing commas before `}` so non-strict JSON still parses.
    const cleaned = metaText.replace(/,\s*\}/g, '}');
    let meta: RawLcc2Meta;
    try {
        meta = JSON.parse(cleaned) as RawLcc2Meta;
    } catch (e) {
        const reason = (e as Error)?.message ?? String(e);
        throw new Error(
            `Failed to parse meta.lcc2 as JSON: ${filename}: ${reason}`
        );
    }

    // 2) Old protocol compatibility branch: only triggered when all three
    // legacy fields are present. Use explicit presence checks (not truthiness)
    // so legitimate zero values (e.g. total_splats: 0, lod_level: 0) don't
    // incorrectly skip the legacy normalization branch.
    if (
        meta.total_splats !== undefined &&
        meta.lod_3dgs_info !== undefined &&
        meta.lod_level !== undefined
    ) {
        // Recursively normalize node field names.
        const normalizeNode = (node: RawLcc2Node) => {
            if (node.datatype !== undefined) {
                node.dataType = node.datatype;
            }
            if (node.child_num !== undefined) {
                node.childNum = node.child_num;
            }
            for (const child of getChildren(node as Lcc2TreeNode)) {
                normalizeNode(child as RawLcc2Node);
            }
        };

        meta.totalSplats = meta.total_splats;
        meta.lodSplats = [...meta.lod_3dgs_info].reverse();
        meta.totalLevels = meta.lod_level;
        // root.files -> root.splatFiles: drop leading '/', append '.sog'.
        meta.root.splatFiles = (meta.root.files ?? []).map((f) => {
            let p = f.startsWith('/') ? f.slice(1) : f;
            if (!p.endsWith('.sog')) {
                p = `${p}.sog`;
            }
            return p;
        });
        normalizeNode(meta.root);
    }

    // 3) env detection.
    const envFileIndex = meta.root?.data?.env ?
        meta.root.data.env.name :
        undefined;

    // 4) Validate required fields rather than masking absence with type
    // assertions. totalLevels and root.splatFiles are mandatory for the reader
    // to resolve LODs and address chunk files; totalSplats / lodSplats are
    // scene metadata that downstream consumers may rely on.
    if (typeof meta.totalLevels !== 'number') {
        throw new Error(
            `Invalid meta.lcc2 (missing or non-numeric totalLevels): ${filename}`
        );
    }
    if (!Array.isArray(meta.root?.splatFiles)) {
        throw new Error(
            `Invalid meta.lcc2 (missing root.splatFiles): ${filename}`
        );
    }
    if (typeof meta.totalSplats !== 'number') {
        throw new Error(
            `Invalid meta.lcc2 (missing or non-numeric totalSplats): ${filename}`
        );
    }
    if (!Array.isArray(meta.lodSplats)) {
        throw new Error(
            `Invalid meta.lcc2 (missing lodSplats): ${filename}`
        );
    }

    // 5) Return normalized model.
    return {
        totalSplats: meta.totalSplats,
        lodSplats: meta.lodSplats,
        totalLevels: meta.totalLevels,
        splatFiles: meta.root.splatFiles,
        splatType: meta.splatType ?? '.sog',
        boundingBox: meta.boundingBox,
        envFileIndex,
        tree: meta.root as Lcc2TreeNode
    };
};

/**
 * Collect the chunk file indices belonging to a target LOD level by traversing
 * the octree.
 *
 * Traversal starts from the root's children with `depth` counting from 1; each
 * node's LOD level is `totalLevels - depth`. Only nodes carrying `data['3dgs']`
 * are collected, and indices equal to `envFileIndex` are skipped.
 *
 * @param tree - The octree root node.
 * @param targetLod - The target LOD level to collect.
 * @param totalLevels - Total LOD level count.
 * @param envFileIndex - Optional environment chunk file index to exclude.
 * @returns Set of chunk file indices for the target LOD.
 * @ignore
 */
const collectFileIndicesForLod = (
    tree: Lcc2TreeNode,
    targetLod: number,
    totalLevels: number,
    envFileIndex: number | undefined
): Set<number> => {
    const result = new Set<number>();
    const traverse = (node: Lcc2TreeNode, depth: number) => {
        for (const child of getChildren(node)) {
            const level = totalLevels - depth;
            if (level === targetLod && child.data?.['3dgs']) {
                const fileIndex = child.data['3dgs'].name;
                if (fileIndex !== envFileIndex) {
                    result.add(fileIndex);
                }
            }
            traverse(child, depth + 1);
        }
    };
    traverse(tree, 1);
    return result;
};

/**
 * Resolve the final ordered list of LOD levels to read.
 *
 * Mirrors readLcc's LOD selection, using `totalLevels` as the valid-LOD basis.
 * Negative indices count from the end; out-of-range indices are filtered out.
 * An empty selection means "all" levels.
 *
 * @param lodSelect - Requested LOD levels (may include negative indices).
 * @param totalLevels - Total LOD level count.
 * @returns Ordered list of input LOD levels. Its index is the output LOD index.
 * @ignore
 */
const resolveLodSelection = (
    lodSelect: number[],
    totalLevels: number
): number[] => {
    if (lodSelect.length > 0) {
        return lodSelect
        .map(lod => (lod < 0 ? totalLevels + lod : lod)) // negative -> from end
        .filter(lod => lod >= 0 && lod < totalLevels); // drop out-of-range
    }
    // empty = all [0, totalLevels)
    return Array.from({ length: totalLevels }, (_, i) => i);
};

/**
 * Whether an error from opening/decoding a chunk indicates the file is simply
 * missing (as opposed to a permission, I/O or corruption error). Covers Node's
 * `ENOENT`, the zip/memory file systems' "Entry not found" and HTTP 404.
 *
 * @param err - The thrown error.
 * @returns True if the error means the file was not found.
 * @ignore
 */
const isMissingError = (err: unknown): boolean => {
    const code = (err as { code?: string })?.code;
    const message = (err as Error)?.message ?? '';
    return code === 'ENOENT' ||
        message.startsWith('Entry not found') ||
        message.startsWith('HTTP error 404');
};

/**
 * Open a chunk file's read source, trying the full path first and falling back
 * to its basename only when the full path is not found (e.g. drag-and-drop
 * imports that provide filenames without directories).
 *
 * The happy path opens the file exactly once. The basename retry only runs when
 * the full path is missing, so remote/HTTP file systems avoid a redundant probe
 * round-trip per chunk. Non-"missing" failures (permission, I/O, corruption)
 * are surfaced immediately rather than masked by a basename retry. If the
 * fallback also fails, the original (full-path) error is surfaced as it is the
 * more informative one.
 *
 * @param fileSystem - File system used to open the chunk.
 * @param fullPath - The full path recorded in meta.lcc2.
 * @returns An open {@link ReadSource} for the chunk.
 * @ignore
 */
const openChunkSource = async (
    fileSystem: ReadFileSystem,
    fullPath: string
): Promise<ReadSource> => {
    try {
        return await fileSystem.createSource(fullPath);
    } catch (err) {
        const base = basename(fullPath);
        // Only retry with the bare filename when the full path is genuinely
        // missing; other failures are real and must not be masked.
        if (!isMissingError(err) || base === fullPath) {
            throw err;
        }
        try {
            return await fileSystem.createSource(base); // fall back to bare filename
        } catch {
            throw err; // surface the original full-path error
        }
    }
};

/**
 * Decode a single LCC2 chunk file into a {@link DataTable}.
 *
 * SOG chunks are ZIP containers decoded via {@link readSog} (wrapped in a
 * {@link ZipReadFileSystem}); SPZ chunks are raw binaries decoded via
 * {@link readSpz}. Both return a `Transform.PLY` DataTable. The chunk source is
 * opened once (full path, with a basename fallback) and always closed.
 *
 * @param fileSystem - File system to read the chunk from.
 * @param splatType - Chunk encoding type ('.sog' or '.spz').
 * @param fullPath - The chunk file path recorded in meta.lcc2.
 * @returns Decoded DataTable.
 * @ignore
 */
const decodeChunk = async (
    fileSystem: ReadFileSystem,
    splatType: string,
    fullPath: string
): Promise<DataTable> => {
    const source = await openChunkSource(fileSystem, fullPath);
    try {
        if (splatType === '.sog') {
            const zipFs = new ZipReadFileSystem(source); // SOG is a ZIP container
            try {
                return await readSog(zipFs, 'meta.json'); // call readSog directly to avoid circular deps
            } finally {
                zipFs.close(); // close the zip wrapper (also closes the source)
            }
        }
        if (splatType === '.spz') {
            return await readSpz(source); // SPZ is a raw binary
        }
        throw new Error(`Unsupported LCC2 splatType: ${splatType}`);
    } finally {
        source.close(); // idempotent; safe even after zipFs closed it
    }
};

/**
 * Reads an XGrids LCC2 format containing multi-LOD Gaussian splat data.
 *
 * Unlike LCC v1 (quadtree + single data.bin/shcoef.bin), LCC2 uses an octree
 * spatial structure where each node's data is an independent `.sog` / `.spz`
 * chunk file, described by a `meta.lcc2` (JSON) file listing the tree, file
 * paths, LOD levels and bounding box.
 *
 * Chunks for the selected LODs are decoded (reusing the internal `readSog` /
 * `readSpz` decoders), tagged with an output `lod` column, combined into a
 * single DataTable and re-tagged with the LCC2 coordinate transform. An
 * optional environment chunk is loaded as an additional table (lod = -1).
 *
 * Behavior (multi-LOD selection, `lod` column, coordinate transform) mirrors
 * `readLcc`.
 *
 * @param fileSystem - File system for reading the LCC2 files.
 * @param filename - Path to the meta.lcc2 file.
 * @param options - Options including LOD selection via `lodSelect`.
 * @returns Promise resolving to an array of DataTables (combined LODs + optional environment).
 * @ignore
 */
const readLcc2 = async (
    fileSystem: ReadFileSystem,
    filename: string,
    options: Options
): Promise<DataTable[]> => {
    // 1) Read and parse meta.lcc2.
    const baseDir = dirname(filename);
    const related = (name: string) => (baseDir ? join(baseDir, name) : name);
    const metaBytes = await readFile(fileSystem, filename);
    const meta = parseLcc2Meta(new TextDecoder().decode(metaBytes), filename);
    const { totalLevels, splatFiles, splatType, envFileIndex, tree } = meta;

    // 2) Resolve LOD selection.
    const inputLods = resolveLodSelection(options.lodSelect, totalLevels);
    if (inputLods.length === 0) {
        throw new Error(
            `No valid LODs selected for LCC2 input file: ${filename} lods: ${JSON.stringify(options.lodSelect)}`
        );
    }

    // 3) Collect decode tasks in deterministic order: by LOD order, then by
    // ascending file index within each LOD.
    const tasks: { outputLod: number; fileIndex: number }[] = [];
    for (let outputLod = 0; outputLod < inputLods.length; outputLod++) {
        const inputLod = inputLods[outputLod];
        const indices = Array.from(
            collectFileIndicesForLod(tree, inputLod, totalLevels, envFileIndex)
        ).sort((a, b) => a - b);
        for (const fileIndex of indices) {
            tasks.push({ outputLod, fileIndex });
        }
    }

    // No decodable chunks for the selected LODs: bail out with a descriptive
    // error rather than letting combine() throw on an empty array below.
    if (tasks.length === 0) {
        throw new Error(
            `No chunks found for selected LODs in LCC2 input file: ${filename} lods: ${JSON.stringify(inputLods)}`
        );
    }

    // 4) Pre-allocate output slots so combine input order stays reproducible.
    // Every slot is filled by exactly one worker below; any decode failure
    // rejects Promise.all and aborts before combine, so there are no holes.
    const tables: DataTable[] = new Array(tasks.length);

    // 5) One progress bar covering all chunks.
    const bar = logger.bar('decoding', tasks.length);
    let done = 0;

    // 6) Bounded-concurrency worker pool. Any failure here propagates: the bar
    // is intentionally not end()ed on the error path, leaving it open so
    // logger's error path can mark it as failed instead of finalizing it first.
    let next = 0;
    const worker = async () => {
        while (true) {
            const i = next++;
            if (i >= tasks.length) break;
            const task = tasks[i];
            const fullPath = related(splatFiles[task.fileIndex]);
            const dt = await decodeChunk(fileSystem, splatType, fullPath);
            // Write lod column = output LOD index.
            dt.addColumn(
                new Column('lod', new Float32Array(dt.numRows).fill(task.outputLod))
            );
            tables[i] = dt; // write pre-allocated slot (no push, keeps order)
            bar.update(++done);
        }
    };
    await Promise.all(
        Array.from({ length: Math.min(LOAD_CONCURRENCY, tasks.length) }, () => worker())
    );

    // 7) Combine + override transform.
    const merged = combine(tables);
    merged.transform = LCC2_TRANSFORM();
    // Close the bar only on success path.
    bar.end();

    const result: DataTable[] = [merged];

    // 8) Optional environment chunk (approach B): a missing file is normal.
    if (envFileIndex !== undefined) {
        try {
            const envFull = related(splatFiles[envFileIndex]);
            const envTable = await decodeChunk(fileSystem, splatType, envFull);
            envTable.addColumn(
                new Column('lod', new Float32Array(envTable.numRows).fill(-1))
            );
            envTable.transform = LCC2_TRANSFORM();
            result.push(envTable);
        } catch (err) {
            if (!isMissingError(err)) {
                const message = (err as Error)?.message ?? '';
                logger.warn(`failed to load LCC2 environment chunk: ${message || err}`);
            }
        }
    }

    return result;
};

export {
    LOAD_CONCURRENCY,
    LCC2_TRANSFORM,
    parseLcc2Meta,
    getChildren,
    collectFileIndicesForLod,
    resolveLodSelection,
    isMissingError,
    openChunkSource,
    decodeChunk,
    readLcc2
};

export type { RawLcc2Node, RawLcc2Meta, Lcc2TreeNode, Lcc2Meta };
