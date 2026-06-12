import { Column, DataTable, TypedArray } from '../data-table';
import { readSog } from './read-sog';
import { readSpz } from './read-spz';
import { basename, dirname, join, readFile, ReadFileSystem, ReadSource, ReadStream, ZipReadFileSystem } from '../io/read';
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
    // points to the optional environment chunk. A chunk file may be shared by
    // several nodes (same level), each owning the [start, start + count) range
    // of its rows.
    data?: {
        '3dgs'?: { name: number; start?: number; count?: number };
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

// Normalized octree node. collectChunksByLevel only relies on
// data['3dgs'] and child.
type Lcc2TreeNode = {
    data?: {
        '3dgs'?: { name: number; start?: number; count?: number };
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
 * Strip trailing commas before `}` or `]` outside string literals, so
 * non-strict JSON (as emitted by some LCC2 exporters) still parses. A linear
 * scan tracking in-string and escape state; string contents (e.g. file names
 * containing ",}") pass through untouched.
 *
 * @param text - The raw (non-strict) JSON text.
 * @returns The text with trailing commas removed.
 * @ignore
 */
const stripTrailingCommas = (text: string): string => {
    let result = '';
    let inString = false;
    let escaped = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (c === '\\') {
                escaped = true;
            } else if (c === '"') {
                inString = false;
            }
            result += c;
            continue;
        }
        if (c === '"') {
            inString = true;
            result += c;
            continue;
        }
        if (c === ',') {
            // Drop the comma when the next non-whitespace char closes the
            // current object/array.
            let j = i + 1;
            while (j < text.length && /\s/.test(text[j])) j++;
            if (j < text.length && (text[j] === '}' || text[j] === ']')) {
                continue;
            }
        }
        result += c;
    }
    return result;
};

/**
 * Parse meta.lcc2 text into a normalized {@link Lcc2Meta}.
 *
 * Tolerates trailing commas before `}` / `]` (non-strict JSON). Handles both
 * the old protocol (total_splats / lod_3dgs_info / lod_level + files /
 * child_num / datatype) and the new protocol (canonical field names) field
 * naming.
 *
 * @param metaText - Raw meta.lcc2 file contents.
 * @param filename - Source filename, used to build descriptive parse errors.
 * @returns Normalized LCC2 meta model.
 * @ignore
 */
const parseLcc2Meta = (metaText: string, filename: string): Lcc2Meta => {
    // 1) Strict parse first (real-world metas are valid JSON); only on failure
    // retry with trailing commas stripped.
    let meta: RawLcc2Meta;
    try {
        meta = JSON.parse(metaText) as RawLcc2Meta;
    } catch {
        try {
            meta = JSON.parse(stripTrailingCommas(metaText)) as RawLcc2Meta;
        } catch (e) {
            const reason = (e as Error)?.message ?? String(e);
            throw new Error(
                `Failed to parse meta.lcc2 as JSON: ${filename}: ${reason}`
            );
        }
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
 * Collect the chunk files of every LOD level in a single octree traversal,
 * along with each file's splat count when the meta provides one.
 *
 * Traversal starts from the root's children with `depth` counting from 1; each
 * node's LOD level is `totalLevels - depth`. Only nodes carrying `data['3dgs']`
 * are collected, and indices equal to `envFileIndex` are skipped. A chunk file
 * may be shared by several nodes of the same level (each owning a row range),
 * so a file's count is the sum of its nodes' `count` fields; if any
 * contributing node lacks a valid count, the file's count is `undefined` and
 * the caller falls back to reading it from the chunk itself.
 *
 * @param tree - The octree root node.
 * @param totalLevels - Total LOD level count.
 * @param envFileIndex - Optional environment chunk file index to exclude.
 * @returns Map of LOD level to (file index -> splat count or undefined).
 * @ignore
 */
const collectChunksByLevel = (
    tree: Lcc2TreeNode,
    totalLevels: number,
    envFileIndex: number | undefined
): Map<number, Map<number, number | undefined>> => {
    const result = new Map<number, Map<number, number | undefined>>();
    const traverse = (node: Lcc2TreeNode, depth: number) => {
        for (const child of getChildren(node)) {
            const data = child.data?.['3dgs'];
            if (data && data.name !== envFileIndex) {
                const level = totalLevels - depth;
                let files = result.get(level);
                if (!files) {
                    files = new Map();
                    result.set(level, files);
                }
                const count = data.count;
                if (!Number.isInteger(count) || (count as number) < 0) {
                    files.set(data.name, undefined);
                } else if (files.has(data.name)) {
                    const prev = files.get(data.name);
                    files.set(data.name, prev === undefined ? undefined : prev + (count as number));
                } else {
                    files.set(data.name, count);
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
                // call readSog directly to avoid circular deps; silence its
                // internal bar - logger bars are strictly LIFO and chunks
                // decode concurrently under readLcc2's own bar
                return await readSog(zipFs, 'meta.json', { logging: 'silent' });
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

// SPZ header: u32 magic 'NGSP', u32 version, u32 numPoints, u8 shDegree,
// u8 fractionalBits, u8 flags, u8 reserved. v4 files start with this header
// in plaintext; v1-3 files are gzip-compressed end-to-end with the same
// header at the start of the decompressed stream.
const SPZ_HEADER_SIZE = 16;

// Compressed prefix to pull when gunzipping just the SPZ header. 64KB of
// compressed input always yields >= 16 decompressed bytes for a real gzip
// stream while keeping remote range reads small.
const GZIP_PREFIX_LIMIT = 65536;

/**
 * Read up to `length` bytes from a stream into a buffer. May return fewer
 * bytes if the stream ends early. The caller closes the stream.
 *
 * @param stream - The stream to read from.
 * @param length - Maximum number of bytes to read.
 * @returns The bytes read (length <= `length`).
 * @ignore
 */
const readPrefix = async (stream: ReadStream, length: number): Promise<Uint8Array> => {
    const result = new Uint8Array(length);
    let read = 0;
    while (read < length) {
        const n = await stream.pull(result.subarray(read));
        if (n === 0) break;
        read += n;
    }
    return result.subarray(0, read);
};

/**
 * Stream-gunzip only the first `length` bytes of a gzip stream, then cancel
 * the decompressor without consuming the rest of the input.
 *
 * @param stream - Stream over the (bounded) compressed input. The caller closes it.
 * @param length - Number of decompressed bytes wanted.
 * @returns The first `length` decompressed bytes.
 * @ignore
 */
const gunzipPrefix = async (stream: ReadStream, length: number): Promise<Uint8Array> => {
    const inputStream = new ReadableStream<Uint8Array>({
        async pull(controller) {
            const chunk = new Uint8Array(32768);
            const n = await stream.pull(chunk);
            if (n === 0) {
                controller.close();
            } else {
                controller.enqueue(chunk.subarray(0, n));
            }
        }
    });
    // Type assertion needed due to TypeScript's strict typing of DecompressionStream
    const reader = inputStream.pipeThrough(
        new DecompressionStream('gzip') as unknown as TransformStream<Uint8Array, Uint8Array>
    ).getReader();
    try {
        const result = new Uint8Array(length);
        let read = 0;
        while (read < length) {
            const { done, value } = await reader.read();
            if (done) {
                throw new Error('Unexpected end of gzip stream');
            }
            const n = Math.min(value.length, length - read);
            result.set(value.subarray(0, n), read);
            read += n;
        }
        return result;
    } finally {
        // Stop decompression early; the bounded input is truncated, so
        // draining it to EOF would throw.
        await reader.cancel().catch(() => {});
    }
};

/**
 * Validate the 'NGSP' magic and extract numPoints from an SPZ header.
 *
 * @param header - The (decompressed) leading bytes of the SPZ data.
 * @param context - Chunk path, used to build descriptive errors.
 * @returns The header's numPoints field.
 * @ignore
 */
const parseSpzNumPoints = (header: Uint8Array, context: string): number => {
    if (header.length < SPZ_HEADER_SIZE ||
        header[0] !== 0x4e || header[1] !== 0x47 || header[2] !== 0x53 || header[3] !== 0x50) {
        throw new Error(`Invalid SPZ chunk header: ${context}`);
    }
    const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
    return view.getUint32(8, true);
};

/**
 * Read a chunk's exact splat count without decoding it. Used as a fallback
 * when meta.lcc2 doesn't provide per-node counts: costs one extra open plus a
 * small read per chunk (noticeable mainly on remote file systems).
 *
 * SOG: opens the ZIP and reads only meta.json (`count` for V2,
 * `means.shape[0]` for V1, mirroring readSog's version dispatch). SPZ: parses
 * `numPoints` from the 16-byte header (gunzipping just the header for the
 * gzip-wrapped v1-3 container).
 *
 * @param fileSystem - File system to read the chunk from.
 * @param splatType - Chunk encoding type ('.sog' or '.spz').
 * @param fullPath - The chunk file path recorded in meta.lcc2.
 * @returns The chunk's splat count.
 * @ignore
 */
const readChunkCount = async (
    fileSystem: ReadFileSystem,
    splatType: string,
    fullPath: string
): Promise<number> => {
    const source = await openChunkSource(fileSystem, fullPath);
    try {
        if (splatType === '.sog') {
            const zipFs = new ZipReadFileSystem(source);
            try {
                const sogMeta = JSON.parse(new TextDecoder().decode(await readFile(zipFs, 'meta.json')));
                const version = sogMeta.version;
                const count = version === undefined ?
                    sogMeta.means?.shape?.[0] : // V1: texture shape
                    (version === 2 ? sogMeta.count : undefined);
                if (!Number.isInteger(count) || count < 0) {
                    throw new Error(`Cannot determine SOG chunk splat count: ${fullPath}`);
                }
                return count;
            } finally {
                zipFs.close();
            }
        }
        if (splatType === '.spz') {
            const headStream = source.read(0, SPZ_HEADER_SIZE);
            let header: Uint8Array;
            try {
                header = await readPrefix(headStream, SPZ_HEADER_SIZE);
            } finally {
                headStream.close();
            }
            if (header.length >= 2 && header[0] === 0x1f && header[1] === 0x8b) {
                const zipped = source.read(0, GZIP_PREFIX_LIMIT);
                try {
                    header = await gunzipPrefix(zipped, SPZ_HEADER_SIZE);
                } finally {
                    zipped.close();
                }
            }
            return parseSpzNumPoints(header, fullPath);
        }
        throw new Error(`Unsupported LCC2 splatType: ${splatType}`);
    } finally {
        source.close();
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
 * `readSpz` decoders) directly into a single preallocated DataTable, tagged
 * with an output `lod` column and the LCC2 coordinate transform. Output
 * arrays are sized up front from the per-node splat counts in meta.lcc2
 * (falling back to reading each chunk's header when absent), so peak memory
 * stays at ~one copy of the scene plus the chunks currently being decoded.
 * An optional environment chunk is loaded as an additional table (lod = -1).
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
    // ascending file index within each LOD. A single traversal yields every
    // level's chunk files along with their meta-provided splat counts.
    const byLevel = collectChunksByLevel(tree, totalLevels, envFileIndex);
    const tasks: { outputLod: number; fileIndex: number; count: number | undefined }[] = [];
    for (let outputLod = 0; outputLod < inputLods.length; outputLod++) {
        const files = byLevel.get(inputLods[outputLod]);
        const indices = files ? Array.from(files.keys()).sort((a, b) => a - b) : [];
        for (const fileIndex of indices) {
            if (!Number.isInteger(fileIndex) || fileIndex < 0 ||
                fileIndex >= splatFiles.length || !splatFiles[fileIndex]) {
                throw new Error(
                    `Invalid chunk file index ${fileIndex} (root.splatFiles has ${splatFiles.length} entries) in LCC2 input file: ${filename}`
                );
            }
            tasks.push({ outputLod, fileIndex, count: files.get(fileIndex) });
        }
    }

    // No decodable chunks for the selected LODs: bail out with a descriptive
    // error rather than producing an empty table below.
    if (tasks.length === 0) {
        throw new Error(
            `No chunks found for selected LODs in LCC2 input file: ${filename} lods: ${JSON.stringify(inputLods)}`
        );
    }

    // 4) Resolve per-chunk splat counts. Counts from meta.lcc2 are preferred;
    // chunks lacking one fall back to a cheap header read (no decode). The
    // fallback costs an extra open plus a small read per chunk - noticeable
    // mainly on remote file systems - so meta counts are used when present.
    const counts = new Array<number>(tasks.length);
    const missing: number[] = [];
    for (let i = 0; i < tasks.length; i++) {
        const count = tasks[i].count;
        if (count === undefined) {
            missing.push(i);
        } else {
            counts[i] = count;
        }
    }
    if (missing.length > 0) {
        const scanBar = logger.bar('scanning', missing.length);
        let scanned = 0;
        let nextScan = 0;
        const scanWorker = async () => {
            while (true) {
                const m = nextScan++;
                if (m >= missing.length) break;
                const i = missing[m];
                counts[i] = await readChunkCount(
                    fileSystem, splatType, related(splatFiles[tasks[i].fileIndex])
                );
                scanBar.update(++scanned);
            }
        };
        await Promise.all(
            Array.from({ length: Math.min(LOAD_CONCURRENCY, missing.length) }, () => scanWorker())
        );
        // Close the bar only on success path (matches the decode bar below).
        scanBar.end();
    }

    // 5) Prefix-sum write offsets in task order: the output layout is fixed
    // before decoding starts, so workers completing out of order still write
    // into disjoint regions and the result stays deterministic (mirrors
    // readLcc's decodeUnitsForLod).
    const offsets = new Array<number>(tasks.length);
    let totalRows = 0;
    for (let i = 0; i < tasks.length; i++) {
        offsets[i] = totalRows;
        totalRows += counts[i];
    }

    // 6) Output columns, preallocated at the final row count. The column set
    // isn't known until chunks decode (it depends on each chunk's SH bands),
    // so each column is allocated on first sighting - still a single
    // exact-size allocation; rows from chunks lacking a column stay zero
    // (matching combine()'s behavior for missing columns). The first sighting
    // (task, position) is recorded so the assembled column order is
    // deterministic regardless of decode completion order.
    type OutputColumn = { data: TypedArray; task: number; pos: number };
    const outputs = new Map<string, OutputColumn>();
    const lodData = new Float32Array(totalRows);

    // 7) Bounded-concurrency decode pool: decode each chunk, validate its row
    // count against the expected count, copy it into the output arrays at its
    // precomputed offset and drop it. Any failure propagates: the bar is
    // intentionally not end()ed on the error path, leaving it open so
    // logger's error path can mark it as failed instead of finalizing it first.
    const bar = logger.bar('decoding', tasks.length);
    let done = 0;
    let next = 0;
    const worker = async () => {
        while (true) {
            const i = next++;
            if (i >= tasks.length) break;
            const task = tasks[i];
            const fullPath = related(splatFiles[task.fileIndex]);
            const dt = await decodeChunk(fileSystem, splatType, fullPath);
            if (dt.numRows !== counts[i]) {
                throw new Error(
                    `LCC2 chunk splat count mismatch for ${fullPath}: expected ${counts[i]}, decoded ${dt.numRows}`
                );
            }
            for (let pos = 0; pos < dt.columns.length; pos++) {
                const column = dt.columns[pos];
                let output = outputs.get(column.name);
                if (!output) {
                    const Ctor = column.data.constructor as new (length: number) => TypedArray;
                    output = { data: new Ctor(totalRows), task: i, pos };
                    outputs.set(column.name, output);
                } else if (i < output.task) {
                    output.task = i;
                    output.pos = pos;
                }
                output.data.set(column.data, offsets[i]);
            }
            // Write lod column = output LOD index.
            lodData.fill(task.outputLod, offsets[i], offsets[i] + dt.numRows);
            bar.update(++done);
            // dt goes out of scope here, releasing the chunk's memory.
        }
    };
    await Promise.all(
        Array.from({ length: Math.min(LOAD_CONCURRENCY, tasks.length) }, () => worker())
    );

    // 8) Assemble the merged table, columns ordered by first sighting.
    const columns = [...outputs.entries()]
    .sort(([, a], [, b]) => (a.task - b.task) || (a.pos - b.pos))
    .map(([name, output]) => new Column(name, output.data));
    columns.push(new Column('lod', lodData));
    const merged = new DataTable(columns, LCC2_TRANSFORM());
    // Close the bar only on success path.
    bar.end();

    const result: DataTable[] = [merged];

    // 9) Optional environment chunk (approach B): a missing file is normal.
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
    stripTrailingCommas,
    parseLcc2Meta,
    getChildren,
    collectChunksByLevel,
    resolveLodSelection,
    isMissingError,
    openChunkSource,
    decodeChunk,
    parseSpzNumPoints,
    readChunkCount,
    readLcc2
};

export type { RawLcc2Node, RawLcc2Meta, Lcc2TreeNode, Lcc2Meta };
