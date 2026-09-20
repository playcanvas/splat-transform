import { extname } from 'pathe';

import { type SplatModel } from '../splat-model';
import { fmtBytes, logger } from '../utils';

// Brush's spelling of the model, which its importer (brush-serde import.rs) and
// ours both read — `default` is what an untagged file already means, so it's
// never written.
const MODEL_TO_MODE: Readonly<Record<SplatModel, string | null>> = {
    default: null,
    antialiased: 'mip',
    '2dgs': '2dgs'
};

/**
 * The PLY header comment tagging a splat model, shared by the three PLY writers.
 *
 * @param model - The model to tag.
 * @returns The comment text (without the leading `comment `), or `null` for an untagged (`default`) scene.
 */
const splatModelComment = (model: SplatModel): string | null => {
    const mode = MODEL_TO_MODE[model];
    return mode && `SplatRenderMode: ${mode}`;
};

/**
 * Emit a single `Writing`-group entry as `<filename> (<formatted size>)`.
 *
 * Lives here (rather than in `io/write/`) so the low-level I/O layer stays
 * decoupled from the logger / formatting code; only writer modules pull this
 * in alongside their renderer-aware output.
 *
 * @param filename - Display name for the written file.
 * @param bytes - Number of bytes written.
 */
const logWrittenFile = (filename: string, bytes: number): void => {
    logger.info(`${filename} (${fmtBytes(bytes)})`);
};

/**
 * Output filename of one frame of an image sequence: `<name>.NNNN<ext>`,
 * the frame number padded to at least four digits and to the width of the
 * last frame's.
 *
 * @param filename - The sequence's output filename.
 * @param frame - The frame number.
 * @param lastFrame - The last frame number of the sequence, which sets the padding.
 * @returns The frame's filename.
 */
const frameFilename = (filename: string, frame: number, lastFrame: number): string => {
    const ext = extname(filename);
    const digits = Math.max(4, String(lastFrame).length);
    return `${filename.slice(0, filename.length - ext.length)}.${String(frame).padStart(digits, '0')}${ext}`;
};

export { frameFilename, logWrittenFile, splatModelComment };
