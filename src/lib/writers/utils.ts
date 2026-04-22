import { fmtBytes, logger } from '../utils';

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

export { logWrittenFile };
