import { type FileSystem } from './file-system';
import { fmtBytes, logger } from '../../utils';

const writeFile = async (fs: FileSystem, filename: string, data: Uint8Array | string) => {
    const outputFile = await fs.createWriter(filename);
    await outputFile.write(data instanceof Uint8Array ? data : new TextEncoder().encode(data));
    await outputFile.close();
};

/**
 * Emit a single `Writing`-group entry as `<filename> (<formatted size>)`.
 *
 * @param filename - Display name for the written file.
 * @param bytes - Number of bytes written.
 */
const logWrittenFile = (filename: string, bytes: number): void => {
    logger.info(`${filename} (${fmtBytes(bytes)})`);
};

export { logWrittenFile, writeFile };
