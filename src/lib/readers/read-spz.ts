import { DataTable } from '../data-table';
import { ReadSource } from '../io/read';
import { gaussianCloudToDataTable, getSpzModule } from '../spz-module';

/**
 * Reads a .spz file using the official SPZ WebAssembly backend.
 *
 * Decodes to RDF (PLY) space — splat-transform's SPZ pipeline writes with
 * `from: RDF` and reads with `to: RDF`, treating SPZ on-disk as the spec'd
 * RUB form.
 *
 * @param source - The read source providing access to the .spz file data.
 * @returns Promise resolving to a DataTable containing the splat data.
 * @ignore
 */
const readSpz = async (source: ReadSource): Promise<DataTable> => {
    const spz = await getSpzModule();
    const fileBuffer = await source.read().readAll();
    const spzBuffer = fileBuffer.byteOffset === 0 && fileBuffer.byteLength === fileBuffer.buffer.byteLength ?
        fileBuffer :
        fileBuffer.slice();
    const cloud = await spz.loadSpzFromBuffer(spzBuffer, {
        to: spz.CoordinateSystem.RDF
    });

    return gaussianCloudToDataTable(cloud);
};

export { readSpz };
