import { DataTable } from '../data-table';
import { ReadSource } from '../io/read';
import { gaussianCloudToDataTable, getSpzModule } from '../spz-module';

/**
 * Reads a .spz file using the official SPZ WebAssembly backend.
 *
 * Data is decoded to RDF/PLY space so the returned DataTable matches the
 * rest of splat-transform's coordinate conventions.
 *
 * @param source - The read source providing access to the .spz file data.
 * @returns Promise resolving to a DataTable containing the splat data.
 * @ignore
 */
const readSpz = async (source: ReadSource): Promise<DataTable> => {
    const spz = await getSpzModule();
    const fileBuffer = await source.read().readAll();
    const cloud = await spz.loadSpzFromBuffer(fileBuffer, {
        to: spz.CoordinateSystem.RDF
    });

    return gaussianCloudToDataTable(cloud);
};

export { readSpz };
