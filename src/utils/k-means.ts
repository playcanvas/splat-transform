import { Column, DataTable } from '../data-table';
import { logger } from '../logger';
import { KdTree } from './kd-tree';
import { GpuClustering } from '../gpu/gpu-clustering';
import { GpuDevice } from '../gpu/gpu-device';

const initializeCentroids = (dataTable: DataTable, centroids: DataTable, row: any) => {
    const chosenRows = new Set();
    for (let i = 0; i < centroids.numRows; ++i) {
        let candidateRow;
        do {
            candidateRow = Math.floor(Math.random() * dataTable.numRows);
        } while (chosenRows.has(candidateRow));

        chosenRows.add(candidateRow);
        dataTable.getRow(candidateRow, row);
        centroids.setRow(i, row);
    }
};

// in the 1d case we can initialize centroids evenly over the input range
const initializeCentroids1D = (dataTable: DataTable, centroids: DataTable) => {
    // calculate min/max
    let m = Infinity;
    let M = -Infinity;

    const data = dataTable.getColumn(0).data;
    for (let i = 0; i < dataTable.numRows; ++i) {
        const value = data[i];
        if (value < m) m = value;
        if (value > M) M = value;
    }

    const centroidsData = centroids.getColumn(0).data;
    for (let i = 0; i < centroids.numRows; ++i) {
        centroidsData[i] = m + (M - m) * i / (centroids.numRows - 1);
    }
};

const calcAverage = (dataTable: DataTable, cluster: number[], row: any) => {
    const keys = dataTable.columnNames;

    for (let i = 0; i < keys.length; ++i) {
        row[keys[i]] = 0;
    }

    const dataRow: any = {};
    for (let i = 0; i < cluster.length; ++i) {
        dataTable.getRow(cluster[i], dataRow);

        for (let j = 0; j < keys.length; ++j) {
            const key = keys[j];
            row[key] += dataRow[key];
        }
    }

    if (cluster.length > 0) {
        for (let i = 0; i < keys.length; ++i) {
            row[keys[i]] /= cluster.length;
        }
    }
};

// cpu cluster
const clusterCpu = (points: DataTable, centroids: DataTable, labels: Uint32Array) => {
    const numColumns = points.numColumns;

    const pData = points.columns.map(c => c.data);
    const cData = centroids.columns.map(c => c.data);

    const point = new Float32Array(numColumns);

    const distance = (centroidIndex: number) => {
        let result = 0;
        for (let i = 0; i < numColumns; ++i) {
            const v = point[i] - cData[i][centroidIndex];
            result += v * v;
        }
        return result;
    };

    for (let i = 0; i < points.numRows; ++i) {
        let mind = Infinity;
        let mini = -1;

        for (let c = 0; c < numColumns; ++c) {
            point[c] = pData[c][i];
        }

        for (let j = 0; j < centroids.numRows; ++j) {
            const d = distance(j);
            if (d < mind) {
                mind = d;
                mini = j;
            }
        }

        labels[i] = mini;
    }
};

const clusterKdTreeCpu = (points: DataTable, centroids: DataTable, labels: Uint32Array) => {
    const kdTree = new KdTree(centroids);

    // construct a kdtree over the centroids so we can find the nearest quickly
    const point = new Float32Array(points.numColumns);
    const row: any = {};

    // assign each point to the nearest centroid
    for (let i = 0; i < points.numRows; ++i) {
        points.getRow(i, row);
        points.columns.forEach((c, i) => {
            point[i] = row[c.name];
        });

        const a = kdTree.findNearest(point);

        labels[i] = a.index;
    }
};

const groupLabels = (labels: Uint32Array, k: number) => {
    const clusters: number[][] = [];

    for (let i = 0; i < k; ++i) {
        clusters[i] = [];
    }

    for (let i = 0; i < labels.length; ++i) {
        clusters[labels[i]].push(i);
    }

    return clusters;
};

const kmeans = async (points: DataTable, k: number, iterations: number, device?: GpuDevice) => {
    // too few data points
    if (points.numRows < k) {
        return {
            centroids: points.clone(),
            // use a typed array here so downstream code can rely on
            // labels supporting subarray(), even in this early-return
            // path used for very small datasets.
            labels: new Uint32Array(points.numRows).map((_, i) => i)
        };
    }

    const row: any = {};

    // construct centroids data table and assign initial values
    const centroids = new DataTable(points.columns.map(c => new Column(c.name, new Float32Array(k))));
    if (points.numColumns === 1) {
        initializeCentroids1D(points, centroids);
    } else {
        initializeCentroids(points, centroids, row);
    }

    const gpuClustering = device && new GpuClustering(device, points.numColumns, k);
    const labels = new Uint32Array(points.numRows);

    let converged = false;
    let steps = 0;

    logger.debug(`Running k-means clustering: dims=${points.numColumns} points=${points.numRows} clusters=${k} iterations=${iterations}...`);

    while (!converged) {
        if (gpuClustering) {
            await gpuClustering.execute(points, centroids, labels);
        } else {
            clusterKdTreeCpu(points, centroids, labels);
        }

        // calculate the new centroid positions
        const groups = groupLabels(labels, k);
        for (let i = 0; i < centroids.numRows; ++i) {
            if (groups[i].length === 0) {
                // re-seed this centroid to a random point to avoid zero vector
                const idx = Math.floor(Math.random() * points.numRows);
                points.getRow(idx, row);
                centroids.setRow(i, row);
            } else {
                calcAverage(points, groups[i], row);
                centroids.setRow(i, row);
            }
        }

        steps++;

        if (steps >= iterations) {
            converged = true;
        }

        logger.progress('#');
    }

    if (gpuClustering) {
        gpuClustering.destroy();
    }

    logger.debug(' done 🎉');

    return { centroids, labels };
};

export { kmeans };
