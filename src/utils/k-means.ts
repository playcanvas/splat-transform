import { Column, DataTable } from '../data-table';
import { KdTree } from './kd-tree';
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

const kmeans = (dataTable: DataTable, k: number, device?: GpuDevice) => {
    // too few data points
    if (dataTable.numRows < k) {
        return {
            centroids: dataTable.clone(),
            labels: new Array(dataTable.numRows).fill(0).map((_, i) => i)
        };
    }

    const row: any = {};

    // construct centroids data table and assign initial values
    const centroids = new DataTable(dataTable.columns.map(c => new Column(c.name, new Float32Array(k))));
    initializeCentroids(dataTable, centroids, row);

    const clusters: number[][] = [];
    for (let i = 0; i < k; ++i) {
        clusters[i] = [];
    }

    let converged = false;
    let steps = 0;

    while (!converged) {
        // reset clusters
        clusters.forEach((c) => {
            c.length = 0;
        });

        // construct a kdtree over the centroids so we can find the nearest quickly
        const kdTree = new KdTree(centroids.columns.map(c => c.data) as Float32Array[]);
        const point = new Float32Array(dataTable.numColumns);

        // assign each point to the nearest centroid
        for (let i = 0; i < dataTable.numRows; ++i) {
            dataTable.getRow(i, row);
            dataTable.columns.forEach((c, i) => {
                point[i] = row[c.name];
            });

            const result = kdTree.findNearest(point);

            clusters[result.index].push(i);
        }

        // calculate the new centroid positions
        for (let i = 0; i < k; ++i) {
            calcAverage(dataTable, clusters[i], row);
            centroids.setRow(i, row);
        }

        steps++;
        if (steps > 10) {
            converged = true;
        }
    }

    const labels = new Uint32Array(dataTable.numRows);

    // construct labels from clusters
    for (let i = 0; i < clusters.length; ++i) {
        const cluster = clusters[i];
        for (let j = 0; j < cluster.length; ++j) {
            labels[cluster[j]] = i;
        }
    }

    return { centroids, labels };
};

export { kmeans };
