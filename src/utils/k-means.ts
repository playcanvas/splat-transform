import { stdout } from 'node:process';

import { Column, DataTable } from '../data-table';
import { GpuDevice } from '../gpu/gpu-device';
import { GpuCluster } from '../gpu/gpu-cluster';
import { KdTree } from './kd-tree';

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

// cpu cluster
const clusterCpu = (points: DataTable, centroids: DataTable, labels: Uint32Array) => {
    const numColumns = points.numColumns;

    const pData = points.columns.map(c => c.data);
    const cData = centroids.columns.map(c => c.data);

    const point = new Float32Array(points.numColumns);

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

        for (let c = 0; c < points.numColumns; ++c) {
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
}

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
            labels: new Array(points.numRows).fill(0).map((_, i) => i)
        };
    }

    const row: any = {};

    // construct centroids data table and assign initial values
    const centroids = new DataTable(points.columns.map(c => new Column(c.name, new Float32Array(k))));
    initializeCentroids(points, centroids, row);

    const gpuCluster = device && new GpuCluster(device, points, k);
    const labels = new Uint32Array(points.numRows);

    let converged = false;
    let steps = 0;

    while (!converged) {
        if (gpuCluster) {
            await gpuCluster.execute(centroids, labels);
        } else {
            clusterKdTreeCpu(points, centroids, labels);
        }

        // calculate the new centroid positions
        const groups = groupLabels(labels, k);
        for (let i = 0; i < centroids.numRows; ++i) {
            calcAverage(points, groups[i], row);
            centroids.setRow(i, row);
        }

        steps++;

        if (steps >= iterations) {
            converged = true;
        }

        stdout.write("#");
    }

    console.log(' done 🎉');

    return { centroids, labels };
};

export { kmeans };
