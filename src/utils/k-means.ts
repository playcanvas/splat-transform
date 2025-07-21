import { stdout } from 'node:process';

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

const cluster = (dataTable: DataTable, kdTree: KdTree) => {
    const k = kdTree.centroids.numRows;

    // construct a kdtree over the centroids so we can find the nearest quickly
    const point = new Float32Array(dataTable.numColumns);

    const clusters: number[][] = [];
    for (let i = 0; i < k; ++i) {
        clusters[i] = [];
    }

    const row: any = {};

    let atot = 0;
    let btot = 0;
    let ctot = 0;

    // assign each point to the nearest centroid
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(i, row);
        dataTable.columns.forEach((c, i) => {
            point[i] = row[c.name];
        });

        const a = kdTree.findNearest(point);
        const b = kdTree.findNearest2(point);
        const c = kdTree.findNearest3(point);

        atot += a.cnt;
        btot += b.cnt;
        ctot += c.cnt;

        clusters[a.index].push(i);
    }

    console.log(`atot=${atot} btot=${btot} ctot=${ctot}`);

    return clusters;
};

const kmeans = async (dataTable: DataTable, k: number, device?: GpuDevice) => {
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

    let converged = false;
    let steps = 0;
    let clusters: number[][];

    while (!converged) {
        const kdTree = new KdTree(centroids);

        /*
        // compare cpu and gpu results
        const a = await device.cluster.execute(dataTable, kdTree);
        const b = cluster(dataTable, kdTree);

        for (let i = 0; i < a.length; ++i) {
            if (a[i].length !== b[i].length) {
                console.log(`Cluster mismatch at index ${i}: GPU has ${a[i].length} points, CPU has ${b[i].length} points`);
            } else {
                for (let j = 0; j < a[i].length; ++j) {
                    if (a[i][j] !== b[i][j]) {
                        console.log(`Point mismatch at cluster ${i}, point index ${j}: GPU has ${a[i][j]}, CPU has ${b[i][j]}`);
                    }
                }
            }
        }
        //*/

        clusters = device ? await device.cluster.execute(dataTable, centroids) : cluster(dataTable, kdTree);

        // calculate the new centroid positions
        for (let i = 0; i < centroids.numRows; ++i) {
            calcAverage(dataTable, clusters[i], row);
            centroids.setRow(i, row);
        }

        steps++;
        if (steps > 10) {
            converged = true;
        }

        stdout.write("#");
    }

    console.log(' done 🎉');

    // construct labels from clusters
    const labels = new Uint32Array(dataTable.numRows);
    for (let i = 0; i < clusters.length; ++i) {
        const cluster = clusters[i];
        for (let j = 0; j < cluster.length; ++j) {
            labels[cluster[j]] = i;
        }
    }

    return { centroids, labels };
};

export { kmeans };
