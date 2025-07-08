import { Column, DataTable } from '../data-table';

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

const findMinDistance = (dataTable: DataTable, row: any) => {
    let minDistance = Infinity;
    let index = -1;

    const keys = dataTable.columnNames;
    const dataRow: any = {};

    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(i, dataRow);

        // calculate squared distance
        let distance = 0;
        for (let j = 0; j < keys.length; ++j) {
            const key = keys[j];
            const diff = row[key] - dataRow[key];
            distance += diff * diff;
        }

        if (distance < minDistance) {
            minDistance = distance;
            index = i;
        }
    }

    return index;
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

    for (let i = 0; i < keys.length; ++i) {
        row[keys[i]] /= cluster.length;
    }
};

const kmeans = (dataTable: DataTable, k: number) => {
    // too few data points
    if (dataTable.numRows < k) {
        return {
            centroids: dataTable.clone(),
            labels: new Array(dataTable.numRows).fill(0).map((_, i) => i)
        };
    }

    const row = {};

    // construct a dataTable to hold the centroids
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

        // assign each point to the nearest centroid
        for (let i = 0; i < dataTable.numRows; ++i) {
            dataTable.getRow(i, row);
            const index = findMinDistance(centroids, row);
            clusters[index].push(i);
        }

        // calculate the new centroid positions
        for (let i = 0; i < k; ++i) {
            calcAverage(dataTable, clusters[i], row);
            centroids.setRow(i, row);
        }

        steps++;
        if (steps > 100) {
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
