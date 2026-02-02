import { DataTable } from './data-table';
import { logger } from '../utils/logger.js';

/**
 * Sorts the provided indices by visibility score (descending order).
 *
 * Visibility is computed as: linear_opacity * volume
 * where:
 * - linear_opacity = sigmoid(opacity) = 1 / (1 + exp(-opacity))
 * - volume = exp(scale_0) * exp(scale_1) * exp(scale_2)
 *
 * After calling this function, indices[0] will contain the index of the most
 * visible splat, indices[1] the second most visible, and so on.
 *
 * @param dataTable - The DataTable containing splat data.
 * @param indices - Array of indices to sort in-place.
 */
const sortByVisibility = (dataTable: DataTable, indices: Uint32Array): void => {
    const opacityCol = dataTable.getColumnByName('opacity');
    const scale0Col = dataTable.getColumnByName('scale_0');
    const scale1Col = dataTable.getColumnByName('scale_1');
    const scale2Col = dataTable.getColumnByName('scale_2');

    if (!opacityCol || !scale0Col || !scale1Col || !scale2Col) {
        logger.debug('missing required columns for visibility sorting (opacity, scale_0, scale_1, scale_2)');
        return;
    }

    if (indices.length === 0) {
        return;
    }

    const opacity = opacityCol.data;
    const scale0 = scale0Col.data;
    const scale1 = scale1Col.data;
    const scale2 = scale2Col.data;

    // Compute visibility scores for each splat
    const scores = new Float32Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
        const ri = indices[i];

        // Convert logit opacity to linear using sigmoid
        const logitOpacity = opacity[ri];
        const linearOpacity = 1 / (1 + Math.exp(-logitOpacity));

        // Convert log scales to linear and compute volume
        // volume = exp(scale_0) * exp(scale_1) * exp(scale_2) = exp(scale_0 + scale_1 + scale_2)
        const volume = Math.exp(scale0[ri] + scale1[ri] + scale2[ri]);

        // Visibility score is opacity * volume
        scores[i] = linearOpacity * volume;
    }

    // Sort indices by score (descending - most visible first)
    const order = new Uint32Array(indices.length);
    for (let i = 0; i < order.length; i++) {
        order[i] = i;
    }
    order.sort((a, b) => scores[b] - scores[a]);

    // Apply the sorted order to indices
    const tmpIndices = indices.slice();
    for (let i = 0; i < indices.length; i++) {
        indices[i] = tmpIndices[order[i]];
    }
};

export { sortByVisibility };
