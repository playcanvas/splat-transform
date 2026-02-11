import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from '../data-table/data-table.js';
import { logger } from '../utils/logger.js';
import type { Bounds } from './sparse-octree.js';

/**
 * Result of computing Gaussian extents.
 */
interface GaussianExtentsResult {
    /**
     * DataTable containing extent_x, extent_y, extent_z columns.
     * To compute AABB for Gaussian i:
     *   minX = x[i] - extent_x[i], maxX = x[i] + extent_x[i]
     *   minY = y[i] - extent_y[i], maxY = y[i] + extent_y[i]
     *   minZ = z[i] - extent_z[i], maxZ = z[i] + extent_z[i]
     */
    extents: DataTable;

    /** Scene bounds (union of all Gaussian AABBs) */
    sceneBounds: Bounds;

    /** Number of Gaussians skipped due to invalid values */
    invalidCount: number;
}

/**
 * Compute axis-aligned bounding box half-extents for all Gaussians in a DataTable.
 *
 * Each Gaussian is an oriented ellipsoid defined by position, rotation (quaternion),
 * and scale (log scale). This function computes the AABB that encloses each
 * rotated ellipsoid and stores only the half-extents. The full AABB can be
 * reconstructed at runtime using: min = position - extent, max = position + extent.
 *
 * @param dataTable - DataTable containing Gaussian splat data
 * @returns GaussianExtentsResult with extents DataTable and scene bounds
 */
const computeGaussianExtents = (dataTable: DataTable): GaussianExtentsResult => {
    const numRows = dataTable.numRows;

    // Get column data
    const x = dataTable.getColumnByName('x').data;
    const y = dataTable.getColumnByName('y').data;
    const z = dataTable.getColumnByName('z').data;
    const rx = dataTable.getColumnByName('rot_1').data;
    const ry = dataTable.getColumnByName('rot_2').data;
    const rz = dataTable.getColumnByName('rot_3').data;
    const rw = dataTable.getColumnByName('rot_0').data;
    const sx = dataTable.getColumnByName('scale_0').data;
    const sy = dataTable.getColumnByName('scale_1').data;
    const sz = dataTable.getColumnByName('scale_2').data;

    // Allocate output arrays
    const extentX = new Float32Array(numRows);
    const extentY = new Float32Array(numRows);
    const extentZ = new Float32Array(numRows);

    // Scene bounds
    const sceneMin = new Vec3(Infinity, Infinity, Infinity);
    const sceneMax = new Vec3(-Infinity, -Infinity, -Infinity);

    // Reusable objects to avoid allocations in the loop
    const position = new Vec3();
    const rotation = new Quat();
    const scale = new Vec3();
    const mat4 = new Mat4();

    // Local AABB centered at origin
    const localBox = new BoundingBox();
    localBox.center.set(0, 0, 0);

    // Transformed AABB
    const worldBox = new BoundingBox();

    let invalidCount = 0;

    for (let i = 0; i < numRows; i++) {
        // Get Gaussian properties
        position.set(x[i], y[i], z[i]);
        rotation.set(rx[i], ry[i], rz[i], rw[i]).normalize();
        scale.set(Math.exp(sx[i]), Math.exp(sy[i]), Math.exp(sz[i]));

        // Set local box half-extents to 3-sigma (Gaussians render out to 3-sigma)
        localBox.halfExtents.set(scale.x * 3, scale.y * 3, scale.z * 3);

        // Create rotation matrix (translation is included to position the AABB correctly)
        mat4.setTRS(position, rotation, Vec3.ONE);

        // Transform local AABB to world space AABB
        worldBox.setFromTransformedAabb(localBox, mat4);

        // Get the half-extents of the world-space AABB
        const halfExtents = worldBox.halfExtents;

        // Validate
        if (!isFinite(halfExtents.x) || !isFinite(halfExtents.y) || !isFinite(halfExtents.z)) {
            // Store zero extents for invalid Gaussians
            extentX[i] = 0;
            extentY[i] = 0;
            extentZ[i] = 0;
            invalidCount++;
            continue;
        }

        // Store half-extents
        extentX[i] = halfExtents.x;
        extentY[i] = halfExtents.y;
        extentZ[i] = halfExtents.z;

        // Update scene bounds (AABB = position +/- halfExtents)
        const minX = position.x - halfExtents.x;
        const minY = position.y - halfExtents.y;
        const minZ = position.z - halfExtents.z;
        const maxX = position.x + halfExtents.x;
        const maxY = position.y + halfExtents.y;
        const maxZ = position.z + halfExtents.z;

        if (minX < sceneMin.x) sceneMin.x = minX;
        if (minY < sceneMin.y) sceneMin.y = minY;
        if (minZ < sceneMin.z) sceneMin.z = minZ;
        if (maxX > sceneMax.x) sceneMax.x = maxX;
        if (maxY > sceneMax.y) sceneMax.y = maxY;
        if (maxZ > sceneMax.z) sceneMax.z = maxZ;
    }

    if (invalidCount > 0) {
        logger.warn(`Skipped ${invalidCount} Gaussians with invalid scale/rotation values`);
    }

    // Create DataTable with extent columns
    const extentsTable = new DataTable([
        new Column('extent_x', extentX),
        new Column('extent_y', extentY),
        new Column('extent_z', extentZ)
    ]);

    return {
        extents: extentsTable,
        sceneBounds: {
            min: sceneMin,
            max: sceneMax
        },
        invalidCount
    };
};

/**
 * Get the AABB for a specific Gaussian.
 *
 * @param extents - DataTable with extent_x, extent_y, extent_z columns
 * @param dataTable - DataTable containing position data (x, y, z)
 * @param index - Gaussian index
 * @param outMin - Output Vec3 for minimum corner
 * @param outMax - Output Vec3 for maximum corner
 */
const getGaussianAABB = (
    extents: DataTable,
    dataTable: DataTable,
    index: number,
    outMin: Vec3,
    outMax: Vec3
): void => {
    const x = dataTable.getColumnByName('x').data;
    const y = dataTable.getColumnByName('y').data;
    const z = dataTable.getColumnByName('z').data;

    const ex = extents.getColumnByName('extent_x').data;
    const ey = extents.getColumnByName('extent_y').data;
    const ez = extents.getColumnByName('extent_z').data;

    outMin.set(x[index] - ex[index], y[index] - ey[index], z[index] - ez[index]);
    outMax.set(x[index] + ex[index], y[index] + ey[index], z[index] + ez[index]);
};

/**
 * Check if a Gaussian's AABB overlaps with a given box.
 *
 * @param extents - DataTable with extent_x, extent_y, extent_z columns
 * @param dataTable - DataTable containing position data (x, y, z)
 * @param index - Gaussian index
 * @param boxMin - Minimum corner of query box
 * @param boxMax - Maximum corner of query box
 * @returns true if AABBs overlap
 */
const gaussianOverlapsBox = (
    extents: DataTable,
    dataTable: DataTable,
    index: number,
    boxMin: Vec3,
    boxMax: Vec3
): boolean => {
    const x = dataTable.getColumnByName('x').data;
    const y = dataTable.getColumnByName('y').data;
    const z = dataTable.getColumnByName('z').data;

    const ex = extents.getColumnByName('extent_x').data;
    const ey = extents.getColumnByName('extent_y').data;
    const ez = extents.getColumnByName('extent_z').data;

    const gMinX = x[index] - ex[index];
    const gMinY = y[index] - ey[index];
    const gMinZ = z[index] - ez[index];
    const gMaxX = x[index] + ex[index];
    const gMaxY = y[index] + ey[index];
    const gMaxZ = z[index] + ez[index];

    // AABB overlap test
    return !(gMaxX < boxMin.x || gMinX > boxMax.x ||
             gMaxY < boxMin.y || gMinY > boxMax.y ||
             gMaxZ < boxMin.z || gMinZ > boxMax.z);
};

export {
    computeGaussianExtents,
    getGaussianAABB,
    gaussianOverlapsBox
};

export type { GaussianExtentsResult };
