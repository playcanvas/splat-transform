import { Vec3 } from 'playcanvas';

import { DataTable, type TypedArray } from '../data-table';
import { quickselect } from '../utils';

/**
 * Axis-aligned bounding box for BVH nodes.
 */
interface BVHBounds {
    minX: number;
    minY: number;
    minZ: number;
    maxX: number;
    maxY: number;
    maxZ: number;
}

/**
 * BVH node for Gaussian AABBs.
 */
interface GaussianBVHNode {
    /** Number of Gaussians in this subtree */
    count: number;

    /** Bounds of all Gaussian AABBs in this subtree */
    bounds: BVHBounds;

    /** Gaussian indices (only for leaf nodes) */
    indices?: Uint32Array;

    /** Left child (only for interior nodes) */
    left?: GaussianBVHNode;

    /** Right child (only for interior nodes) */
    right?: GaussianBVHNode;
}

/**
 * Check if two AABBs overlap.
 *
 * @param a - First AABB (BVHBounds object)
 * @param bMinX - Second AABB minimum X
 * @param bMinY - Second AABB minimum Y
 * @param bMinZ - Second AABB minimum Z
 * @param bMaxX - Second AABB maximum X
 * @param bMaxY - Second AABB maximum Y
 * @param bMaxZ - Second AABB maximum Z
 * @returns True if the AABBs overlap
 */
const boundsOverlap = (a: BVHBounds, bMinX: number, bMinY: number, bMinZ: number, bMaxX: number, bMaxY: number, bMaxZ: number): boolean => {
    return !(a.maxX < bMinX || a.minX > bMaxX ||
             a.maxY < bMinY || a.minY > bMaxY ||
             a.maxZ < bMinZ || a.minZ > bMaxZ);
};

/**
 * BVH (Bounding Volume Hierarchy) for efficient spatial queries on Gaussian AABBs.
 *
 * Unlike the centroid-based BTree, this BVH stores the full AABB for each node,
 * computed from position +/- extent for all Gaussians in the subtree.
 */
class GaussianBVH {
    /** Root node of the BVH */
    root: GaussianBVHNode;

    /** Position data from the original DataTable */
    private x: TypedArray;
    private y: TypedArray;
    private z: TypedArray;

    /** Extent data */
    private extentX: TypedArray;
    private extentY: TypedArray;
    private extentZ: TypedArray;

    /** Maximum leaf size */
    private static readonly MAX_LEAF_SIZE = 64;

    /**
     * Construct a BVH from Gaussian data.
     *
     * @param dataTable - DataTable containing position (x, y, z) columns
     * @param extents - DataTable containing extent (extent_x, extent_y, extent_z) columns
     */
    constructor(dataTable: DataTable, extents: DataTable) {
        // Cache column data for fast access
        this.x = dataTable.getColumnByName('x').data;
        this.y = dataTable.getColumnByName('y').data;
        this.z = dataTable.getColumnByName('z').data;
        this.extentX = extents.getColumnByName('extent_x').data;
        this.extentY = extents.getColumnByName('extent_y').data;
        this.extentZ = extents.getColumnByName('extent_z').data;

        const numRows = dataTable.numRows;
        const indices = new Uint32Array(numRows);
        for (let i = 0; i < numRows; i++) {
            indices[i] = i;
        }

        this.root = this.buildNode(indices);
    }

    /**
     * Compute the AABB that bounds all Gaussian AABBs for the given indices.
     *
     * @param indices - Array of Gaussian indices
     * @returns The bounding box that contains all specified Gaussians
     */
    private computeBounds(indices: Uint32Array): BVHBounds {
        const { x, y, z, extentX, extentY, extentZ } = this;

        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;

        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            const gMinX = x[idx] - extentX[idx];
            const gMinY = y[idx] - extentY[idx];
            const gMinZ = z[idx] - extentZ[idx];
            const gMaxX = x[idx] + extentX[idx];
            const gMaxY = y[idx] + extentY[idx];
            const gMaxZ = z[idx] + extentZ[idx];

            if (gMinX < minX) minX = gMinX;
            if (gMinY < minY) minY = gMinY;
            if (gMinZ < minZ) minZ = gMinZ;
            if (gMaxX > maxX) maxX = gMaxX;
            if (gMaxY > maxY) maxY = gMaxY;
            if (gMaxZ > maxZ) maxZ = gMaxZ;
        }

        return { minX, minY, minZ, maxX, maxY, maxZ };
    }

    /**
     * Build a BVH node recursively.
     *
     * @param indices - Array of Gaussian indices to include in this subtree
     * @returns The constructed BVH node
     */
    private buildNode(indices: Uint32Array): GaussianBVHNode {
        const bounds = this.computeBounds(indices);

        // Create leaf node if small enough
        if (indices.length <= GaussianBVH.MAX_LEAF_SIZE) {
            return {
                count: indices.length,
                bounds,
                indices
            };
        }

        // Find the largest axis to split on (based on centroid positions for better balance)
        const { x, y, z } = this;
        let centroidMinX = Infinity, centroidMaxX = -Infinity;
        let centroidMinY = Infinity, centroidMaxY = -Infinity;
        let centroidMinZ = Infinity, centroidMaxZ = -Infinity;

        for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            const px = x[idx];
            const py = y[idx];
            const pz = z[idx];

            if (px < centroidMinX) centroidMinX = px;
            if (px > centroidMaxX) centroidMaxX = px;
            if (py < centroidMinY) centroidMinY = py;
            if (py > centroidMaxY) centroidMaxY = py;
            if (pz < centroidMinZ) centroidMinZ = pz;
            if (pz > centroidMaxZ) centroidMaxZ = pz;
        }

        const extX = centroidMaxX - centroidMinX;
        const extY = centroidMaxY - centroidMinY;
        const extZ = centroidMaxZ - centroidMinZ;

        // Choose axis with largest extent
        let splitAxis: TypedArray;
        if (extX >= extY && extX >= extZ) {
            splitAxis = x;
        } else if (extY >= extZ) {
            splitAxis = y;
        } else {
            splitAxis = z;
        }

        // Partition around median
        const mid = indices.length >>> 1;
        quickselect(splitAxis, indices, mid);

        // Recursively build children
        const left = this.buildNode(indices.subarray(0, mid));
        const right = this.buildNode(indices.subarray(mid));

        return {
            count: left.count + right.count,
            bounds,
            left,
            right
        };
    }

    /**
     * Query all Gaussian indices whose AABBs overlap the given box.
     *
     * @param boxMin - Minimum corner of query box
     * @param boxMax - Maximum corner of query box
     * @returns Array of Gaussian indices that overlap the box
     */
    queryOverlapping(boxMin: Vec3, boxMax: Vec3): number[] {
        const result: number[] = [];
        this.queryNode(this.root, boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z, result);
        return result;
    }

    /**
     * Query all Gaussian indices whose AABBs overlap the given box (using raw coordinates).
     *
     * @param minX - Minimum X of query box
     * @param minY - Minimum Y of query box
     * @param minZ - Minimum Z of query box
     * @param maxX - Maximum X of query box
     * @param maxY - Maximum Y of query box
     * @param maxZ - Maximum Z of query box
     * @returns Array of Gaussian indices that overlap the box
     */
    queryOverlappingRaw(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): number[] {
        const result: number[] = [];
        this.queryNode(this.root, minX, minY, minZ, maxX, maxY, maxZ, result);
        return result;
    }

    /**
     * Recursive query helper.
     *
     * @param node - Current BVH node to query
     * @param minX - Query box minimum X
     * @param minY - Query box minimum Y
     * @param minZ - Query box minimum Z
     * @param maxX - Query box maximum X
     * @param maxY - Query box maximum Y
     * @param maxZ - Query box maximum Z
     * @param result - Array to append matching indices to
     */
    private queryNode(
        node: GaussianBVHNode,
        minX: number,
        minY: number,
        minZ: number,
        maxX: number,
        maxY: number,
        maxZ: number,
        result: number[]
    ): void {
        // Early exit if node bounds don't overlap query box
        if (!boundsOverlap(node.bounds, minX, minY, minZ, maxX, maxY, maxZ)) {
            return;
        }

        // Leaf node: check each Gaussian individually
        if (node.indices) {
            const { x, y, z, extentX, extentY, extentZ } = this;

            for (let i = 0; i < node.indices.length; i++) {
                const idx = node.indices[i];
                const gMinX = x[idx] - extentX[idx];
                const gMinY = y[idx] - extentY[idx];
                const gMinZ = z[idx] - extentZ[idx];
                const gMaxX = x[idx] + extentX[idx];
                const gMaxY = y[idx] + extentY[idx];
                const gMaxZ = z[idx] + extentZ[idx];

                // Check overlap
                if (!(gMaxX < minX || gMinX > maxX ||
                      gMaxY < minY || gMinY > maxY ||
                      gMaxZ < minZ || gMinZ > maxZ)) {
                    result.push(idx);
                }
            }
            return;
        }

        // Interior node: recurse into children
        if (node.left) {
            this.queryNode(node.left, minX, minY, minZ, maxX, maxY, maxZ, result);
        }
        if (node.right) {
            this.queryNode(node.right, minX, minY, minZ, maxX, maxY, maxZ, result);
        }
    }

    /**
     * Get the total number of Gaussians in the BVH.
     *
     * @returns The total count of Gaussians
     */
    get count(): number {
        return this.root.count;
    }

    /**
     * Get the bounds of the entire scene.
     *
     * @returns The scene bounding box
     */
    get sceneBounds(): BVHBounds {
        return this.root.bounds;
    }
}

export { GaussianBVH, GaussianBVHNode, BVHBounds };
