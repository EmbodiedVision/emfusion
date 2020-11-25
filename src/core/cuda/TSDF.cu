/*
 * This file is part of EM-Fusion.
 *
 * Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://emfusion.is.tue.mpg.de>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 *
 * EM-Fusion is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * EM-Fusion is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with EM-Fusion.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "EMFusion/core/cuda/TSDF.cuh"

namespace emf {
namespace cuda {
namespace TSDF {

// Lookup tables for marching cubes algorithm.
__device__
const int edgeTable[256] = {
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

__device__
const int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};


__global__
void kernel_updateTSDF ( const cv::cuda::PtrStepSz<float> depth,
                         const cv::cuda::PtrStep<float> assocWeights,
                         cv::cuda::PtrStep<float> tsdfVol,
                         cv::cuda::PtrStep<float> tsdfWeigths,
                         const float33 rot_OC, const float3 trans_OC,
                         const float33 intr, const int3 volSize,
                         const float voxelSize, const float truncdist,
                         const float maxWeight ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= volSize.x || y_ >= volSize.y * volSize.z )
        return;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    const float3 pos_obj = make_float3 (
                               ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                               ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                               ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize );
    const float3 pos_cam = rot_OC * pos_obj + trans_OC;

    if ( pos_cam.z <= 0.f ) {
        // Avoid artifacts during raycasting if these voxels are not yet valid
        if ( tsdfWeigths ( y_, x ) == 0 )
            tsdfVol ( y_, x ) = 0;
        return;
    }

    const float3 proj = intr * pos_cam;

    const int2 pix = make_int2 ( __float2int_rn ( proj.x / proj.z ),
                                 __float2int_rn ( proj.y / proj.z ) );

    if ( pix.x < 0 || pix.x >= depth.cols || pix.y < 0 || pix.y >= depth.rows )
        return;

    const float depthVal = depth ( pix.y, pix.x );
    if ( depthVal <= 0.f ) {
        // Avoid artifacts during raycasting if these voxels are not yet valid
        if ( tsdfWeigths ( y_, x ) == 0 )
            tsdfVol ( y_, x ) = 0;
        return;
    }

    const float lambda = norm ( make_float3 (
                                    ( pix.x - intr ( 0, 2 ) ) / intr ( 0, 0 ),
                                    ( pix.y - intr ( 1, 2 ) ) / intr ( 1, 1 ),
                                    1.f ) );

    const float sdf = depthVal - ( 1.f / lambda ) * norm ( pos_cam );

    const float prev_weight = tsdfWeigths ( y_, x );
    if ( sdf >= -truncdist ) {
        const float tsdf = copysignf ( fminf ( 1.f, abs ( sdf / truncdist ) ),
                                       sdf );

        const float prev_tsdf = tsdfVol ( y_, x );
        const float assoc_weight = sdf < truncdist ?
                                   assocWeights ( pix.y, pix.x ) : 1.f;

        const float new_weight = assoc_weight;

        if ( prev_weight + new_weight > 0 ) {
            tsdfVol ( y_, x ) = ( prev_weight * prev_tsdf + new_weight * tsdf )
                                / ( prev_weight + new_weight );
            tsdfWeigths ( y_, x ) = fminf ( prev_weight + new_weight,
                                            maxWeight );
        }
    } else if ( prev_weight == 0 ) {
        tsdfVol ( y_, x ) = -1;
    }
}

void updateTSDF ( const cv::cuda::GpuMat& depth,
                  const cv::cuda::GpuMat& assocWeights,
                  cv::cuda::GpuMat& tsdfVol, cv::cuda::GpuMat& tsdfWeights,
                  const cv::Matx33f& rel_rot_OC, const cv::Vec3f& rel_trans_OC,
                  const cv::Matx33f& intr, const cv::Vec3i& volumeRes,
                  const float voxelSize, const float truncdist,
                  const float maxWeight, cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( tsdfVol.cols + threads.x - 1 ) / threads.x,
                  ( tsdfVol.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    const float33 rot = * ( float33 * ) rel_rot_OC.val;
    const float3 trans = * ( float3 * ) rel_trans_OC.val;

    const float33 camIntr = * ( float33 * ) intr.val;

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_updateTSDF<<<blocks, threads, 0, stream_cu>>> (
                        depth, assocWeights, tsdfVol, tsdfWeights, rot, trans,
                        camIntr, volSize, voxelSize, truncdist, maxWeight );
}

__global__
void kernel_computeTSDFGrads ( const cv::cuda::PtrStepSz<float> tsdfVol,
                               cv::cuda::PtrStep<float3> tsdfGrads,
                               const int3 volSize ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    if ( x >= volSize.x - 1 || y >= volSize.y - 1 || z >= volSize.z - 1 )
        return;

    const float tsdfVal = tsdfVol ( y_, x );

    tsdfGrads ( y_, x ) = make_float3 (
                              tsdfVol ( y_, x + 1 ) - tsdfVal,
                              tsdfVol ( y_ + 1, x ) - tsdfVal,
                              tsdfVol ( y_ + volSize.y, x ) - tsdfVal );
}

void computeTSDFGrads ( const cv::cuda::GpuMat& tsdfVol,
                        cv::cuda::GpuMat& tsdfGrads, const cv::Vec3i& volumeRes,
                        cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( tsdfVol.cols + threads.x - 1 ) / threads.x,
                  ( tsdfVol.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_computeTSDFGrads<<<blocks, threads, 0, stream_cu>>> (
        tsdfVol, tsdfGrads, volSize );
}

__global__
void kernel_raycastTSDF ( const cv::cuda::PtrStep<float> tsdfVol,
                          const cv::cuda::PtrStep<float3> tsdfGrads,
                          const cv::cuda::PtrStep<float> tsdfWeights,
                          cv::cuda::PtrStepSz<float> raylengths,
                          cv::cuda::PtrStep<float3> vertices,
                          cv::cuda::PtrStep<float3> normals,
                          cv::cuda::PtrStep<bool> mask, const float33 rot_CO,
                          const float3 trans_CO, const float33 intr,
                          const int3 volSize, const float voxelSize,
                          const float truncdist ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 0 || x >= raylengths.cols || y < 0 || y >= raylengths.rows )
        return;

    const float3 unproj = make_float3 ( ( x - intr ( 0, 2 ) ) / intr ( 0, 0 ),
                                        ( y - intr ( 1, 2 ) ) / intr ( 1, 1 ),
                                        1.f );

    const float3 ray = rot_CO * unproj;
    const float3 dir = ray / norm ( ray );

    const float3 boxBounds = ( volSize - 1 ) / 2 * voxelSize;

    float raylength = enterVolStep ( dir, trans_CO, boxBounds );
    float maxRaylength = exitVolStep ( dir, trans_CO, boxBounds );

    // For not searching past ray in different volume
    const float old_raylength = raylengths ( y, x );
    raylength += voxelSize; // Avoid memory access outside volume
    maxRaylength -= voxelSize; // Avoid memory access outside volume
    if ( old_raylength != 0 )
        maxRaylength = min ( old_raylength, maxRaylength );

    if ( raylength >= maxRaylength ) // ray not intersecting volume
        return;

    float raystep = truncdist;

    float3 v = ( trans_CO + dir * raylength ) / voxelSize
               + ( volSize - 1 ) / 2.f;
    while ( ( v.x < 0 || v.x + 1 >= volSize.x ||
              v.y < 0 || v.y + 1 >= volSize.y ||
              v.z < 0 || v.z + 1 >= volSize.z ) && raylength < maxRaylength ) {
        raylength += raystep;
        v = ( trans_CO + dir * raylength ) / voxelSize + ( volSize - 1 ) / 2.f;
    }

    float tsdf = interpolateTrilinear ( tsdfVol, v, volSize );

    if ( abs ( tsdf ) < 1.f )
        raystep = voxelSize;
    if ( abs ( tsdf ) < .8f )
        raystep = 0.5f * voxelSize;

    while ( ( raylength += raystep ) <= maxRaylength ) {
        v = ( trans_CO + raylength * dir ) / voxelSize + ( volSize - 1 ) / 2.f;
        if ( v.x < 0 || v.x + 2 >= volSize.x ||
                v.y < 0 || v.y + 2 >= volSize.y ||
                v.z < 0 || v.z + 2 >= volSize.z )
            continue;

        float next_tsdf = interpolateTrilinear ( tsdfVol, v, volSize );
        float tsdfWeight = interpolateTrilinear ( tsdfWeights, v, volSize );

        // Zero-crossing from behind
        if ( tsdf < 0 && next_tsdf > 0 && tsdfWeight > 0.f )
            break;

        if ( abs ( next_tsdf ) < 1.f )
            raystep = voxelSize;
        if ( abs ( next_tsdf ) < .8f )
            raystep = 0.5f * voxelSize;

        if ( tsdf > 0 && next_tsdf < 0 ) { // Surface interface found
            const float t_star = raylength - raystep * tsdf
                                 / ( next_tsdf - tsdf );

            const float3 v_star = ( trans_CO + t_star * dir ) / voxelSize
                                  + ( volSize - 1 ) / 2.f;
            if ( v_star.x < 0 || v_star.x + 2 >= volSize.x ||
                    v_star.y < 0 || v_star.y + 2 >= volSize.y ||
                    v_star.z < 0 || v_star.z + 2 >= volSize.z )
                continue;

            tsdfWeight = interpolateTrilinear ( tsdfWeights, v_star, volSize );

            if ( tsdfWeight > 0.f ) {
                const float3 grad_tsdf =
                    interpolateTrilinear ( tsdfGrads, v_star, volSize );

                raylengths ( y, x ) = t_star;

                // Convert to camera coordinates
                const float33 rot_OC = transpose ( rot_CO );
                vertices ( y, x ) = rot_OC * ( t_star * dir );
                normals ( y, x ) = rot_OC * ( grad_tsdf / norm ( grad_tsdf ) );
                mask ( y, x ) = true;

                break;
            }
        }

        tsdf = next_tsdf;
    }
}

void raycastTSDF ( const cv::cuda::GpuMat& tsdfVol,
                   const cv::cuda::GpuMat& tsdfGrads,
                   const cv::cuda::GpuMat& tsdfWeights,
                   cv::cuda::GpuMat& raylengths, cv::cuda::GpuMat& vertices,
                   cv::cuda::GpuMat& normals, cv::cuda::GpuMat& mask,
                   const cv::Matx33f& rel_rot_CO, const cv::Vec3f& rel_trans_CO,
                   const cv::Matx33f& intr, const cv::Vec3i& volumeRes,
                   const float voxelSize, const float truncdist,
                   cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 16, 16 );
    dim3 blocks ( ( raylengths.cols + threads.x - 1 ) / threads.x,
                  ( raylengths.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    const float33 rot = * ( float33 * ) rel_rot_CO.val;
    const float3 trans = * ( float3 * ) rel_trans_CO.val;

    const float33 camIntr = * ( float33 * ) intr.val;

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_raycastTSDF<<<blocks, threads, 0, stream_cu>>> (
        tsdfVol, tsdfGrads, tsdfWeights, raylengths, vertices, normals,
        mask, rot, trans, camIntr, volSize, voxelSize, truncdist );
}

__global__
void kernel_computePoseGradients ( const cv::cuda::PtrStep<float3> tsdfGrads,
                                   const cv::cuda::PtrStepSz<float3> points,
                                   cv::cuda::PtrStep<float> grads,
                                   const float33 rotmat, const float3 trans,
                                   const int3 volSize, const float voxelSize ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 0 || x >= points.cols || y < 0 || y >= points.rows )
        return;

    const float3 p_cam = points ( y, x );
    if ( p_cam.z <= 0 )
        return;

    const float3 p = rotmat * p_cam + trans;
    const float3 v_obj = p / voxelSize + ( volSize - 1 ) / 2.f;

    if ( v_obj.x < 0.f || v_obj.x + 2 >= volSize.x || v_obj.y < 0.f ||
            v_obj.y + 2 >= volSize.y || v_obj.z < 0.f ||
            v_obj.z + 2 >= volSize.z )
        return;

    const float3 grad_tsdf = interpolateTrilinear ( tsdfGrads, v_obj, volSize )
                             / voxelSize;
    const float3 grad_r = make_float33 ( 0.f, -p.z,  p.y,
                                         p.z,  0.f, -p.x,
                                         -p.y,  p.x,  0.f ) * grad_tsdf;

    float *gs = grads.ptr ( y * points.cols + x );
    * ( float3 * ) gs = grad_tsdf;
    * ( float3 * ) ( gs + 3 ) = grad_r;
}

void computePoseGradients ( const cv::cuda::GpuMat& tsdfGrads,
                            const cv::cuda::GpuMat& points,
                            const cv::Matx33f& rel_rot_CO,
                            const cv::Vec3f& rel_trans_CO,
                            const cv::Vec3i& volumeRes, const float voxelSize,
                            cv::cuda::GpuMat& grads,
                            cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( points.cols + threads.x - 1 ) / threads.x,
                  ( points.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;
    const float33 rot = * ( float33 * ) rel_rot_CO.val;
    const float3 trans = * ( float3 * ) rel_trans_CO.val;

    grads.setTo ( cv::Scalar::all ( 0 ), stream );

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_computePoseGradients<<<blocks, threads, 0, stream_cu>>> (
        tsdfGrads, points, grads, rot, trans, volSize, voxelSize );
}

template<typename T>
__global__
void kernel_getVolumeVals ( const cv::cuda::PtrStepSz<T> vol,
                            const cv::cuda::PtrStepSz<float3> points,
                            cv::cuda::PtrStep<T> vals, const float33 rotmat,
                            const float3 trans, const int3 volSize,
                            const float voxelSize ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 0 || x >= points.cols || y < 0 || y >= points.rows )
        return;

    const float3 p_cam = points ( y, x );
    if ( p_cam.z <= 0 )
        return;

    const float3 p = rotmat * p_cam + trans;
    const float3 v_obj = p / voxelSize + ( volSize - 1 ) / 2.f;

    if ( v_obj.x < 0.f || v_obj.x + 1 >= volSize.x ||
            v_obj.y < 0.f || v_obj.y + 1 >= volSize.y ||
            v_obj.z < 0.f || v_obj.z + 1 >= volSize.z )
        return;

    vals ( y, x ) = interpolateTrilinear ( vol, v_obj, volSize );
}

void getVolumeVals ( const cv::cuda::GpuMat& vol,
                     const cv::cuda::GpuMat& points,
                     const cv::Matx33f& rel_rot_CO,
                     const cv::Vec3f& rel_trans_CO, const cv::Vec3i& volumeRes,
                     const float voxelSize, cv::cuda::GpuMat& vals,
                     cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( points.cols + threads.x - 1 ) / threads.x,
                  ( points.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;
    const float33 rot = * ( float33 * ) rel_rot_CO.val;
    const float3 trans = * ( float3 * ) rel_trans_CO.val;

    vals.setTo ( cv::Scalar::all ( 0 ), stream );

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    assert ( vol.depth() == CV_32F );
    assert ( vol.channels() <= 3 );

    switch ( vol.channels() ) {
    case 1:
        kernel_getVolumeVals<float><<<blocks, threads, 0, stream_cu>>> (
            vol, points, vals, rot, trans, volSize, voxelSize );
        break;
    case 2:
        kernel_getVolumeVals<float2><<<blocks, threads, 0, stream_cu>>> (
            vol, points, vals, rot, trans, volSize, voxelSize );
        break;
    case 3:
        kernel_getVolumeVals<float3><<<blocks, threads, 0, stream_cu>>> (
            vol, points, vals, rot, trans, volSize, voxelSize );
        break;
    }
}


__global__
void kernel_computeAb ( const cv::cuda::PtrStepSz<float> grads,
                        const cv::cuda::PtrStep<float> tsdfVals,
                        cv::cuda::PtrStep<float> As,
                        cv::cuda::PtrStep<float> bs ) {
    const int Aidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idx < 0 || idx >= grads.rows || Aidx < 0 ||
            Aidx >= grads.cols * grads.cols )
        return;

    const int gidx = Aidx / grads.cols;
    const int i = Aidx % grads.cols;

    const float* const gs = grads.ptr ( idx );

    As.ptr ( idx ) [Aidx] = gs[gidx] * gs[i];

    if ( i > 0 )
        return;

    bs.ptr ( idx ) [gidx] = tsdfVals.ptr() [idx] * gs[gidx];
}

void computeAb ( const cv::cuda::GpuMat& grads,
                 const cv::cuda::GpuMat& tsdfVals, cv::cuda::GpuMat& As,
                 cv::cuda::GpuMat& bs, cv::cuda::Stream& stream ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 8, 32 );
    dim3 blocks ( ( As.cols + threads.x - 1 ) / threads.x,
                  ( grads.rows + threads.y - 1 ) / threads.y );

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_computeAb<<<blocks, threads, 0, stream_cu>>> (
        grads, tsdfVals.reshape ( 1, 1 ), As, bs );
}

template<typename T>
__global__
void kernel_copyValues ( const cv::cuda::PtrStepSz<T> src,
                         cv::cuda::PtrStep<T> dst, const int3 offset,
                         const int3 srcRes, const int3 dstRes ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 0 || x >= src.cols || y_ < 0 || y_ >= src.rows )
        return;

    const int y = y_ % srcRes.y;
    const int z = y_ / srcRes.y;

    const int x_new = x - offset.x;
    const int y_new = y - offset.y;
    const int z_new = z - offset.z;

    if ( x_new < 0 || x_new >= dstRes.x || y_new < 0 || y_new >= dstRes.y
            || z_new < 0 || z_new >= dstRes.z )
        return;

    dst ( z_new * dstRes.y + y_new, x_new ) = src ( y_, x );
}

void copyValues ( const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                  const cv::Vec3i& offset, const cv::Vec3i& srcRes,
                  const cv::Vec3i& dstRes ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( src.cols + threads.x - 1 ) / threads.x,
                  ( src.rows + threads.y - 1 ) / threads.y );

    const int3 offset_kernel = * ( int3 * ) offset.val;
    const int3 srcVolSize = * ( int3 * ) srcRes.val;
    const int3 dstVolSize = * ( int3 * ) dstRes.val;

    switch ( src.channels() ) {
    case 1:
        kernel_copyValues<float><<<blocks, threads>>> (
            src, dst, offset_kernel, srcVolSize, dstVolSize );
        break;
    case 2:
        kernel_copyValues<float2><<<blocks, threads>>> (
            src, dst, offset_kernel, srcVolSize, dstVolSize );
        break;
    case 3:
        kernel_copyValues<float3><<<blocks, threads>>> (
            src, dst, offset_kernel, srcVolSize, dstVolSize );
        break;
    }
}

__global__
void kernel_multSingleton ( const cv::cuda::PtrStepSz<float> m,
                            const cv::cuda::PtrStep<float> p,
                            cv::cuda::PtrStep<float> mo ) {
    const int midx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idx < 0 || idx >= m.rows || midx < 0 || midx >= m.cols )
        return;

    mo ( idx, midx ) = m ( idx, midx ) * p.ptr() [idx];
}

void multSingletonCol ( const cv::cuda::GpuMat& src1,
                        const cv::cuda::GpuMat& src2, cv::cuda::GpuMat& dst,
                        cv::cuda::Stream& stream ) {
    assert ( src1.cols == 1 || src2.cols == 1 );
    assert ( src1.rows == src2.rows && src1.rows == dst.rows &&
             max ( src1.cols, src2.cols ) == dst.cols );

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    dim3 threads ( 8, 32 );
    dim3 blocks ( ( max ( src1.cols, src2.cols ) + threads.x - 1 ) / threads.x,
                  ( src1.rows + threads.y - 1 ) / threads.y );

    if ( src1.cols == 1 )
        kernel_multSingleton<<<blocks, threads, 0, stream_cu>>> (
            src2, src1.reshape ( 1, 1 ), dst );
    else
        kernel_multSingleton<<<blocks, threads, 0, stream_cu>>> (
            src1, src2.reshape ( 1, 1 ), dst );
}

inline __device__
int countVerts ( int cubeClass ) {
    int numVerts = 0;
    int edgeClass = edgeTable[cubeClass];
    while ( edgeClass ) {
        numVerts += ( edgeClass & 1 );
        edgeClass >>= 1;
    }
    return numVerts;
}

inline __device__
int countTris ( int cubeClass ) {
    int numTris = 0;
    for ( int i = 0; triTable[cubeClass][i] != -1; i += 3 )
        ++numTris;
    return numTris;
}

__global__
void kernel_classifyCubes ( const cv::cuda::PtrStepSz<float> tsdf,
                            const cv::cuda::PtrStep<bool> mask, int3 volSize,
                            cv::cuda::PtrStep<uchar> cubeClasses,
                            cv::cuda::PtrStep<int> vertIdxBuffer,
                            cv::cuda::PtrStep<int> triIdxBuffer ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    if ( x >= volSize.x - 1 || y >= volSize.y - 1 || z >= volSize.z - 1 )
        return;

    for ( int i = 0; i < 8; ++i )
        if ( !mask ( y_ + volSize.y * ( ( i >> 2 ) & 1 ) + ( ( i >> 1 ) & 1 ),
                     x + ( i & 1 ) ) )
            return;

    uchar cubeClass = 0;
    cubeClass |= ( tsdf ( y_, x ) < 0.f );
    cubeClass |= ( tsdf ( y_, x + 1 ) < 0.f ) << 1;
    cubeClass |= ( tsdf ( y_ + volSize.y, x + 1 ) < 0.f ) << 2;
    cubeClass |= ( tsdf ( y_ + volSize.y, x ) < 0.f ) << 3;
    cubeClass |= ( tsdf ( y_ + 1, x ) < 0.f ) << 4;
    cubeClass |= ( tsdf ( y_ + 1, x + 1 ) < 0.f ) << 5;
    cubeClass |= ( tsdf ( y_ + volSize.y + 1, x + 1 ) < 0.f ) << 6;
    cubeClass |= ( tsdf ( y_ + volSize.y + 1, x ) < 0.f ) << 7;

    cubeClasses ( z * ( volSize.y - 1 ) + y, x ) = cubeClass;
    vertIdxBuffer ( z * ( volSize.y - 1 ) + y, x ) = countVerts ( cubeClass );
    triIdxBuffer ( z * ( volSize.y - 1 ) + y, x ) = 4 * countTris ( cubeClass );
}

__device__
float3 vertexInterp ( float3 p1, float3 p2, float val1, float val2 ) {
    if ( fabs ( val1 ) < 0.00001 )
        return p1;
    if ( fabs ( val2 ) < 0.00001 )
        return p2;
    if ( fabs ( val1 - val2 ) < 0.00001 )
        return p1;

    const float mu = -val1 / ( val2 - val1 );
    return p1 + mu * ( p2 - p1 );
}

__global__
void kernel_createTriangles ( const cv::cuda::PtrStepSz<float> tsdf,
                              const cv::cuda::PtrStepSz<float3> grads,
                              const cv::cuda::PtrStep<uchar> cubeClasses,
                              const int3 volSize, const float voxelSize,
                              const cv::cuda::PtrStep<int> vertIdxBuffer,
                              const cv::cuda::PtrStep<int> triIdxBuffer,
                              cv::cuda::PtrStep<float3> vertices,
                              cv::cuda::PtrStep<float3> normals,
                              cv::cuda::PtrStep<int> triangles ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    if ( x >= volSize.x - 1 || y >= volSize.y - 1 || z >= volSize.z - 1 )
        return;

    const uchar cubeClass = cubeClasses ( z * ( volSize.y - 1 ) + y, x );
    if ( edgeTable[cubeClass] == 0 )
        return;

    const float3 ps[8] = {
        make_float3 ( ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x + 1 - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x + 1 - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z + 1 - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z + 1 - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y + 1 - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x + 1 - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y + 1 - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x + 1 - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y + 1 - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z + 1 - ( volSize.z - 1 ) / 2.f ) * voxelSize ),
        make_float3 ( ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                      ( y + 1 - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                      ( z + 1 - ( volSize.z - 1 ) / 2.f ) * voxelSize )
    };
    const float3 ns[8] = {
        grads ( y_, x ), grads ( y_, x + 1 ), grads ( y_ + volSize.y, x + 1 ),
        grads ( y_ + volSize.y, x ), grads ( y_ + 1, x ), grads ( y_ + 1, x + 1 ),
        grads ( y_ + volSize.y + 1, x + 1 ), grads ( y_ + volSize.y + 1, x )
    };
    for ( int i = 0; i < 8; ++i )
        ns[i] /= norm ( ns[i] );

    const float vals[8] = {
        tsdf ( y_, x ), tsdf ( y_, x + 1 ), tsdf ( y_ + volSize.y, x + 1 ),
        tsdf ( y_ + volSize.y, x ), tsdf ( y_ + 1, x ), tsdf ( y_ + 1, x + 1 ),
        tsdf ( y_ + volSize.y + 1, x + 1 ), tsdf ( y_ + volSize.y + 1, x )
    };

    const int vertBaseIdx = vertIdxBuffer ( z * ( volSize.y - 1 ) + y, x );
    int offsets[12];
    int offset = 0;
    /* Find the vertices where the surface intersects the cube */
    if ( edgeTable[cubeClass] & 1 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[0], ps[1], vals[0], vals[1] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[0], ns[1], vals[0], vals[1] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[0] = offset++;
    }
    if ( edgeTable[cubeClass] & 2 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[1], ps[2], vals[1], vals[2] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[1], ns[2], vals[1], vals[2] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[1] = offset++;
    }
    if ( edgeTable[cubeClass] & 4 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[2], ps[3], vals[2], vals[3] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[2], ns[3], vals[2], vals[3] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[2] = offset++;
    }
    if ( edgeTable[cubeClass] & 8 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[3], ps[0], vals[3], vals[0] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[3], ns[0], vals[3], vals[0] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[3] = offset++;
    }
    if ( edgeTable[cubeClass] & 16 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[4], ps[5], vals[4], vals[5] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[4], ns[5], vals[4], vals[5] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[4] = offset++;
    }
    if ( edgeTable[cubeClass] & 32 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[5], ps[6], vals[5], vals[6] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[5], ns[6], vals[5], vals[6] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[5] = offset++;
    }
    if ( edgeTable[cubeClass] & 64 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[6], ps[7], vals[6], vals[7] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[6], ns[7], vals[6], vals[7] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[6] = offset++;
    }
    if ( edgeTable[cubeClass] & 128 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[7], ps[4], vals[7], vals[4] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[7], ns[4], vals[7], vals[4] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[7] = offset++;
    }
    if ( edgeTable[cubeClass] & 256 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[0], ps[4], vals[0], vals[4] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[0], ns[4], vals[0], vals[4] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[8] = offset++;
    }
    if ( edgeTable[cubeClass] & 512 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[1], ps[5], vals[1], vals[5] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[1], ns[5], vals[1], vals[5] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[9] = offset++;
    }
    if ( edgeTable[cubeClass] & 1024 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[2], ps[6], vals[2], vals[6] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[2], ns[6], vals[2], vals[6] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[10] = offset++;
    }
    if ( edgeTable[cubeClass] & 2048 ) {
        vertices.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ps[3], ps[7], vals[3], vals[7] );
        normals.ptr() [vertBaseIdx + offset] =
            vertexInterp ( ns[3], ns[7], vals[3], vals[7] );
        normals.ptr() [vertBaseIdx + offset] /=
            norm ( normals.ptr() [vertBaseIdx + offset] );
        offsets[11] = offset++;
    }

    const int triBaseIdx = triIdxBuffer ( z * ( volSize.y - 1 ) + y, x );
    for ( int i = 0, j = 0; triTable[cubeClass][i] != -1; i += 3, j += 4 ) {
        triangles.ptr() [triBaseIdx + j] = 3;
        triangles.ptr() [triBaseIdx + j + 1] =
            vertBaseIdx + offsets[triTable[cubeClass][i  ]];
        triangles.ptr() [triBaseIdx + j + 2] =
            vertBaseIdx + offsets[triTable[cubeClass][i+1]];
        triangles.ptr() [triBaseIdx + j + 3] =
            vertBaseIdx + offsets[triTable[cubeClass][i+2]];
    }
}

void marchingCubes ( const cv::cuda::GpuMat& tsdf, const cv::cuda::GpuMat& grad,
                     const cv::cuda::GpuMat& mask, const cv::Vec3i& volumeRes,
                     const float voxelSize, cv::cuda::GpuMat& cubeClasses,
                     cv::cuda::GpuMat& vertIdxBuffer,
                     cv::cuda::GpuMat& triIdxBuffer, cv::cuda::GpuMat& vertices,
                     cv::cuda::GpuMat& normals, cv::cuda::GpuMat& triangles ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( tsdf.cols + threads.x - 1 ) / threads.x,
                  ( tsdf.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    kernel_classifyCubes<<<blocks, threads>>> (
        tsdf, mask, volSize, cubeClasses, vertIdxBuffer, triIdxBuffer );
    cudaDeviceSynchronize();

    int numVerts = cv::cuda::sum ( vertIdxBuffer ) [0];
    if ( !numVerts ) {
        vertices = cv::cuda::GpuMat();
        normals = cv::cuda::GpuMat();
        triangles = cv::cuda::GpuMat();
        return;
    }

    createContinuous ( 1, numVerts, CV_32FC3, vertices );
    createContinuous ( 1, numVerts, CV_32FC3, normals );

    thrust::exclusive_scan ( GpuMatBeginItr<int> ( vertIdxBuffer ),
                             GpuMatEndItr<int> ( vertIdxBuffer ),
                             GpuMatBeginItr<int> ( vertIdxBuffer ) );

    int numTriVerts = cv::cuda::sum ( triIdxBuffer ) [0];
    createContinuous ( 1, numTriVerts, CV_32SC1, triangles );

    thrust::exclusive_scan ( GpuMatBeginItr<int> ( triIdxBuffer ),
                             GpuMatEndItr<int> ( triIdxBuffer ),
                             GpuMatBeginItr<int> ( triIdxBuffer ) );

    kernel_createTriangles<<<blocks, threads>>> (
        tsdf, grad, cubeClasses, volSize, voxelSize, vertIdxBuffer,
        triIdxBuffer, vertices, normals, triangles );
    cudaDeviceSynchronize();
}

}
}
}
