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
#pragma once

#include "EMFusion/core/cuda/common.cuh"

namespace emf {
namespace cuda {
namespace TSDF {

inline __device__
float enterVolStep ( const float3& dir, const float3& campos,
                     const float3& boxBounds ) {
    // If direction positive: should pass through negative bound first.
    // Step for each dimension computed as distance from plane to camera div by
    // direction vector.
    float minstepx = ( ( dir.x > 0.f ? -boxBounds.x : boxBounds.x ) - campos.x )
                     / dir.x;
    float minstepy = ( ( dir.y > 0.f ? -boxBounds.y : boxBounds.y ) - campos.y )
                     / dir.y;
    float minstepz = ( ( dir.z > 0.f ? -boxBounds.z : boxBounds.z ) - campos.z )
                     / dir.z;

    // Maximum: last dimension enters volume.
    return fmaxf ( fmaxf ( minstepx, minstepy ), minstepz );
}

inline __device__
float exitVolStep ( const float3& dir, const float3& campos,
                    const float3& boxBounds ) {
    // If direction positive: should pass through positive bound last.
    // Step for each dimension computed as distance from plane to camera div by
    // direction vector.
    float maxstepx = ( ( dir.x > 0.f ? boxBounds.x : -boxBounds.x ) - campos.x )
                     / dir.x;
    float maxstepy = ( ( dir.y > 0.f ? boxBounds.y : -boxBounds.y ) - campos.y )
                     / dir.y;
    float maxstepz = ( ( dir.z > 0.f ? boxBounds.z : -boxBounds.z ) - campos.z )
                     / dir.z;

    // Minimum: fist dimension left the volume.
    return fminf ( fminf ( maxstepx, maxstepy ), maxstepz );
}

template<typename T>
inline __device__
T interpolateTrilinear ( const cv::cuda::PtrStep<T> vol, const float3& idx,
                         const int3& volSize ) {
    const int3 lowIdx = make_int3 ( static_cast<int> ( idx.x ),
                                    static_cast<int> ( idx.y ),
                                    static_cast<int> ( idx.z ) );
    const int3 highIdx = make_int3 ( lowIdx.x + 1, lowIdx.y + 1, lowIdx.z + 1 );

    const float3 interpFac = idx - lowIdx;

    T vs[] = { vol ( lowIdx.z * volSize.y + lowIdx.y, lowIdx.x ),
               vol ( lowIdx.z * volSize.y + lowIdx.y, highIdx.x ),
               vol ( lowIdx.z * volSize.y + highIdx.y, lowIdx.x ),
               vol ( lowIdx.z * volSize.y + highIdx.y, highIdx.x ),
               vol ( highIdx.z * volSize.y + lowIdx.y, lowIdx.x ),
               vol ( highIdx.z * volSize.y + lowIdx.y, highIdx.x ),
               vol ( highIdx.z * volSize.y + highIdx.y, lowIdx.x ),
               vol ( highIdx.z * volSize.y + highIdx.y, highIdx.x )
             };

    for ( int i = 0; i < 4; ++i ) {
        vs[i] = ( 1 - interpFac.x ) * vs[ 2 * i ]
                + interpFac.x * vs[ 2 * i + 1 ];
    }

    for ( int i = 0; i < 2; ++i ) {
        vs[i] = ( 1 - interpFac.y ) * vs[ 2 * i ]
                + interpFac.y * vs[ 2 * i + 1 ];
    }

    return ( 1 - interpFac.z ) * vs[0] + interpFac.z * vs[1];
}

/**
 * Update the TSDF according to the current depth measurement.
 *
 * @param depth input depth measurement
 * @param assocWeights association weights (do not update if depth pixel does
 *                     not belong to current volume)
 * @param tsdfVol TSDF volume to be updated
 * @param tsdfWeights TSDF integration weights for weighted averaging
 * @param rel_rot_OC relative rotation of object and camera
 * @param rel_trans_OC relative translation of object and camera
 * @param intr camera intrinsic matrix
 * @param truncdist truncation distance
 * @param maxWeight maximum weight for weight-capping (avoid overconfident
 *                  model)
 * @param stream optional Stream for parallel processing
 */
void updateTSDF ( const cv::cuda::GpuMat& depth,
                  const cv::cuda::GpuMat& assocWeights,
                  cv::cuda::GpuMat& tsdfVol, cv::cuda::GpuMat& tsdfWeights,
                  const cv::Matx33f& rel_rot_OC, const cv::Vec3f& rel_trans_OC,
                  const cv::Matx33f& intr, const cv::Vec3i& volumeRes,
                  const float voxelSize, const float truncdist,
                  const float maxWeight,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Compute gradients in TSDF volume with forward differences.
 *
 * @param tsdfVol the volume to compute the gradient for
 * @param tsdfGrads output gradient volume
 * @param volumeRes volume resolution
 * @param stream optional Stream for parallel processing
 */
void computeTSDFGrads ( const cv::cuda::GpuMat& tsdfVol,
                        cv::cuda::GpuMat& tsdfGrads, const cv::Vec3i& volumeRes,
                        cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Raycast the TSDF volume for visualization and current model segmentation.
 *
 * @param tsdfVol the volume to raycast through
 * @param tsdfGrads the gradients of the volume (for surface normals)
 * @param tsdfWeights current integration weights
 * @param raylenghts output ray lengths from raycast
 * @param vertices output pointcloud
 * @param normals output surface normals
 * @param mask output mask for valid pixels from current volume
 * @param rel_rot_CO relative rotation between camera and object
 * @param rel_trans_CO relative translation between camera and object
 * @param intr camera intrinsic matrix
 * @param volumeRes volume resolution
 * @param voxelSize edge length of voxels
 * @param truncdist trucation distance
 * @param stream optional stream for parallel processing
 */
void raycastTSDF ( const cv::cuda::GpuMat& tsdfVol,
                   const cv::cuda::GpuMat& tsdfGrads,
                   const cv::cuda::GpuMat& tsdfWeights,
                   cv::cuda::GpuMat& raylengths, cv::cuda::GpuMat& vertices,
                   cv::cuda::GpuMat& normals, cv::cuda::GpuMat& mask,
                   const cv::Matx33f& rel_rot_CO, const cv::Vec3f& rel_trans_CO,
                   const cv::Matx33f& intr, const cv::Vec3i& volumeRes,
                   const float voxelSize, const float truncdist,
                   cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Compute pixel-wise gradients of SDF wrt camera pose increment.
 *
 * @param tsdfGrads gradients of TSDF volume
 * @param points pointcloud for evaluation of gradients
 * @param rel_rot_CO relative rotation of camera and object
 * @param rel_trans_CO relative translation of camera and object
 * @param volumeRes volume resolution
 * @param voxelSize edge length of voxels
 * @param grads output pose gradients
 * @param stream optional stream for parallel processing
 */
void computePoseGradients ( const cv::cuda::GpuMat& tsdfGrads,
                            const cv::cuda::GpuMat& points,
                            const cv::Matx33f& rel_rot_CO,
                            const cv::Vec3f& rel_trans_CO,
                            const cv::Vec3i& volumeRes,
                            const float voxelSize, cv::cuda::GpuMat& grads,
                            cv::cuda::Stream& stream = cv::cuda::Stream::Null()
                          );

/**
 * Get values from volume at 3D locations with trilinear interpolation.
 *
 * @param vol the volume to get the values from
 * @param points the evaluation points in the volume (in camera coordinates)
 * @param rel_rot_CO relative rotation between camera and object
 * @param rel_trans_CO relative translation beween camera and object
 * @param volumeRes volume resolution
 * @param voxelSize edge length of voxels
 * @param vals output matrix for extracted values
 * @param stream optional stream for parallel processing
 */
void getVolumeVals ( const cv::cuda::GpuMat& vol,
                     const cv::cuda::GpuMat& points,
                     const cv::Matx33f& rel_rot_CO,
                     const cv::Vec3f& rel_trans_CO,
                     const cv::Vec3i& volumeRes, const float voxelSize,
                     cv::cuda::GpuMat& vals,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Compute pixel-wise Hessians from pose gradients.
 *
 * @param grads pixel-wise gradients wrt pose increment
 * @param tsdfVals TSDF values at the 3D points
 * @param As output pixel-wise 6x6 matrix (stored as (H*W)x36)
 * @param bs output pixel-wise gradient scaled with TSDF value ((H*W)x6)
 * @param stream optional stream for parallel processing
 */
void computeAb ( const cv::cuda::GpuMat& grads,
                 const cv::cuda::GpuMat& tsdfVals, cv::cuda::GpuMat& As,
                 cv::cuda::GpuMat& bs,
                 cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Copy values into different volume with offset.
 *
 * @param src source volume
 * @param dst destination volume
 * @param offset pixel offset between volume indices (new = old - offset)
 * @param srcRes source volume resolution
 * @param dstRes destination volume resolution
 */
void copyValues ( const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                  const cv::Vec3i& offset, const cv::Vec3i& srcRes,
                  const cv::Vec3i& dstRes );

/**
 * Multiply matrix with column vector (singleton expansion).
 *
 * @param src1 matrix or column vector
 * @param src2 matrix or column vector
 * @param dst resulting matrix
 * @param stream optional stream for parallel processing
 */
void multSingletonCol ( const cv::cuda::GpuMat& src1,
                        const cv::cuda::GpuMat& src2, cv::cuda::GpuMat& dst,
                        cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

/**
 * Perform the marching cubes algorithm to extract a mesh from the TSDF.
 *
 * @param tsdf the tsdf volume to be processed
 * @param mask a mask for valid voxels for mesh extraction (foreground)
 * @param volumeRes the resolution of the tsdfvolume
 * @param voxelSize the size of a single voxel
 * @param cubeClasses array for storing cube classes, preallocated as CV_8U of
 *                    size ((volumeRes[2]-1) * (volumeRes[1]-1), volumeRes[0]-1)
 * @param vertIdxBuffer array for storing numer of vertices/staring index for
 *                      each cube, preallocated as CV_32SC1 of size as above
 * @param triIdxBuffer array for storing numer of triangles/staring index for
 *                     each cube, preallocated as CV_32SC1 of size as above
 * @param vertices output array for vertices, allocated/resized as needed
 * @param triangles output array for triangles, allocated/resized as needed
 */
void marchingCubes ( const cv::cuda::GpuMat& tsdf, const cv::cuda::GpuMat& grad,
                     const cv::cuda::GpuMat& mask, const cv::Vec3i& volumeRes,
                     const float voxelSize, cv::cuda::GpuMat& cubeClasses,
                     cv::cuda::GpuMat& vertIdxBuffer,
                     cv::cuda::GpuMat& triIdxBuffer, cv::cuda::GpuMat& vertices,
                     cv::cuda::GpuMat& normals, cv::cuda::GpuMat& triangles );

}
}
}
