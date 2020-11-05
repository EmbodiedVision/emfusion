/*
 * This file is part of EM-Fusion.
 *
 * Copyright (C) 2020 Max-Planck-Gesellschaft.
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
#include "EMFusion/core/cuda/ObjTSDF.cuh"

namespace emf {
namespace cuda {
namespace ObjTSDF {

__global__
void kernel_updateFgBgProbs ( const cv::cuda::PtrStepSz<bool> mask,
                              const cv::cuda::PtrStep<bool> occluded_mask,
                              const cv::cuda::PtrStep<float> tsdfVol,
                              const cv::cuda::PtrStep<float> tsdfWeights,
                              cv::cuda::PtrStep<float2> fgBgProbs,
                              const float33 rot, const float3 trans,
                              const float33 intr, const int3 volSize,
                              const float voxelSize ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= volSize.x || y_ >= volSize.y * volSize.z )
        return;

    const float tsdfVal = tsdfVol ( y_, x );
    const float tsdfWeight = tsdfWeights ( y_, x );

    // Only integrate foreground probability for seen voxels within truncation
    // distance.
    if ( abs ( tsdfVal ) >= 1.f || tsdfWeight == 0.f )
        return;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    const float3 pos_obj =
        make_float3 (
            static_cast<float> ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
            static_cast<float> ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
            static_cast<float> ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize
        );
    const float3 pos_cam = rot * pos_obj + trans;

    if ( pos_cam.z <= 0.f )
        return;

    const float3 proj = intr * pos_cam;

    const int2 pix = make_int2 ( __float2int_rn ( proj.x / proj.z ),
                                 __float2int_rn ( proj.y / proj.z ) );

    if ( pix.x < 0 || pix.x >= mask.cols || pix.y < 0 || pix.y >= mask.rows )
        return;

    if ( !occluded_mask ( pix.y, pix.x ) ) {
        const float2 prev_fgBgProbs = fgBgProbs ( y_, x );
        const float fg_prob = prev_fgBgProbs.x + mask ( pix.y, pix.x );
        const float bg_prob = prev_fgBgProbs.y + ( 1 - mask ( pix.y, pix.x ) );
        fgBgProbs ( y_, x ) = make_float2 ( fg_prob, bg_prob );
    }
}


void updateFgBgProbs ( const cv::cuda::GpuMat& mask,
                       const cv::cuda::GpuMat& occluded_mask,
                       const cv::cuda::GpuMat& tsdfVol,
                       const cv::cuda::GpuMat& tsdfWeights,
                       cv::cuda::GpuMat& fgBgProbs, const cv::Matx33f& rel_rot,
                       const cv::Vec3f& rel_trans, const cv::Matx33f& intr,
                       const cv::Vec3i& volumeRes, const float voxelSize,
                       cv::cuda::Stream& stream ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( fgBgProbs.cols + threads.x - 1 ) / threads.x,
                  ( fgBgProbs.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    const float33 rot = * ( float33 * ) rel_rot.val;
    const float3 trans = * ( float3 * ) rel_trans.val;

    const float33 camIntr = * ( float33 * ) intr.val;

    cudaStream_t stream_cu = cv::cuda::StreamAccessor::getStream ( stream );

    kernel_updateFgBgProbs<<<blocks, threads, 0, stream_cu>>> (
        mask, occluded_mask, tsdfVol, tsdfWeights, fgBgProbs, rot, trans,
        camIntr, volSize, voxelSize );
}

}
}
}
