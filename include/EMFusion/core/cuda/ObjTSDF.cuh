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
#include "EMFusion/core/cuda/TSDF.cuh"

namespace emf {
namespace cuda {
namespace ObjTSDF {

/**
 * Update foreground/background counts for voxels according to new Mask R-CNN
 * segmentation.
 *
 * @param mask new segmentation
 * @param occluded_mask mask of occluded pixels for current object
 * @param tsdfVol TSDF volume (fg probability only valid within truncation dist)
 * @param tsdfWeigths TSDF weights (only update fg prob if measurements were
 *                    integrated in this voxels
 * @param fgBgProbs input/output foreground/background counts
 * @param rel_rot relative rotation of camera and object
 * @param rel_trans relative translation of camera and object
 * @param intr camera intrinsic matrix
 * @param volumeRes volume resolution
 * @param voxelSize edge length of voxels
 * @param stream optional stream for parallel processing
 */
void updateFgBgProbs ( const cv::cuda::GpuMat& mask,
                       const cv::cuda::GpuMat& occluded_mask,
                       const cv::cuda::GpuMat& tsdfVol,
                       const cv::cuda::GpuMat& tsdfWeights,
                       cv::cuda::GpuMat& fgBgProbs, const cv::Matx33f& rel_rot,
                       const cv::Vec3f& rel_trans, const cv::Matx33f& intr,
                       const cv::Vec3i& volumeRes, const float voxelSize,
                       cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

}
}
}
