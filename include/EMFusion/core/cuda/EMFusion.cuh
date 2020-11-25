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
namespace EMFusion {

/**
 * Computes points from depth map in camera coordinate system given camera
 * parameters.
 *
 * @param depth input depthmap
 * @param points output pointcloud
 * @param params intrinsic camera matrix
 */
void computePoints ( const cv::cuda::GpuMat& depth, cv::cuda::GpuMat& points,
                     const cv::Matx33f& params );

/**
 * Filters points according to mask. The result is a 1xN matrix where N is the
 * number of positive mask entries.
 *
 * @param points input pointcloud
 * @param mask input mask
 * @param filtered 1-dimensional vector of filtered points
 */
void filterPoints ( const cv::cuda::GpuMat& points,
                    const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& filtered );

/**
 * Computes the 10th and 90th percentile for each coordinate.
 *
 * @param points input pointcloud
 * @param p10 output 10th percentile
 * @param p90 outout 90th percentile
 */
void computePercentiles ( const cv::cuda::GpuMat& points, cv::Vec3f& p10,
                          cv::Vec3f& p90 );

/**
 * Render pointcloud and normals with phong shading.
 *
 * @param points pointcloud to be rendered (one point for each pixel)
 * @param normals surface normals at the points
 * @param segmentation segmentation for object rendering
 * @param colorMap colormap for object identification
 * @param image output image of rendering
 * @param lightPose simulated light pose for rendering
 */
void renderGPU ( const cv::cuda::GpuMat& points,
                 const cv::cuda::GpuMat& normals,
                 const cv::cuda::GpuMat& segmentation, const cv::Mat& colorMap,
                 cv::cuda::GpuMat& image, const cv::Affine3f& lightPose );

}
}
}
