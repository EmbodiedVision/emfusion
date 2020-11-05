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

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

#include "EMFusion/core/data.h"

namespace emf {

/**
 * Class for storing a truncated signed distance volume for processing in the
 * style of KinectFusion.
 */
class TSDF {
public:
    /**
     * Constructor with given truncation distance.
     *
     * @param _volumeRes volume resolution
     * @param _voxelSize size of individual voxels
     * @param _truncdist truncation distance (in meters)
     * @param _pose volume pose (pose of volume center)
     * @param frameSize size of input frames (in pixels)
     */
    TSDF ( cv::Vec3i _volumeRes, const float _voxelSize, const float _truncdist,
           cv::Affine3f _pose, TSDFParams _params, cv::Size frameSize );

    /**
     * Reset the volume to zero and set the pose to the given one.
     *
     * @param _pose the pose to set.
     */
    virtual void reset ( const cv::Affine3f& _pose );

    /**
     * Get corners in object coordinates relative to volume center.
     *
     * @param low low corner
     * @param high high corner
     */
    void getCorners ( cv::Vec3f& low, cv::Vec3f& high ) const;

    /**
     * Get metric volume size.
     *
     * @return the metric size of the volume in all three dimensions.
     */
    cv::Vec3f getVolumeSize() const;

    /**
     * Get the volume resolution.
     *
     * @return a vector containing the current volume resolution.
     */
    cv::Vec3i getVolumeRes() const;

    /**
     * Get the size of a single voxel.
     *
     * @return the size of a single voxel.
     */
    float getVoxelSize() const;

    /**
     * Get the truncation distance of the current volume.
     *
     * @return the trucation distance.
     */
    float getTruncDist() const;

    /**
     * Get the pose of the current volume.
     *
     * @return the pose.
     */
    cv::Affine3f getPose() const;

    /**
     * Integrate a new depth frame into the volume.
     *
     * @param depth the depth to be integrated
     * @param weights the association weights for integration
     * @param cam_pose the camera pose in world coordinates
     * @param intr the camera intrinsic matrix
     * @param stream optional Stream object for parallel processing
     */
    void integrate ( const cv::cuda::GpuMat& depth,
                     const cv::cuda::GpuMat& weights,
                     const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
                     cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

    /**
     * Pre-compute TSDF gradients using forward differences.
     *
     * @param stream optional Stream object for parallel processing
     */
    void updateGradients ( cv::cuda::Stream& stream =
                               cv::cuda::Stream::Null() );

    /**
     * Raycast the current volume.
     *
     * @param cam_pose the camera pose used for rendering.
     * @param intr the camera intrinsic matrix.
     * @param raylengths the raylengths of the resulting raycast.
     * @param vertices the resulting vertices in camera coordinates
     * @param normals the resulting normals in camera coordinates
     * @param mask mask of valid pixels from raycast
     */
    void raycast ( const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
                   cv::cuda::GpuMat& raylengths, cv::cuda::GpuMat& vertices,
                   cv::cuda::GpuMat& normals, cv::cuda::GpuMat& mask,
                   cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

    /**
     * Compute the association (data likelihood) for the current volume.
     *
     * @param points input pointcloud from current depth frame
     * @param cam_pose current camera pose
     * @param associationWeights output weights
     * @param stream optional Stream object for parallel processing
     */
    void computeAssociation ( const cv::cuda::GpuMat& points,
                              const cv::Affine3f& cam_pose,
                              cv::cuda::GpuMat& associationWeights,
                              cv::cuda::Stream& stream =
                                  cv::cuda::Stream::Null() );

    /**
     * Prepare internal variables for processing of tracking algorithm.
     *
     * @param cam_pose current camera pose
     * @param stream optional stream for parallel processing
     */
    void prepareTracking ( const cv::Affine3f& cam_pose,
                           cv::cuda::Stream& stream = cv::cuda::Stream::Null()
                         );

    /**
     * Compute pose gradients for all pixels wrt. current depthmap/pointcloud.
     * (Part of tracking algorithm.)
     *
     * @param points current pointcloud generated from depthmap
     */
    void computeGradients ( const cv::cuda::GpuMat& points );

    /**
     * Compute TSDF values at 3D locations of input pointcloud. (Part of
     * tracking algorithm.)
     *
     * @param points pointcloud generated from depthmap
     */
    void computeTSDFVals ( const cv::cuda::GpuMat& points );

    /**
     * Compute TSDF weights at 3D locations of input pointcloud. (Part of
     * tracking algorithm.)
     *
     * @param points pointcloud generated from depthmap
     */
    void computeTSDFWeights ( const cv::cuda::GpuMat& points );

    /**
     * Compute weights for realizing Huber norm in tracking. (Part of tracking
     * algorithm.)
     */
    void computeHuberWeights ();

    /**
     * Normalize TSDF weights to range [0,1] for correct weighting of tracking.
     * (Part of tracking algorithm.)
     */
    void normalizeTSDFWeights ();

    /**
     * Combine all precomputed weights with association weights to get final
     * tracking weights. (Part of tracking algorithm.)
     *
     * @param associationWeights association weights for current volume.
     */
    void combineWeights ( const cv::cuda::GpuMat& associationWeights );

    /**
     * Compute pixel-wise Hessians from gradients. (Part of tracking algorithm.)
     */
    void computeHessians ();

    /**
     * Accumulate pixel-wise data with weights to single linear system. (Part of
     * tracking algorithm.)
     */
    void reduceHessians ();

    /**
     * Compute pose update from linear system and evaluate next steps using
     * input pointcloud as dataterm. (Part of tracking algorithm.)
     *
     * @param points pointcloud generated from depthmap.
     */
    void computePoseUpdate ( const cv::cuda::GpuMat& points );

    /**
     * Synchronize all tracking streams and update camera pose. (Part of
     * tracking algorithm.)
     *
     * @param cam_pose camera pose that will be updated.
     */
    void syncTrack ( cv::Affine3f& cam_pose );

    /**
     * Get Huber weights from last tracking step for output.
     *
     * @param weights output matrix
     */
    void getHuberWeights ( cv::Mat& weights ) const;

    /**
     * Get final tracking weights from last step for output.
     *
     * @param weights output matrix
     */
    void getTrackingWeights ( cv::Mat& weights ) const;

    /**
     * Extract mesh from TSDF volume using marching cubes.
     *
     * @return the resulting mesh.
     */
    virtual cv::viz::Mesh getMesh ();

    /**
     * Download the TSDF volume from GPU and return the resulting Mat.
     *
     * @return a Mat containing the SDF data.
     */
    virtual cv::Mat getTSDF () const;

    /**
     * Download the integration weight volume and return the resulting Mat.
     *
     * @return a Mat containing the integration weights.
     */
    virtual cv::Mat getWeightsVol () const;


protected:
    /**
     * Reduce pixel-wise variables to linear system for whole image.
     */
    void reduceAb ();

    /**
     * Compute the current tracking error.
     *
     * @return the error value.
     */
    float computeError ();

    /**
     * Compute Laplace data likelihood.
     *
     * @param points input pointcloud
     * @param cam_pose input camera pose
     * @param stream optional Stream object for parallel processing
     */
    void computeLaplace ( const cv::cuda::GpuMat& points,
                          const cv::Affine3f& cam_pose,
                          cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

    TSDFParams params;

    cv::Vec3i volumeRes;
    float voxelSize;
    float truncdist;
    cv::cuda::GpuMat tsdfVol;
    cv::cuda::GpuMat tsdfWeights;
    cv::cuda::GpuMat tsdfGrads;
    cv::Affine3f pose;

    int frameCount = 0;

    bool trackingConverged = false, evaluateGradient = true,
         firstIteration = true;
    float mu, nu, rho;

    // Caching variables needed during processing
    Sophus::SE3f rel_pose_CO;
    cv::Matx33f rel_rot_CO;
    cv::Vec3f rel_trans_CO;
    cv::cuda::GpuMat grads, tsdfVals, intWeights;
    cv::cuda::GpuMat trackWeights, tsdfVolMask;
    cv::cuda::GpuMat As, bs, A_gpu, b_gpu;
    cv::cuda::GpuMat associationMask;
    cv::cuda::GpuMat errors;
    cv::cuda::GpuMat tmpAssocWeights;
    cv::cuda::GpuMat cubeClasses, vertIdxBuffer, triIdxBuffer, vertices;
    cv::cuda::GpuMat normals, triangles;

    cv::Mat A, A_lam, A_lam_new, b, x;

    cv::cuda::Stream streams[6];
    cv::cuda::Event events[6];
};

}
