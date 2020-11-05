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

#include "EMFusion/core/TSDF.h"

namespace emf {

/**
 * Class for storing object TSDFs. Extends TSDF for foreground probabilities,
 * object IDs, and the possibility to resize the volume.
 */
class ObjTSDF : public TSDF {
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
    ObjTSDF ( cv::Vec3i _volumeRes, const float _voxelSize,
              const float _truncdist, cv::Affine3f _pose, TSDFParams _params,
              cv::Size frameSize );

    /**
     * Define objects to be equal if their IDs match.
     *
     * @param other the object to compare to.
     *
     * @return true, if objects are the same, false otherwise
     */
    bool operator== ( const ObjTSDF& other ) const;

    /**
     * The inverse of ==.
     *
     * @param other the object to compare to.
     *
     * @return the inverse of operator==.
     */
    bool operator!= ( const ObjTSDF& other ) const;

    /**
     * Get object id.
     *
     * @return the ID of this object.
     */
    const int getID() const {
        return id;
    }

    /**
     * Reset the volume to zero and set the pose to the given one.
     *
     * @param _pose the pose to set.
     */
    virtual void reset ( const cv::Affine3f& _pose ) override;

    /**
     * Get the existence probability of the current object.
     *
     * @return existence probability compute from counts how often object is
     *         matched with masks.
     */
    float getExProb();

    /**
     * Update existence and non-existence counts depending on whether the object
     * was matched to an incoming mask.
     *
     * @param exists indicator if the object was matched to a mask in the
     *               current frame.
     */
    void updateExProb ( const bool exists );

    /**
     * Update class probabilities for the current object according to Mask R-CNN
     * distribution of matched mask.
     *
     * @param _classProbs class probability vector from Mask R-CNN
     */
    void updateClassProbs ( const std::vector<double>& _classProbs );

    /**
     * Resize object to fit percentiles of incoming pointcloud with padding.
     *
     * @param p10 10th percentile of incoming pointcloud
     * @param p90 90th percentile of incoming pointcloud
     * @param volPad volume padding factor
     *
     * @return offset vector in object coordinates
     */
    cv::Vec3f resize ( const cv::Vec3f& p10, const cv::Vec3f& p90,
                       const float volPad );

    /**
     * Integrade a new Mask R-CNN segmentation into the volume.
     *
     * @param mask Mask R-CNN segmentation matched with object
     * @param occluded_mask Occlusion mask for other objects or background
     * @param cam_pose the camera pose in world coordinates
     * @param intr the camera intrinsic matrix
     * @param stream optional Stream object for parallel processing
     */
    void integrateMask ( const cv::cuda::GpuMat& mask,
                         const cv::cuda::GpuMat& occluded_mask,
                         const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
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
     * Synchronize all tracking streams and update object pose. (Part of
     * tracking algorithm.)
     *
     * @param cam_pose camera pose for computing object pose from relative pose.
     */
    void syncTrack ( const cv::Affine3f& cam_pose );

    /**
     * Get foreground probability image for points mapped into volume during
     * tracking (for visualization).
     *
     * @param vals image storing foreground probabilities for points from
     *             current depthmap.
     */
    void getFgProbVals ( cv::Mat& vals ) const;

    /**
     * Get class ID from current class distribution.
     *
     * @return the ID of the class with maximum probability
     */
    int getClassID () const;

    /**
     * Extract mesh from TSDF volume using marching cubes.
     *
     * @return the resulting mesh.
     */
    virtual cv::viz::Mesh getMesh () override;

    cv::Mat getFgProbVol();

    cv::cuda::GpuMat getFgVolMask ();

private:

    /**
     * Compute foreground probabilities for foreground/background counts.
     *
     * @param stream optional Stream for parallel processing
     */
    void computeFgProbs ( cv::cuda::Stream& stream = cv::cuda::Stream::Null() );

    int id;
    static int nextID;

    int nonExCount = 0;
    int exCount = 0;
    std::vector<double> classProbs;
    cv::cuda::GpuMat fgBgProbs;

    // Caching variables.
    cv::cuda::GpuMat fgProbs, splitFgBgProbs[2], fgProbVals, fgVolMask;
    cv::cuda::GpuMat raycastWeights;
};

}
