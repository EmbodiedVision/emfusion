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

namespace emf {

/**
 * Class for TSDF parameters.
 */
class TSDFParams {
public:
    /**
     * Default parameters used in paper experiments.
     */
    TSDFParams() :
        tau ( 1e3 ),
        eps1 ( 1e-8 ),
        eps2 ( 1e-8 ),
        nu_init ( 2.0f ),
        huberThresh ( 0.2f ),
        maxTSDFWeight ( 64.f ),
        assocSigma ( 0.02f ),
        alpha ( 0.8f ),
        uniPrior ( 1.0f ) {}

    /** Factor for identity prior in LM algorithm. */
    float tau;
    /** Convergence threshold 1 (gradient of energy small). */
    float eps1;
    /** Convergence threshold 2 (small change). */
    float eps2;
    /** Initial nu (parameter for rescaling damping after LM-iterations. */
    float nu_init;

    /**
     * Delta-parameter for Huber norm used in tracking (relative to truncation
     * distance).
     */
    float huberThresh;
    /** Weight capping for TSDF. */
    float maxTSDFWeight;

    /** Sigma for Laplace data likelihood (association weight computation). */
    float assocSigma;
    /** Mixture parameter for association likelihood with uniform prior. */
    float alpha;
    /** Value for uniform prior. */
    float uniPrior;
};

/**
 * Class for parameters used during processing.
 */
class Params {
public:
    /**
     * Default parameters working with TUM-RGBD benchmark.
     */
    Params() {
        frameSize = cv::Size ( 640, 480 );

        float fx, fy, cx, cy;
        fx = fy = 525.f;
        cx = frameSize.width/2 - .5f;
        cy = frameSize.height/2 - .5f;
        intr = cv::Matx33f ( fx,  0, cx,
                             0, fy, cy,
                             0,  0,  1 );

        bilateral_sigma_depth = 0.04f; // in meters
        bilateral_sigma_spatial = 4.5f; // in pixels
        bilateral_kernel_size = 7;

        globalVolumeDims = cv::Vec3i::all ( 512 );
        float volSize = 5.12f;
        globalVoxelSize = volSize/globalVolumeDims[0];
        globalRelTruncDist = 10.f;
        objVolumeDims = cv::Vec3i::all ( 64 );
        objRelTruncDist = 10.f;

        volumePose = cv::Affine3f().translate ( cv::Vec3f ( 0, 0, volSize/2 ) );

        volPad = 2.f;
        maxTrackingIter = 100;
        maskRCNNFrames = 30;
        existenceThresh = 0.1f;
        volIOUThresh = 0.5f;
        matchIOUThresh = 0.2f;
        distanceThresh = 5.f;
        visibilityThresh = 40 * 40;
        assocThresh = 0.1f;
        boundary = 20;

        STATIC_OBJECTS = { "traffic light", "fire hydrant", "stop sign",
                           "parking meter", "bench", "couch", "potted plant",
                           "bed", "dining table", "toilet", "oven", "sink",
                           "refrigerator"
                         };
        ignore_person = false;
    }

    /** Frame size in pixels. */
    cv::Size frameSize;

    /** Intrinsic camera matrix. */
    cv::Matx33f intr;

    /** Depth sigma in meters for bilateral filter. */
    float bilateral_sigma_depth;
    /** Spatial sigma in pixels for bilateral filter. */
    float bilateral_sigma_spatial;
    /** Bilateral filter kernel size. */
    int bilateral_kernel_size;

    /** Coarse background model voxel resolution. */
    cv::Vec3i globalVolumeDims;
    /** Coarse background model voxel size in meters. */
    float globalVoxelSize;
    /** Relative truncation distance for background (factor of voxel size). */
    float globalRelTruncDist;
    /** Initial voxel resolution for objects models. */
    cv::Vec3i objVolumeDims;
    /** Relative truncation distance for objects (factor of voxel size). */
    float objRelTruncDist;

    /** Initial volume pose (center of background model relative to camera). */
    cv::Affine3f volumePose;

    /** Padding factor for object models. */
    float volPad;

    /** Maximum number of iterations for tracking algorithm. */
    int maxTrackingIter;

    /** Frame numbers for running Mask R-CNN. */
    int maskRCNNFrames;

    /** Threshold for deleting objects based on existence probability. */
    float existenceThresh;

    /** Threshold for initialization (volumetric IOU). */
    float volIOUThresh;

    /** Threshold for matching Mask R-CNN detections to models. */
    float matchIOUThresh;

    /** Distance threshold (avoid instantiation of far-away objects). */
    float distanceThresh;

    /** Minimum number of pixels to classify object as visible. */
    int visibilityThresh;

    /**
     * Average association likelihood on object-pixels needed (if too low,
     * the tracking likely failed).
     */
    float assocThresh;

    /**
     * Do not take into account these pixels from the boundary for visibility.
     */
    int boundary;

    /** TSDF parameters for tracking and mapping. */
    TSDFParams tsdfParams;

    /** Filter Mask R-CNN detections to contain only these classes. */
    std::vector<std::string> FILTER_CLASSES;
    /** Do not consider these classes as dynamic objects. */
    std::vector<std::string> STATIC_OBJECTS;

    /**
     * Whether to ignore "person" objects during rendering (e.g. for static
     * background tracking).
     */
    bool ignore_person;
};

}
