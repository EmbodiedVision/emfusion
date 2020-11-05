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
#include "EMFusion/core/ObjTSDF.h"
#include "EMFusion/core/cuda/ObjTSDF.cuh"

namespace emf {

int ObjTSDF::nextID = 0;

ObjTSDF::ObjTSDF ( cv::Vec3i _volumeRes, const float _voxelSize,
                   const float _truncdist, cv::Affine3f _pose,
                   TSDFParams _params, cv::Size frameSize ) :
    TSDF ( _volumeRes, _voxelSize, _truncdist, _pose, _params, frameSize ),
    id ( ++nextID ),
    fgBgProbs (
        cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                     _volumeRes[0], CV_32FC2 ) ),
    fgProbs (
        cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                     _volumeRes[0], CV_32FC1 ) ),
    fgProbVals ( cv::cuda::createContinuous ( frameSize, CV_32FC1 ) ),
    raycastWeights (
        cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                     _volumeRes[0], CV_32FC1 ) ) {
    reset ( _pose );
}

bool ObjTSDF::operator== ( const ObjTSDF& other ) const {
    return id == other.id;
}

bool ObjTSDF::operator!= ( const ObjTSDF& other ) const {
    return ! ( *this == other );
}

void ObjTSDF::reset ( const cv::Affine3f& pose ) {
    TSDF::reset ( pose );
    fgBgProbs.setTo ( cv::Scalar ( 0, 0 ) );
}

float ObjTSDF::getExProb() {
    return static_cast<float> ( exCount ) / ( exCount + nonExCount );
}

void ObjTSDF::updateExProb ( const bool exists ) {
    exCount += exists;
    nonExCount += 1 - exists;
}

void ObjTSDF::updateClassProbs ( const std::vector<double>& _classProbs ) {
    if ( classProbs.empty() ) {
        classProbs = _classProbs;
    } else {
        assert ( classProbs.size() == _classProbs.size() );
        transform ( classProbs.begin(), classProbs.end(), _classProbs.begin(),
                    classProbs.begin(), std::plus<double>() );
    }
}

cv::Vec3f ObjTSDF::resize ( const cv::Vec3f& p10, const cv::Vec3f& p90,
                            const float volPad ) {
    cv::Vec3f lowcorner = - ( ( cv::Vec3f ( volumeRes )
                                - cv::Vec3f::all ( 1 ) ) / 2.f )
                          * voxelSize;
    cv::Vec3f highcorner = ( ( cv::Vec3f ( volumeRes )
                               - cv::Vec3f::all ( 1 ) ) / 2.f )
                           * voxelSize;

    bool contained = true;
    for ( int i = 0; i < 3; ++i )
        if ( p10[i] < lowcorner[i] || p90[i] > highcorner[i] ) {
            contained = false;
            break;
        }

    if ( !contained ) {
        cv::Vec3f newCenter = ( p10 + p90 ) / 2.f;
        cv::Vec3i pixOffset = newCenter / voxelSize;
        newCenter = cv::Vec3f ( pixOffset ) * voxelSize;
        // TODO: this might influence the pose trajectory, possibly store an
        //       offset?
        pose = pose.translate ( pose.rotation() * newCenter );

        const cv::Vec3f newDims = p90 - p10;
        const float newVolSize = volPad * ( *std::max_element ( newDims.val,
                                            newDims.val + 3 ) ) / voxelSize;
        // New resolution: ceil volumeSize to get largest needed index, then get
        // next larger even int by adding 1, then int division and
        // multiplication by 2
        const cv::Vec3i newRes =
            cv::Vec3i::all ( ( ( int ) ( std::ceil ( newVolSize ) ) + 1 )
                             / 2 * 2 );

        pixOffset -= ( newRes - volumeRes ) / 2;

        cv::cuda::GpuMat newVol = cv::cuda::createContinuous (
                                      newRes[2] * newRes[1], newRes[0],
                                      CV_32FC1 );
        cv::cuda::GpuMat newWeights = cv::cuda::createContinuous (
                                          newRes[2] * newRes[1], newRes[0],
                                          CV_32FC1 );
        cv::cuda::GpuMat newGrads = cv::cuda::createContinuous (
                                        newRes[2] * newRes[1], newRes[0],
                                        CV_32FC3 );
        cv::cuda::GpuMat newFgBgProbs = cv::cuda::createContinuous (
                                            newRes[2] * newRes[1], newRes[0],
                                            CV_32FC2 );

        newVol.setTo ( 0.f );
        newWeights.setTo ( 0.f );
        newGrads.setTo ( cv::Scalar::all ( 0.f ) );
        newFgBgProbs.setTo ( cv::Scalar::all ( 0.f ) );

        cuda::TSDF::copyValues ( tsdfVol, newVol, pixOffset, volumeRes,
                                 newRes );
        cuda::TSDF::copyValues ( tsdfWeights, newWeights, pixOffset, volumeRes,
                                 newRes );
        cuda::TSDF::copyValues ( tsdfGrads, newGrads, pixOffset, volumeRes,
                                 newRes );
        cuda::TSDF::copyValues ( fgBgProbs, newFgBgProbs, pixOffset, volumeRes,
                                 newRes );

        tsdfVol = newVol;
        tsdfWeights = newWeights;
        tsdfGrads = newGrads;
        fgBgProbs = newFgBgProbs;
        volumeRes = newRes;

        cv::cuda::createContinuous ( ( volumeRes[1] - 1 )
                                     * ( volumeRes[2] - 1 ),
                                     volumeRes[0] - 1, CV_8UC1, cubeClasses );
        cv::cuda::createContinuous ( ( volumeRes[1] - 1 )
                                     * ( volumeRes[2] - 1 ),
                                     volumeRes[0] - 1, CV_32SC1,
                                     vertIdxBuffer );
        cv::cuda::createContinuous ( ( volumeRes[1] - 1 )
                                     * ( volumeRes[2] - 1 ),
                                     volumeRes[0] - 1, CV_32SC1, triIdxBuffer );
        computeFgProbs();

        return newCenter;
    }

    return cv::Vec3f::all ( 0.f );
}

void ObjTSDF::integrateMask ( const cv::cuda::GpuMat& mask,
                              const cv::cuda::GpuMat& occluded_mask,
                              const cv::Affine3f& cam_pose,
                              const cv::Matx33f& intr,
                              cv::cuda::Stream& stream ) {
    const cv::Affine3f rel_pose = cam_pose.inv() * pose;

    cuda::ObjTSDF::updateFgBgProbs ( mask, occluded_mask, tsdfVol, tsdfWeights,
                                     fgBgProbs, rel_pose.rotation(),
                                     rel_pose.translation(), intr, volumeRes,
                                     voxelSize, stream );
    computeFgProbs ( stream );
}

void ObjTSDF::computeAssociation ( const cv::cuda::GpuMat& points,
                                   const cv::Affine3f& cam_pose,
                                   cv::cuda::GpuMat& associationWeights,
                                   cv::cuda::Stream& stream ) {
    TSDF::computeLaplace ( points, cam_pose, stream );

    const cv::Affine3f rel_pose_CO = pose.inv() * cam_pose;

    cuda::TSDF::getVolumeVals ( fgProbs, points, rel_pose_CO.rotation(),
                                rel_pose_CO.translation(), volumeRes, voxelSize,
                                fgProbVals, stream );

    cv::cuda::multiply ( tmpAssocWeights, fgProbVals, tmpAssocWeights, 1, -1,
                         stream );

    cv::cuda::multiply ( tmpAssocWeights, params.alpha, associationWeights, 1,
                         -1, stream );
    cv::cuda::add ( associationWeights, ( 1 - params.alpha ) * params.uniPrior,
                    associationWeights, cv::noArray(), -1, stream );
    associationWeights.setTo ( 0, associationMask, stream );
}

void ObjTSDF::raycast ( const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
                        cv::cuda::GpuMat& raylengths,
                        cv::cuda::GpuMat& vertices, cv::cuda::GpuMat& normals,
                        cv::cuda::GpuMat& mask, cv::cuda::Stream& stream ) {
    const cv::Affine3f rel_pose_CO = pose.inv() * cam_pose;

    raycastWeights.setTo ( 0.f, stream );
    tsdfWeights.copyTo ( raycastWeights, fgVolMask, stream );

    cuda::TSDF::raycastTSDF ( tsdfVol, tsdfGrads, raycastWeights, raylengths,
                              vertices, normals, mask, rel_pose_CO.rotation(),
                              rel_pose_CO.translation(), intr, volumeRes,
                              voxelSize, truncdist, stream );
}

void ObjTSDF::computeFgProbs ( cv::cuda::Stream& stream ) {
    cv::cuda::split ( fgBgProbs, splitFgBgProbs, stream );
    cv::cuda::add ( splitFgBgProbs[0], splitFgBgProbs[1], fgProbs,
                    cv::noArray(), -1, stream );
    cv::cuda::divide ( splitFgBgProbs[0], fgProbs, fgProbs, 1, -1, stream );
    cv::cuda::compare ( fgProbs, fgProbs, fgVolMask, cv::CMP_NE, stream );
    fgProbs.setTo ( 0.f, fgVolMask );
    cv::cuda::compare ( fgProbs, 0.5f, fgVolMask, cv::CMP_GT, stream );
}

void ObjTSDF::syncTrack ( const cv::Affine3f& cam_pose ) {
    streams[0].waitForCompletion();

    cv::Matx44f rel_pose_mat;
    eigen2cv ( rel_pose_CO.inverse().matrix(), rel_pose_mat );

    pose = cam_pose * cv::Affine3f ( rel_pose_mat );
}

void ObjTSDF::getFgProbVals ( cv::Mat& vals ) const {
    fgProbVals.download ( vals );
    vals.convertTo ( vals, CV_8U, 255 );
}

int ObjTSDF::getClassID() const {
    return distance ( classProbs.begin(), max_element ( classProbs.begin(),
                      classProbs.end() ) );
}

cv::viz::Mesh ObjTSDF::getMesh() {
    if ( fgVolMask.empty() )
        computeFgProbs();

    cv::cuda::compare ( tsdfWeights, 0, tsdfVolMask, cv::CMP_GT );
    cv::cuda::bitwise_and ( tsdfVolMask, fgVolMask, tsdfVolMask );
    cubeClasses.setTo ( 0 );
    vertIdxBuffer.setTo ( 0 );
    triIdxBuffer.setTo ( 0 );
    vertices.setTo ( 0 );
    triangles.setTo ( 0 );
    cuda::TSDF::marchingCubes ( tsdfVol, tsdfGrads, tsdfVolMask, volumeRes,
                                voxelSize, cubeClasses, vertIdxBuffer,
                                triIdxBuffer, vertices, normals, triangles );

    cv::viz::Mesh mesh;
    vertices.download ( mesh.cloud );
    normals.download ( mesh.normals );
    triangles.download ( mesh.polygons );

    return mesh;
}

cv::Mat ObjTSDF::getFgProbVol() {
    if ( fgVolMask.empty() )
        computeFgProbs();

    cv::Mat cpu_probs;
    fgProbs.download ( cpu_probs );
    return cpu_probs;
}

cv::cuda::GpuMat ObjTSDF::getFgVolMask () {
    if ( fgVolMask.empty() )
        computeFgProbs ();

    return fgVolMask;
}

}
