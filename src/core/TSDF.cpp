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
#include "EMFusion/core/TSDF.h"
#include "EMFusion/core/cuda/TSDF.cuh"

namespace emf {

TSDF::TSDF ( cv::Vec3i _volumeRes, const float _voxelSize,
             const float _truncdist, cv::Affine3f _pose, TSDFParams _params,
             cv::Size frameSize ) :
    params ( _params ),
    volumeRes ( _volumeRes ),
    voxelSize ( _voxelSize ),
    truncdist ( _truncdist ),
    tsdfVol ( cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                           _volumeRes[0], CV_32FC1 ) ),
    tsdfWeights (
        cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                     _volumeRes[0], CV_32FC1 ) ),
    tsdfGrads (
        cv::cuda::createContinuous ( _volumeRes[1] * _volumeRes[2],
                                     _volumeRes[0], CV_32FC3 ) ) {
    reset ( _pose );

    grads = cv::cuda::createContinuous ( frameSize.height * frameSize.width, 6,
                                         CV_32FC1 );
    bs = cv::cuda::createContinuous ( frameSize.height * frameSize.width, 6,
                                      CV_32FC1 );
    tsdfVals = cv::cuda::createContinuous ( frameSize, CV_32FC1 );
    trackWeights = cv::cuda::createContinuous ( frameSize, CV_32FC1 );
    intWeights = cv::cuda::createContinuous ( frameSize, CV_32FC1 );
    tmpAssocWeights = cv::cuda::createContinuous ( frameSize, CV_32FC1 );

    cubeClasses =
        cv::cuda::createContinuous (
            ( _volumeRes[1] - 1 ) * ( _volumeRes[2] - 1 ), _volumeRes[0] - 1,
            CV_8UC1 );
    vertIdxBuffer =
        cv::cuda::createContinuous (
            ( _volumeRes[1] - 1 ) * ( _volumeRes[2] - 1 ), _volumeRes[0] - 1,
            CV_32SC1 );
    triIdxBuffer =
        cv::cuda::createContinuous (
            ( _volumeRes[1] - 1 ) * ( _volumeRes[2] - 1 ), _volumeRes[0] - 1,
            CV_32SC1 );

    As = cv::cuda::createContinuous ( frameSize.height * frameSize.width, 36,
                                      CV_32FC1 );

    A_gpu = cv::cuda::createContinuous ( 6, 6, CV_32FC1 );
    b_gpu = cv::cuda::createContinuous ( 6, 1, CV_32FC1 );
}

void TSDF::reset ( const cv::Affine3f& _pose ) {
    tsdfVol.setTo ( 0 );
    tsdfWeights.setTo ( 0 );
    tsdfGrads.setTo ( cv::Scalar::all ( 0 ) );
    pose = _pose;
}

void TSDF::getCorners ( cv::Vec3f& low, cv::Vec3f& high ) const {
    const cv::Vec3f corner = ( cv::Vec3f ( volumeRes ) - cv::Vec3f::all ( 1 ) )
                             * voxelSize / 2;
    low = -corner;
    high = corner;
}

cv::Vec3f TSDF::getVolumeSize() const {
    return cv::Vec3f ( volumeRes ) * voxelSize;
}

cv::Vec3i TSDF::getVolumeRes() const {
    return volumeRes;
}

float TSDF::getVoxelSize() const {
    return voxelSize;
}

float TSDF::getTruncDist() const {
    return truncdist;
}

cv::Affine3f TSDF::getPose() const {
    return pose;
}

void TSDF::integrate ( const cv::cuda::GpuMat& depth,
                       const cv::cuda::GpuMat& weights,
                       const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
                       cv::cuda::Stream& stream ) {
    const cv::Affine3f rel_pose_OC = cam_pose.inv() * pose;

    cuda::TSDF::updateTSDF ( depth, weights, tsdfVol, tsdfWeights,
                             rel_pose_OC.rotation(), rel_pose_OC.translation(),
                             intr, volumeRes, voxelSize, truncdist,
                             params.maxTSDFWeight, stream );
}

void TSDF::updateGradients ( cv::cuda::Stream& stream ) {
    tsdfGrads.setTo ( cv::Scalar::all ( 0.f ), stream );
    cuda::TSDF::computeTSDFGrads ( tsdfVol, tsdfGrads, volumeRes, stream );
}

void TSDF::computeAssociation ( const cv::cuda::GpuMat& points,
                                const cv::Affine3f& cam_pose,
                                cv::cuda::GpuMat& associationWeights,
                                cv::cuda::Stream& stream ) {
    computeLaplace ( points, cam_pose, stream );

    cv::cuda::multiply ( tmpAssocWeights, params.alpha, associationWeights, 1,
                         -1, stream );
    cv::cuda::add ( associationWeights, ( 1 - params.alpha ) * params.uniPrior,
                    associationWeights, cv::noArray(), -1, stream );
    associationWeights.setTo ( 0, associationMask, stream );
}

void TSDF::computeLaplace ( const cv::cuda::GpuMat& points,
                            const cv::Affine3f& cam_pose,
                            cv::cuda::Stream& stream ) {
    const cv::Affine3f rel_pose_CO = pose.inv() * cam_pose;

    tmpAssocWeights.setTo ( 0, stream );
    cuda::TSDF::getVolumeVals ( tsdfVol, points, rel_pose_CO.rotation(),
                                rel_pose_CO.translation(), volumeRes, voxelSize,
                                tmpAssocWeights, stream );

    cv::cuda::compare ( tmpAssocWeights, 0, associationMask, cv::CMP_EQ,
                        stream );
    cv::cuda::abs ( tmpAssocWeights, tmpAssocWeights, stream );
    cv::cuda::multiply ( tmpAssocWeights, - truncdist / params.assocSigma,
                         tmpAssocWeights, 1, -1, stream );
    cv::cuda::exp ( tmpAssocWeights, tmpAssocWeights, stream );
    cv::cuda::multiply ( tmpAssocWeights, 1.f / ( 2.f * params.assocSigma ),
                         tmpAssocWeights, 1, -1, stream );
}

void TSDF::raycast ( const cv::Affine3f& cam_pose, const cv::Matx33f& intr,
                     cv::cuda::GpuMat& raylengths, cv::cuda::GpuMat& vertices,
                     cv::cuda::GpuMat& normals, cv::cuda::GpuMat& mask,
                     cv::cuda::Stream& stream ) {
    const cv::Affine3f rel_pose_CO = pose.inv() * cam_pose;

    cuda::TSDF::raycastTSDF ( tsdfVol, tsdfGrads, tsdfWeights, raylengths,
                              vertices, normals, mask, rel_pose_CO.rotation(),
                              rel_pose_CO.translation(), intr, volumeRes,
                              voxelSize, truncdist, stream );
}

void TSDF::prepareTracking ( const cv::Affine3f& cam_pose,
                             cv::cuda::Stream& stream ) {
    streams[0] = stream;

    cv::Affine3f rel_pose = pose.inv() * cam_pose;
    Sophus::Matrix4f rel_pose_mat;
    cv2eigen ( rel_pose.matrix, rel_pose_mat );
    auto QR = rel_pose_mat.block<3,3> ( 0,0 ).householderQr();
    rel_pose_mat.block<3,3> ( 0,0 ) = QR.householderQ();
    for ( int i = 0; i < 3; ++i )
        if ( QR.matrixQR().diagonal() [i] < 0 ) {
            rel_pose_mat.block<3,1> ( 0,i ) *= -1;
        }

    rel_pose_CO = Sophus::SE3f ( rel_pose_mat );
    eigen2cv ( rel_pose_CO.rotationMatrix(), rel_rot_CO );
    eigen2cv ( rel_pose_CO.translation(), rel_trans_CO );

    nu = params.nu_init;
    trackingConverged = false;
    firstIteration = true;
    evaluateGradient = true;
}

void TSDF::computeGradients ( const cv::cuda::GpuMat& points ) {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }

    cuda::TSDF::computePoseGradients ( tsdfGrads, points, rel_rot_CO,
                                       rel_trans_CO, volumeRes, voxelSize,
                                       grads, streams[0] );
}

void TSDF::computeTSDFVals ( const cv::cuda::GpuMat& points ) {
    if ( trackingConverged ) {
        return;
    }

    cuda::TSDF::getVolumeVals ( tsdfVol, points, rel_rot_CO, rel_trans_CO,
                                volumeRes, voxelSize, tsdfVals, streams[1] );
    events[1].record ( streams[1] );
}

void TSDF::computeTSDFWeights ( const cv::cuda::GpuMat& points ) {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }

    cuda::TSDF::getVolumeVals ( tsdfWeights, points, rel_rot_CO, rel_trans_CO,
                                volumeRes, voxelSize, intWeights, streams[2] );
}

void TSDF::computeHuberWeights () {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }

    streams[3].waitEvent ( events[1] );
    cv::cuda::abs ( tsdfVals, trackWeights, streams[3] );
    cv::cuda::divide ( params.huberThresh, trackWeights, trackWeights, 1, -1,
                       streams[3] );
    cv::cuda::min ( trackWeights, 1.0f, trackWeights, streams[3] );
}

void TSDF::normalizeTSDFWeights () {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }
    cv::cuda::min ( intWeights, params.maxTSDFWeight, intWeights, streams[2] );
    cv::cuda::normalize ( intWeights, intWeights, 1.0, 0.0, cv::NORM_INF, -1,
                          cv::noArray(), streams[2] );
    events[2].record ( streams[2] );
}

void TSDF::combineWeights ( const cv::cuda::GpuMat& associationWeights ) {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }
    streams[3].waitEvent ( events[2] );
    cv::cuda::multiply ( trackWeights, intWeights, intWeights, 1, -1,
                         streams[3] );
    cv::cuda::multiply ( intWeights, associationWeights, intWeights, 1, -1,
                         streams[3] );

    events[3].record ( streams[3] );
}

void TSDF::computeHessians() {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }

    streams[0].waitEvent ( events[4] );
    cuda::TSDF::computeAb ( grads, tsdfVals, As, bs, streams[0] );
    events[0].record ( streams[0] );
}

void TSDF::reduceHessians() {
    if ( trackingConverged || !evaluateGradient ) {
        return;
    }
    reduceAb ();

    A_gpu.download ( A, streams[0] );
    b_gpu.download ( b, streams[1] );
    streams[0].waitForCompletion();
    streams[1].waitForCompletion();

    double maxB;
    cv::minMaxLoc ( abs ( b ), NULL, &maxB );
    trackingConverged = ( maxB < params.eps1 );
}

void TSDF::computePoseUpdate ( const cv::cuda::GpuMat& points ) {
    if ( trackingConverged ) {
        return;
    }

    if ( firstIteration ) {
        double maxA;
        cv::minMaxLoc ( A.diag(), NULL, &maxA );
        mu = params.tau * maxA;
        firstIteration = false;
    }

    solve ( A + mu * cv::Mat::eye ( A.size(), A.type() ), b, x );
    Sophus::Vector6f rel_pose_vec = rel_pose_CO.log();
    if ( norm ( x ) < params.eps2 * ( rel_pose_vec.norm() + params.eps2 ) ) {
        trackingConverged = true;
        return;
    }

    float err = computeError();

    Sophus::Vector6f x_soph;
    cv2eigen ( x, x_soph );
    Sophus::SE3f pose_incr_soph = Sophus::SE3f::exp ( -x_soph );
    Sophus::SE3f pose_old = rel_pose_CO;
    rel_pose_CO = pose_incr_soph * rel_pose_CO;
    eigen2cv ( rel_pose_CO.rotationMatrix(), rel_rot_CO );
    eigen2cv ( rel_pose_CO.translation(), rel_trans_CO );

    computeTSDFVals ( points );
    streams[1].waitForCompletion();

    float err_new = computeError();

    cv::Mat gain = ( 0.5f * -x.t() * ( mu * -x - b ) );
    rho = ( err - err_new ) / gain.at<float> ( 0, 0 );

    if ( rho > 0 ) {
        float rho_fac = ( 1.f - ( 2.f * rho - 1.f ) * ( 2.f * rho - 1.f )
                          * ( 2.f * rho - 1.f ) );
        mu *= std::max ( 1.f/3.f, rho_fac );
        nu = params.nu_init;
        evaluateGradient = true;
    } else {
        rel_pose_CO = pose_old;
        eigen2cv ( rel_pose_CO.rotationMatrix(), rel_rot_CO );
        eigen2cv ( rel_pose_CO.translation(), rel_trans_CO );

        mu *= nu;
        nu *= params.nu_init;
        evaluateGradient = false;
    }
}

void TSDF::syncTrack ( cv::Affine3f& cam_pose ) {
    streams[0].waitForCompletion();

    cv::Matx44f rel_pose_mat;
    eigen2cv ( rel_pose_CO.matrix(), rel_pose_mat );
    cam_pose = pose * cv::Affine3f ( rel_pose_mat );
}

void TSDF::getHuberWeights ( cv::Mat& weights ) const {
    trackWeights.download ( weights );
    weights.convertTo ( weights, CV_8U, 255 );
}

void TSDF::getTrackingWeights ( cv::Mat& weights ) const {
    intWeights.download ( weights );
    weights.convertTo ( weights, CV_8U, 255 );
}

cv::viz::Mesh TSDF::getMesh() {
    cv::cuda::compare ( tsdfWeights, 0, tsdfVolMask, cv::CMP_GT );
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

void TSDF::reduceAb () {
    streams[0].waitEvent ( events[3] );
    streams[1].waitEvent ( events[3] );
    streams[1].waitEvent ( events[0] );

    cuda::TSDF::multSingletonCol ( intWeights.reshape ( 1, As.rows ), As, As,
                                   streams[0] );
    cuda::TSDF::multSingletonCol ( intWeights.reshape ( 1, bs.rows ), bs, bs,
                                   streams[1] );

    cv::cuda::reduce ( As, A_gpu.reshape ( 1, 1 ), 0, cv::REDUCE_SUM, -1,
                       streams[0] );
    cv::cuda::reduce ( bs, b_gpu.reshape ( 1, 1 ), 0, cv::REDUCE_SUM, -1,
                       streams[1] );
}

float TSDF::computeError () {
    cv::cuda::sqr ( tsdfVals, errors );
    cv::cuda::multiply ( errors, intWeights, errors );
    return cv::cuda::sum ( errors ) [0];
}

cv::Mat TSDF::getTSDF () const {
    cv::Mat cpu_tsdf;
    tsdfVol.download ( cpu_tsdf );
    return cpu_tsdf;
}

cv::Mat TSDF::getWeightsVol () const {
    cv::Mat cpu_weights;
    tsdfWeights.download ( cpu_weights );
    return cpu_weights;
}

}
