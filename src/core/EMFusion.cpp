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
#include "EMFusion/core/EMFusion.h"
#include "EMFusion/core/cuda/EMFusion.cuh"

namespace emf {

EMFusion::EMFusion ( const Params& _params ) :
    params ( _params ),
    background ( _params.globalVolumeDims, _params.globalVoxelSize,
                 _params.globalRelTruncDist * _params.globalVoxelSize,
                 _params.volumePose, _params.tsdfParams, _params.frameSize ),
    maskrcnn ( _params.FILTER_CLASSES, _params.STATIC_OBJECTS ),
    frameCount ( 0 ) {
    colorMap = randomColors();
    raylengths = cv::cuda::createContinuous ( _params.frameSize, CV_32FC1 );
    vertices = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    normals = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    modelSegmentation = cv::cuda::createContinuous ( _params.frameSize, CV_8U );
    bg_raylengths = cv::cuda::createContinuous ( _params.frameSize, CV_32FC1 );
    bg_vertices = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    bg_normals = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    mask = cv::cuda::createContinuous ( _params.frameSize, CV_8UC1 );
    points = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    points_w = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    image = cv::cuda::createContinuous ( _params.frameSize, CV_8UC3 );
    bg_mask = cv::cuda::createContinuous ( _params.frameSize, CV_8UC1 );
    bg_mask.setTo ( 0 );
    noObjMask = cv::cuda::createContinuous ( _params.frameSize, CV_8UC1 );
    noObjMask.setTo ( 1 );
    associationNorm =
        cv::cuda::createContinuous ( _params.frameSize, CV_32FC1 );
    bg_associationWeights =
        cv::cuda::createContinuous ( _params.frameSize, CV_32FC1 );
    bg_associationWeights.setTo ( 1.f );
}

void EMFusion::reset() {
    pose = cv::Affine3f::Identity();
    background.reset ( params.volumePose );
    objects.clear();
    streams.clear();
    obj_raylengths.clear();
    obj_vertices.clear();
    obj_normals.clear();
    obj_modelSegmentation.clear();
    associationWeights.clear();
}

void EMFusion::processFrame ( const RGBD& frame ) {
    cv::Mat rgb ( frame.getRGB() );
    depth_raw.upload ( frame.getDepth() );

    preprocessDepth ( depth_raw, depth );

    cuda::EMFusion::computePoints ( depth, points, params.intr );

    if ( frameCount > 0 ) {
        computeAssociationWeights();
        if ( saveOutput ) {
            storeAssocs ( bg_associationWeights, bg_assocWeight_preTrack,
                          associationWeights, obj_assocWeights_preTrack );
        }

        performTracking();

        computeAssociationWeights();
        if ( saveOutput ) {
            storeAssocs ( bg_associationWeights, bg_assocWeight_postTrack,
                          associationWeights, obj_assocWeights_postTrack );
        }

        // Raycast to get reprojected object masks
        raycast();
    }
    storePoses();

    std::map<int, cv::cuda::GpuMat> matches;
    int numInstances = -1;
    if ( frameCount % params.maskRCNNFrames == 0 )
        numInstances = initOrMatchObjs ( rgb, matches );

    integrateDepth ();

    if ( numInstances > 0 )
        integrateMasks ( matches );

    cleanUpObjs ( numInstances, matches );

    if ( saveOutput ) {
        background.getHuberWeights ( bg_huberWeights[frameCount] );
        background.getTrackingWeights ( bg_trackWeights[frameCount] );

        if ( expFrameMeshes )
            frame_meshes[frameCount] = background.getMesh();
        for ( auto& obj : objects ) {
            obj.getHuberWeights ( obj_huberWeights[obj.getID()][frameCount] );
            obj.getTrackingWeights (
                obj_trackWeights[obj.getID()][frameCount] );
            obj.getFgProbVals ( obj_fgProbs[obj.getID()][frameCount] );
            if ( ! ( params.ignore_person &&
                     emf::MaskRCNN::getClassName ( obj.getClassID() )
                     == "person" ) && expFrameMeshes )
                frame_obj_meshes[obj.getID()][frameCount] = obj.getMesh();
        }
    }

    ++frameCount;
}

void EMFusion::render ( cv::Mat& rendered, cv::viz::Viz3d* window ) {
    if ( frameCount < 1 ) {
        rendered = cv::Mat::zeros ( params.frameSize, CV_8UC3 );
        return;
    } else if ( frameCount == 1 ) {
        raycast();
    }

    if ( params.ignore_person ) {
        // Ignore person label for rendering
        for ( const auto& obj : objects ) {
            if ( MaskRCNN::getClassName ( obj.getClassID() ) == "person" ) {
                cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                                    cv::CMP_EQ );
                modelSegmentation.setTo ( 0, obj_mask );
                bg_vertices.copyTo ( vertices, obj_mask );
                bg_normals.copyTo ( normals, obj_mask );
            }
        }
    }

    image.setTo ( cv::Vec3b::all ( 0 ) );
    cuda::EMFusion::renderGPU ( vertices, normals, modelSegmentation, colorMap,
                                image, cv::Affine3f::Identity() );

    image.download ( rendered );

    if ( saveOutput ) {
        rendered.copyTo ( renderings[frameCount-1] );
    }

    if ( window ) {
        if ( frameCount > 0 ) {
            window->removeAllWidgets();
            cv::viz::WCameraPosition camPos ( params.intr, rendered, 0.1 );
            camPos.applyTransform ( pose );
            window->showWidget ( "CamPos", camPos );
            for ( auto& obj : objects ) {
                if ( MaskRCNN::getClassName ( obj.getClassID() ) == "person"
                        && params.ignore_person ) {
                    continue;
                }
                cv::Vec3f low, high;
                obj.getCorners ( low, high );
                cv::Affine3f obj_pose = obj.getPose();
                cv::viz::WCube obj_cube ( ( cv::Vec3d ( low ) ),
                                          ( cv::Vec3d ( high ) ) );
                obj_cube.applyTransform ( obj_pose );
                obj_cube.setColor ( cv::viz::Color (
                                        colorMap.at<cv::Vec3b> ( 0,
                                                obj.getID() ) ) );
                cv::viz::Mesh m_obj;
                if ( expFrameMeshes )
                    m_obj = frame_obj_meshes[obj.getID()][frameCount-1];
                else
                    m_obj = obj.getMesh();
                if ( m_obj.cloud.rows == 1 ) {
                    cv::viz::WMesh m_show_obj ( m_obj );
                    m_show_obj.setColor ( cv::viz::Color (
                                              colorMap.at<cv::Vec3b> ( 0,
                                                      obj.getID() ) ) );
                    m_show_obj.applyTransform ( obj.getPose() );
                    std::stringstream mesh_str;
                    mesh_str << "Mesh " << obj.getID();
                    window->showWidget ( mesh_str.str(), m_show_obj );
                    window->setRenderingProperty ( mesh_str.str(),
                                                   cv::viz::SHADING,
                                                   cv::viz::SHADING_PHONG );
                }
                cv::viz::WCoordinateSystem obj_coords ( 0.2 );
                cv::Affine3f coord_pose = obj_pose
                                          * cv::Affine3f().translate ( low );
                obj_coords.applyTransform ( coord_pose );

                std::stringstream mesh_str, cube_str, coords_str;
                cube_str << "Cube " << obj.getID();
                coords_str << "Coords " << obj.getID();
                window->showWidget ( cube_str.str(), obj_cube );
                window->showWidget ( coords_str.str(), obj_coords );
            }
            cv::viz::Mesh bg_mesh;
            if ( expFrameMeshes )
                bg_mesh = frame_meshes[frameCount-1];
            else
                bg_mesh = background.getMesh();
            cv::Vec3f low, high;
            background.getCorners ( low, high );
            cv::viz::WCube cube ( ( cv::Vec3d ( low ) ), ( cv::Vec3d ( high ) ) );
            cube.applyTransform ( background.getPose() );

            if ( bg_mesh.cloud.rows == 1 ) {
                cv::viz::WMesh viz_mesh ( bg_mesh );
                viz_mesh.applyTransform ( background.getPose() );
                window->showWidget ( "mesh", viz_mesh );
                window->showWidget ( "cube", cube );
                window->setRenderingProperty ( "mesh", cv::viz::SHADING,
                                               cv::viz::SHADING_PHONG );
            }

            if ( saveOutput ) {
                window->getScreenshot().copyTo ( mesh_vis[frameCount - 1] );
            }
        }
    }
}

void EMFusion::getLastMasks ( cv::Mat& maskim ) {
    mask_vis[ ( ( frameCount - 1 ) / params.maskRCNNFrames )
              * params.maskRCNNFrames].copyTo ( maskim );
}


void EMFusion::setupOutput ( bool exp_frame_meshes, bool exp_vols ) {
    saveOutput = true;
    expFrameMeshes = exp_frame_meshes;
    expVols = exp_vols;
}

void EMFusion::usePreprocMasks ( const std::string& path ) {
    maskPath = path;
}

void EMFusion::writeResults ( const std::string& path ) {
    boost::filesystem::path p ( path );
    boost::filesystem::create_directories ( p );

    writePoses ( p );

    writeRenderings ( p );

    writeMeshVis ( p );

    writeMasks ( p );

    writeAssocs ( p );

    writeHuberWeights ( p );

    writeTrackWeights ( p );

    writeFgProbs ( p );

    for ( auto& obj : objects ) {
        if ( params.ignore_person &&
                emf::MaskRCNN::getClassName ( obj.getClassID() )
                == "person" )
            continue;
        meshes[obj.getID()] = obj.getMesh();
        if ( expVols ) {
            tsdfs[obj.getID()] = obj.getTSDF();
            intWeights[obj.getID()] = obj.getWeightsVol();
            fgProbs[obj.getID()] = obj.getFgProbVol();
            meta[obj.getID()] = std::make_pair ( obj.getVolumeRes(),
                                                 obj.getVoxelSize() );
        }
    }

    writeMeshes ( p );

    if ( expVols )
        writeTSDFs ( p );
}

void EMFusion::preprocessDepth ( const cv::cuda::GpuMat& depth_raw,
                                 cv::cuda::GpuMat& depth ) {
    cv::cuda::bilateralFilter ( depth_raw, depth, params.bilateral_kernel_size,
                                params.bilateral_sigma_depth,
                                params.bilateral_sigma_spatial );

    // Patch NaN values introduced by bilateral filter
    cv::cuda::compare ( depth, depth, depth_mask, cv::CMP_NE );
    depth.setTo ( 0, depth_mask );
    cv::cuda::compare ( depth_raw, 0, depth_mask, cv::CMP_EQ );
    depth.setTo ( 0, depth_mask );
}

void EMFusion::storeAssocs ( const cv::cuda::GpuMat& bg_associationWeights,
                             std::map<int, cv::Mat>& bg_assocWeights_frame,
                             const std::map<int, cv::cuda::GpuMat>& assocs,
                             std::map<int, std::map<int, cv::Mat>>& assocs_frame
                           ) {
    bg_associationWeights.download ( bg_assocWeights_frame[frameCount] );
    bg_assocWeights_frame[frameCount].convertTo (
        bg_assocWeights_frame[frameCount], CV_8U, 255 );
    for ( const auto& assocW : assocs ) {
        assocW.second.download ( assocs_frame[assocW.first][frameCount] );
        assocs_frame[assocW.first][frameCount].convertTo (
            assocs_frame[assocW.first][frameCount], CV_8U, 255 );
    }
}

void EMFusion::storePoses() {
    poses[frameCount] = pose;
    for ( const auto& obj : objects ) {
        obj_poses[obj.getID()][frameCount] = obj.getPose();
    }
}

int EMFusion::initOrMatchObjs ( const cv::Mat& rgb,
                                std::map<int, cv::cuda::GpuMat>& matches ) {
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Mat> segmentation;
    std::vector<std::vector<double>> scores;

    int numInstances = -1;
    numInstances = runMaskRCNN ( rgb, bounding_boxes, segmentation, scores );

//     std::map<int, cv::cuda::GpuMat> matches;
    std::map<int, std::vector<double>> score_matches;
    std::set<int> unmatchedMasks;

    // Prepare for matching with existing models: Get mask for valid depth
    // measurements and transform points to world coordinates for
    // initialization.
    if ( numInstances > 0 ) {
        seg_gpus.resize ( numInstances );
        for ( int i = 0; i < segmentation.size(); ++i )
            seg_gpus[i].upload ( segmentation[i] );

        computeValidPoints ( points, validPoints );

        transformPoints ( points, pose, points_w );

        matchSegmentation ( seg_gpus, scores, matches, score_matches,
                            unmatchedMasks );

        initObjsFromUnmatched ( seg_gpus, scores, unmatchedMasks, matches,
                                score_matches );
        for ( auto& obj : objects ) {
            if ( matches.count ( obj.getID() ) ) {
                obj_pose_offsets[obj.getID()][frameCount] =
                    updateObj ( obj, points_w, matches[obj.getID()],
                                score_matches[obj.getID()] );
                obj.updateExProb ( true );
            } else {
                // Unmatched object -> lower existence probability
                obj.updateExProb ( false );
            }
        }
    }

    return numInstances;
}

int EMFusion::runMaskRCNN ( const cv::Mat& rgb,
                            std::vector<cv::Rect>& bounding_boxes,
                            std::vector<cv::Mat>& segmentation,
                            std::vector<std::vector<double>>& scores ) {
    int numInstances = -1;
    if ( maskPath.empty() ) {
        numInstances = maskrcnn.execute ( rgb, bounding_boxes, segmentation,
                                          scores );
    } else {
        std::stringstream filename;
        filename << maskPath << "/" << "Mask" << std::setw ( 4 )
                 << std::setfill ( '0' ) << frameCount << ".plk";
        numInstances =
            maskrcnn.loadPreprocessed ( filename.str(), bounding_boxes,
                                        segmentation, scores );
    }
    MaskRCNN::visualize ( mask_vis[frameCount], rgb, numInstances,
                          bounding_boxes, segmentation, scores );
//         imshow ( "Masked RGB", mask_vis[frameCount] );
    return numInstances;
}

void EMFusion::computeValidPoints ( const cv::cuda::GpuMat& points,
                                    cv::cuda::GpuMat& validPoints ) {
    // Only points not mapping to [0,0,0] in camera coordinates are valid.
    cv::cuda::compare ( points, cv::Vec3f::zeros(), validPoints,
                        cv::CMP_NE );
    // Get single-channel matrix
    cv::cuda::cvtColor ( validPoints, validPoints, cv::COLOR_RGB2GRAY );
    // Value for true needs to be 1
    cv::cuda::threshold ( validPoints, validPoints, 1, 1,
                          cv::THRESH_BINARY );
}

void EMFusion::transformPoints ( const cv::cuda::GpuMat& points,
                                 const cv::Affine3f& pose,
                                 cv::cuda::GpuMat& points_w ) {
    cv::cuda::transformPoints ( points.reshape ( 3, 1 ),
                                cv::Mat ( pose.rvec().t() ),
                                cv::Mat ( pose.translation().t() ), points_w );
    points_w = points_w.reshape ( 3, points.rows );
}

void EMFusion::matchSegmentation (
    const std::vector<cv::cuda::GpuMat>& seg_gpus,
    const std::vector<std::vector<double>>& scores,
    std::map<int, cv::cuda::GpuMat>& matches,
    std::map<int, std::vector<double>>& score_matches,
    std::set<int>& unmatchedMasks ) {

    for ( int i = 0; i < seg_gpus.size(); ++i ) {
        int matched_id = -1;
        if ( frameCount > 0 ) {
            float new_iou = 0.f;
            matched_id = matchSegmentation ( seg_gpus[i], new_iou );
            if ( matched_id >= 0 && matches.count ( matched_id ) ) {
                cv::cuda::compare ( modelSegmentation, matched_id, obj_mask,
                                    cv::CMP_EQ );
                cv::cuda::bitwise_and ( matches[matched_id], obj_mask,
                                        seg_inter );
                cv::cuda::bitwise_or ( matches[matched_id], obj_mask, seg_uni );

                const float prev_iou =
                    static_cast<float> ( cv::cuda::countNonZero ( seg_inter ) )
                    / cv::cuda::countNonZero ( seg_uni );
                if ( new_iou > prev_iou ) {
                    matches[matched_id] = seg_gpus[i];
                    score_matches[matched_id] = scores[i];
                }
                matched_id = -1;
            }
        }

        if ( matched_id >= 0 ) {
            matches.insert ( std::make_pair ( matched_id, seg_gpus[i] ) );
            score_matches.insert ( std::make_pair ( matched_id, scores[i] ) );
        } else {
            unmatchedMasks.insert ( i );
        }
    }
}

void EMFusion::initObjsFromUnmatched (
    std::vector<cv::cuda::GpuMat>& seg_gpus,
    const std::vector<std::vector<double>>& scores,
    const std::set<int>& unmatchedMasks,
    std::map<int, cv::cuda::GpuMat>& matches,
    std::map<int, std::vector<double>>& score_matches ) {
    for ( int i : unmatchedMasks ) {
        for ( const auto& obj : objects ) {
            // Get reprojected mask for current object
            cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                                cv::CMP_EQ );
            // If mask was matched, join it with the new mask
            if ( matches.count ( obj.getID() ) ) {
                cv::cuda::bitwise_or ( obj_mask, matches[obj.getID()],
                                       obj_mask );
            }

            // Invert mask and do bitwise and to remove existing object mask
            // from new unmatched mask
            cv::cuda::threshold ( obj_mask, obj_mask, 0, 1,
                                  cv::THRESH_BINARY_INV );
            int count_mask_pre = cv::cuda::countNonZero ( seg_gpus[i] );
            cv::cuda::bitwise_and ( seg_gpus[i], obj_mask, seg_gpus[i] );
            // If we removed more than half of the mask in this process, we do
            // not initialize a new volume.
            if ( static_cast<float> ( cv::cuda::countNonZero ( seg_gpus[i] ) )
                    / static_cast<float> ( count_mask_pre ) < .5f ) {
                seg_gpus[i].setTo ( 0 ); // Zero mask will not initialize volume
            }
        }

        cv::cuda::bitwise_and ( validPoints, seg_gpus[i], mask );
        int obj_id = initNewObjVolume ( mask, points_w, pose );
        matches.insert ( std::make_pair ( obj_id, seg_gpus[i] ) );
        score_matches.insert ( std::make_pair ( obj_id, scores[i] ) );
    }
}

int EMFusion::initNewObjVolume ( const cv::cuda::GpuMat& mask,
                                 const cv::cuda::GpuMat& points,
                                 const cv::Affine3f& pose ) {
    if ( cv::cuda::countNonZero ( mask ) < params.visibilityThresh ) {
        return -1;
    }

    cv::cuda::GpuMat filteredPoints, objFilteredPoints;
    cuda::EMFusion::filterPoints ( points, mask, filteredPoints );

    // Check if overlap with already existing objects is too large
    for ( const auto& obj : objects ) {
        cv::Affine3f obj_pose = obj.getPose().inv();
        cv::cuda::transformPoints ( filteredPoints,
                                    cv::Mat ( obj_pose.rvec().t() ),
                                    cv::Mat ( obj_pose.translation().t() ),
                                    objFilteredPoints );

        // Compute volume boundaries in object coordinates for overlap check
        cv::Vec3f p10, p90;
        cuda::EMFusion::computePercentiles ( objFilteredPoints, p10, p90 );

        // Check if intersection over union > .5
        float iou = volumeIOU ( obj, p10, p90 );
        if ( iou > params.volIOUThresh ) {
            return -1;
        }
    }

    // New object instance is spawned alinged with world coordinate system
    cv::Vec3f p10, p90;
    cuda::EMFusion::computePercentiles ( filteredPoints, p10, p90 );

    cv::Vec3f center = ( p10 + p90 ) / 2;

    // Only initialize volume if center not too far from camera.
    if ( norm ( center - pose.translation() ) > params.distanceThresh ) {
        return -1;
    }

    // Volume size determined by largest axis of percentile difference
    // vector times padding.
    cv::Vec3f dims = p90 - p10;
    float volSize = params.volPad *
                    ( *std::max_element ( dims.val, dims.val + 3 ) );

    // New objects are aligned with the world coordinate system, so only
    // the center is relevant for the pose.
    cv::Affine3f obj_pose ( cv::Matx33f::eye(), center );

    ObjTSDF obj ( params.objVolumeDims, volSize/params.objVolumeDims[0],
                  params.objRelTruncDist * volSize/params.objVolumeDims[0],
                  obj_pose, params.tsdfParams, params.frameSize );

    objects.push_back ( obj );
    vis_objs.insert ( obj.getID() );
    createObj ( obj.getID() );

    obj_poses[obj.getID()][frameCount] = obj.getPose();

    std::cout << "Created new Object with ID: " << obj.getID() << std::endl;

    return obj.getID();
}

float EMFusion::volumeIOU ( const ObjTSDF& obj, const cv::Vec3f& p10,
                            const cv::Vec3f& p90 ) {
    cv::Vec3f center = ( p10 + p90 ) / 2;

    // Volume size determined by largest axis of percentile difference
    // vector times padding.
    cv::Vec3f dims = p90 - p10;
    float volSize = params.volPad *
                    ( *std::max_element ( dims.val, dims.val + 3 ) );

    // Corner points of new volume.
    cv::Vec3f low_new = center - cv::Vec3f::all ( volSize / 2 );
    cv::Vec3f high_new = center + cv::Vec3f::all ( volSize / 2 );

    // Get corners and volume of existing object
    cv::Vec3f low, high;
    obj.getCorners ( low, high );

    cv::Vec3f prevVolSize = obj.getVolumeSize();
    auto vol = accumulate ( prevVolSize.val, prevVolSize.val + 3, 1.f,
                            std::multiplies<float>() );

    float vol_new = pow ( volSize, 3 );
    // Compute corners of intersection volume ( low_int = max( low, low_new );
    // high_int = min( high, high_new ) )
    cv::Vec3f low_int, high_int;
    std::transform ( low_new.val, low_new.val + 3, low.val, low_int.val,
    [] ( const float& a, const float& b ) {
        return std::max ( a, b );
    } );
    std::transform ( high_new.val, high_new.val + 3, high.val, high_int.val,
    [] ( const float& a, const float& b ) {
        return std::min ( a, b );
    } );

    // Compute dimensions of intersection
    cv::Vec3f dims_int = high_int - low_int;

    // If any dimension is negative => no overlap
    bool noOverlap = std::any_of ( dims_int.val, dims_int.val + 3,
    [] ( const float& f ) {
        return f < 0;
    } );
    if ( noOverlap ) {
        return 0;
    }

    // Compute volume of intersection
    auto vol_int = accumulate ( dims_int.val, dims_int.val + 3, 1.f,
                                std::multiplies<float>() );

    return vol_int / ( vol_new + vol - vol_int );
}

cv::Mat EMFusion::randomColors () {
    cv::Mat hsv ( 1, 256, CV_32FC3 );

    for ( int i = 1; i < 256; ++i ) {
        hsv.at<cv::Vec3f> ( i ) = cv::Vec3f ( ( i / 256.f ) * 360.f, 1.f,
                                              1.f );
    }

    cv::Mat rgb;
    cv::cvtColor ( hsv, rgb, cv::COLOR_HSV2RGB );

    rgb.convertTo ( rgb, CV_8U, 255 );

    cv::RNG rng ( 6893 );
    randShuffle ( rgb, 1, &rng );

    rgb.at<cv::Vec3b> ( 0 ) = cv::Vec3b ( 255, 255, 255 );

    return rgb;
}

void EMFusion::computeAssociationWeights() {
    bg_associationWeights.setTo ( 0, streams[0] );
    for ( const auto& obj : objects ) {
        associationWeights[obj.getID()].setTo ( 0, streams[obj.getID()] );
    }

    // Get individual Laplace probabilities
    background.computeAssociation ( points, pose, bg_associationWeights,
                                    streams[0] );
    for ( auto& obj : objects ) {
        obj.computeAssociation ( points, pose, associationWeights[obj.getID()],
                                 streams[obj.getID()] );
    }

    for ( auto& stream : streams ) {
        stream.second.waitForCompletion();
    }

    // Normalize
    bg_associationWeights.copyTo ( associationNorm );
    for ( const auto& assW : associationWeights ) {
        cv::cuda::add ( associationNorm, assW.second, associationNorm );
    }

    cv::cuda::divide ( bg_associationWeights, associationNorm,
                       bg_associationWeights, 1, -1, streams[0] );
    for ( const auto& obj : objects ) {
        cv::cuda::divide ( associationWeights[obj.getID()], associationNorm,
                           associationWeights[obj.getID()], 1, -1,
                           streams[obj.getID()] );
    }

    for ( auto& stream : streams ) {
        stream.second.waitForCompletion();
    }
}

void EMFusion::performTracking() {
    background.prepareTracking ( pose, streams[0] );
    for ( int i = 0; i < params.maxTrackingIter; ++i ) {
        background.computeGradients ( points );
        background.computeTSDFVals ( points );
        background.computeTSDFWeights ( points );
        background.computeHuberWeights();
        background.normalizeTSDFWeights();
        background.combineWeights ( bg_associationWeights );
        background.computeHessians();
        background.reduceHessians();
        background.computePoseUpdate ( points );
    }
    background.syncTrack ( pose );

    computeAssociationWeights();

    for ( auto& obj : objects ) {
        obj.prepareTracking ( pose, streams[obj.getID()] );
    }
    for ( int i = 0; i < params.maxTrackingIter; ++i ) {
        for ( auto& obj : objects ) {
            obj.computeGradients ( points );
        }
        for ( auto& obj : objects ) {
            obj.computeTSDFVals ( points );
        }
        for ( auto& obj : objects ) {
            obj.computeTSDFWeights ( points );
        }
        for ( auto& obj : objects ) {
            obj.computeHuberWeights();
        }
        for ( auto& obj : objects ) {
            obj.normalizeTSDFWeights();
        }
        for ( auto& obj : objects ) {
            obj.combineWeights ( associationWeights[obj.getID()] );
        }
        for ( auto& obj : objects ) {
            obj.computeHessians();
        }
        for ( auto& obj : objects ) {
            obj.reduceHessians();
        }
        for ( auto& obj : objects ) {
            obj.computePoseUpdate ( points );
        }
    }
    for ( auto &&obj : objects ) {
        obj.syncTrack ( pose );
    }
}

void EMFusion::raycast () {
    raylengths.setTo ( 0, streams[objects.size()] );
    bg_raylengths.setTo ( 0, streams[0] );
    vertices.setTo ( cv::Vec3f::all ( 0 ), streams[objects.size()] );
    bg_vertices.setTo ( cv::Vec3f::all ( 0 ), streams[0] );
    normals.setTo ( cv::Vec3f::all ( 0 ), streams[objects.size()] );
    bg_normals.setTo ( cv::Vec3f::all ( 0 ), streams[0] );
    modelSegmentation.setTo ( 0, streams[objects.size()] );
    bg_mask.setTo ( false, streams[0] );
    for ( auto& obj : objects ) {
        obj_raylengths[obj.getID()].setTo ( 0, streams[obj.getID()] );
        obj_vertices[obj.getID()].setTo ( cv::Vec3f::all ( 0 ),
                                          streams[obj.getID()] );
        obj_normals[obj.getID()].setTo ( cv::Vec3f::all ( 0 ),
                                         streams[obj.getID()] );
        obj_modelSegmentation[obj.getID()].setTo ( false,
                streams[obj.getID()] );
    }

    vis_objs.clear();

    background.raycast ( pose, params.intr, bg_raylengths, bg_vertices,
                         bg_normals, bg_mask, streams[0] );
    for ( auto& obj : objects ) {
        obj.raycast ( pose, params.intr, obj_raylengths[obj.getID()],
                      obj_vertices[obj.getID()], obj_normals[obj.getID()],
                      obj_modelSegmentation[obj.getID()],
                      streams[obj.getID()] );
    }

    for ( auto &&stream : streams ) {
        stream.second.waitForCompletion();
    }

    for ( const auto& obj : objects ) {
        cv::cuda::compare ( raylengths, 0, mask, cv::CMP_LE );
        cv::cuda::compare ( obj_raylengths[obj.getID()], raylengths, obj_mask,
                            cv::CMP_LT );
        cv::cuda::bitwise_or ( obj_mask, mask, mask );
        cv::cuda::bitwise_and ( obj_modelSegmentation[obj.getID()], mask,
                                mask );
        obj_raylengths[obj.getID()].copyTo ( raylengths, mask );
        obj_vertices[obj.getID()].copyTo ( vertices, mask );
        obj_normals[obj.getID()].copyTo ( normals, mask );
        modelSegmentation.setTo ( obj.getID(), mask );
    }

    cv::cuda::subtract ( raylengths, bg_raylengths, diffRaylengths, bg_mask );
    cv::cuda::compare ( diffRaylengths, 0.05f, takeBgMask, cv::CMP_GT );
    modelSegmentation.setTo ( 0, takeBgMask );
    cv::cuda::compare ( modelSegmentation, 0, noObjMask, cv::CMP_EQ );

    for ( const auto& obj : objects ) {
        cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                            cv::CMP_EQ );
        if ( cv::cuda::countNonZero (
                    obj_mask (
                        cv::Rect ( params.boundary, params.boundary,
                                   params.frameSize.width - 2 * params.boundary,
                                   params.frameSize.height - 2 * params.boundary
                                 )
                    )
                ) > params.visibilityThresh ) {
            vis_objs.insert ( obj.getID() );
        }
    }

    bg_vertices.copyTo ( vertices, noObjMask );
    bg_normals.copyTo ( normals, noObjMask );
}

int EMFusion::matchSegmentation ( const cv::cuda::GpuMat& new_seg,
                                  float& match_iou ) {
    int match_id = -1;

    for ( const auto& obj : objects ) {
        if ( find ( vis_objs.begin(), vis_objs.end(), obj.getID() )
                == vis_objs.end() ) {
            continue;
        }
        cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                            cv::CMP_EQ );
        cv::cuda::bitwise_and ( new_seg, obj_mask, seg_inter );
        cv::cuda::bitwise_or ( new_seg, obj_mask, seg_uni );

        const float iou =
            static_cast<float> ( cv::cuda::countNonZero ( seg_inter ) )
            / cv::cuda::countNonZero ( seg_uni );
        if ( iou > match_iou ) {
            match_iou = iou;
            match_id = obj.getID();
        }
    }

    if ( match_iou > params.matchIOUThresh ) {
        return match_id;
    }

    return -1;
}

cv::Vec3f EMFusion::updateObj ( ObjTSDF& obj, const cv::cuda::GpuMat& points,
                                const cv::cuda::GpuMat& seg_gpu,
                                const std::vector<double>& scores ) {
    obj.updateClassProbs ( scores );

    cv::cuda::bitwise_and ( validPoints, seg_gpu, mask );

    if ( cv::cuda::countNonZero ( mask ) == 0 ) {
        return cv::Vec3f::all ( 0.f );
    }

    cv::cuda::GpuMat obj_points, filteredPoints, objFilteredPoints;
    cuda::EMFusion::filterPoints ( points, mask, filteredPoints );

    cv::viz::Mesh obj_mesh = obj.getMesh();
    obj_points = cv::cuda::createContinuous (
                     1, obj_mesh.cloud.cols + filteredPoints.cols, CV_32FC3 );
    obj_points.colRange ( 0, obj_mesh.cloud.cols ).upload ( obj_mesh.cloud );

    cv::Affine3f obj_pose = obj.getPose().inv();
    cv::cuda::transformPoints ( filteredPoints, cv::Mat ( obj_pose.rvec().t() ),
                                cv::Mat ( obj_pose.translation().t() ),
                                objFilteredPoints );
    objFilteredPoints.copyTo ( obj_points.colRange (
                                   obj_mesh.cloud.cols, obj_points.cols ) );

    cv::Vec3f p10, p90;
    cuda::EMFusion::computePercentiles ( obj_points, p10, p90 );

    cv::Vec3f offset = obj.resize ( p10, p90, params.volPad );

    // Object pose might have changed leading to inconsistent mesh/pose
    // combinations
    obj_poses[obj.getID()][frameCount] = obj.getPose();

    return offset;
}

void EMFusion::integrateDepth () {
    background.integrate ( depth, bg_associationWeights, pose, params.intr,
                           streams[0] );
    for ( auto& obj : objects ) {
        if ( find ( vis_objs.begin(), vis_objs.end(),
                    obj.getID() ) == vis_objs.end() ) {
            continue;
        }
        obj.integrate ( depth, associationWeights[obj.getID()], pose,
                        params.intr, streams[obj.getID()] );
    }

    background.updateGradients ( streams[0] );
    for ( auto& obj : objects ) {
        if ( find ( vis_objs.begin(), vis_objs.end(),
                    obj.getID() ) == vis_objs.end() ) {
            continue;
        }
        obj.updateGradients ( streams[obj.getID()] );
    }

    for ( auto &&stream : streams ) {
        stream.second.waitForCompletion();
    }
}

void EMFusion::integrateMasks (
    const std::map<int, cv::cuda::GpuMat>& matches ) {
    for ( auto& obj : objects ) {
        if ( matches.count ( obj.getID() ) ) {
            // Find occluded pixels in model and do not integrate Fg prob
            // for those
            cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                                cv::CMP_EQ );
            cv::cuda::subtract ( obj_modelSegmentation[obj.getID()], obj_mask,
                                 obj_mask );

            obj.integrateMask ( matches.at ( obj.getID() ), obj_mask, pose,
                                params.intr );
        }
    }
}

void EMFusion::createObj ( const int id ) {
    obj_raylengths[id] =
        cv::cuda::createContinuous ( params.frameSize, CV_32FC1 );
    obj_vertices[id] =
        cv::cuda::createContinuous ( params.frameSize, CV_32FC3 );
    obj_normals[id] = cv::cuda::createContinuous ( params.frameSize, CV_32FC3 );
    obj_modelSegmentation[id] =
        cv::cuda::createContinuous ( params.frameSize, CV_8UC1 );
    obj_modelSegmentation[id].setTo ( 0 );
    associationWeights[id] =
        cv::cuda::createContinuous ( params.frameSize, CV_32FC1 );
    associationWeights[id].setTo ( 1 );
}

void EMFusion::cleanUpObjs ( int numInstances,
                             const std::map<int, cv::cuda::GpuMat>& matches ) {
    std::vector<int> spurious;
    if ( numInstances > 0 ) {
        for ( auto& obj : objects ) {
            if ( obj.getExProb() < params.existenceThresh ) {
                spurious.push_back ( obj.getID() );
                std::cout << "Deleting Object " << obj.getID()
                          << " because of low existence probability!"
                          << std::endl;
            }
        }
    }

    for ( auto id : vis_objs ) {
        obj_modelSegmentation[id].copyTo ( obj_mask );
        if ( matches.count ( id ) ) {
            cv::cuda::bitwise_or ( obj_mask, matches.at ( id ), obj_mask );
        }

        if ( params.assocThresh * cv::cuda::countNonZero ( obj_mask ) >
                cv::cuda::sum ( associationWeights[id], obj_mask ) [0] ) {
            spurious.push_back ( id );
            std::cout << "Deleting Object " << id
                      << " because association does not fit to mask!"
                      << std::endl;
        }
    }

    for ( auto it = objects.begin(); it != objects.end(); ) {
        if ( find ( spurious.begin(), spurious.end(), it->getID() )
                != spurious.end() ||
                find ( vis_objs.begin(), vis_objs.end(), it->getID() )
                == vis_objs.end() ) {
            if ( find ( vis_objs.begin(), vis_objs.end(), it->getID() )
                    == vis_objs.end() ) {
                std::cout << "Deleting Object " << it->getID()
                          << " because it is not visible!" << std::endl;
            }
            deleteObj ( it->getID() );
            if ( saveOutput &&
                    ! ( params.ignore_person &&
                        emf::MaskRCNN::getClassName ( it->getClassID() )
                        == "person" ) ) {
                meshes[it->getID()] = it->getMesh();
                if ( expVols ) {
                    tsdfs[it->getID()] = it->getTSDF();
                    intWeights[it->getID()] = it->getWeightsVol();
                    fgProbs[it->getID()] = it->getFgProbVol();
                    meta[it->getID()] = std::make_pair ( it->getVolumeRes(),
                                                         it->getVoxelSize() );
                }
            }
            it = objects.erase ( it );
        } else {
            ++it;
        }
    }
}

void EMFusion::deleteObj ( const int id ) {
    streams.erase ( id );
    obj_raylengths.erase ( id );
    obj_vertices.erase ( id );
    obj_normals.erase ( id );
    obj_modelSegmentation.erase ( id );
    associationWeights.erase ( id );
}

void EMFusion::writePoses ( const boost::filesystem::path& p ) {
    writePoseFile ( ( p / "poses-cam.txt" ).string(), poses );

    for ( const auto& obj_pose : obj_poses ) {
        std::stringstream filename;
        filename << "poses-" << obj_pose.first << ".txt";
        writePoseFile ( ( p / filename.str() ).string(), obj_pose.second );
    }

    std::map<int, std::map<int, cv::Affine3f>> obj_poses_final =
            addPoseOffsets ( obj_poses, obj_pose_offsets );
    for ( const auto& obj_pose : obj_poses_final ) {
        std::stringstream filename;
        filename << "poses-" << obj_pose.first << "-corrected.txt";
        writePoseFile ( ( p / filename.str() ).string(), obj_pose.second );
    }
}

void EMFusion::writeRenderings ( const boost::filesystem::path& p ) {
    boost::filesystem::path output = p / "output";
    boost::filesystem::create_directories ( output );

    for ( const auto& rend : renderings ) {
        writeImage ( output, rend.first, rend.second );
    }
}

void EMFusion::writeMeshVis ( const boost::filesystem::path& p ) {
    boost::filesystem::path mesh_vis_out = p / "mesh_vis_out";
    boost::filesystem::create_directories ( mesh_vis_out );

    for ( const auto& mesh_v : mesh_vis ) {
        writeImage ( mesh_vis_out, mesh_v.first, mesh_v.second );
    }
}

void EMFusion::writeMasks ( const boost::filesystem::path& p ) {
    boost::filesystem::path masks = p / "masks";
    boost::filesystem::create_directories ( masks );

    for ( const auto& mask : mask_vis ) {
        writeImage ( masks, mask.first, mask.second );
    }
}

void EMFusion::writeAssocs ( const boost::filesystem::path& p ) {
    boost::filesystem::path assoc_w = p / "assoc_weights";
    boost::filesystem::create_directories ( assoc_w );

    boost::filesystem::path assoc_w_bg = assoc_w / "bg";
    boost::filesystem::create_directories ( assoc_w_bg );

    boost::filesystem::path assoc_w_bg_preTrack = assoc_w_bg / "preTrack";
    boost::filesystem::create_directories ( assoc_w_bg_preTrack );

    for ( const auto& assoc : bg_assocWeight_preTrack ) {
        writeImage ( assoc_w_bg_preTrack, assoc.first, assoc.second );
    }

    boost::filesystem::path assoc_w_bg_postTrack = assoc_w_bg / "postTrack";
    boost::filesystem::create_directories ( assoc_w_bg_postTrack );

    for ( const auto& assoc : bg_assocWeight_postTrack ) {
        writeImage ( assoc_w_bg_postTrack, assoc.first, assoc.second );
    }

    for ( const auto& obj_assocw : obj_assocWeights_preTrack ) {
        std::stringstream obj_id;
        obj_id << obj_assocw.first;
        boost::filesystem::path assoc_w_obj = assoc_w / obj_id.str();
        boost::filesystem::create_directories ( assoc_w_obj );

        boost::filesystem::path assoc_w_obj_preTrack = assoc_w_obj / "preTrack";
        boost::filesystem::create_directories ( assoc_w_obj_preTrack );

        for ( const auto& assoc : obj_assocw.second ) {
            writeImage ( assoc_w_obj_preTrack, assoc.first, assoc.second );
        }
    }

    for ( const auto& obj_assocw : obj_assocWeights_postTrack ) {
        std::stringstream obj_id;
        obj_id << obj_assocw.first;
        boost::filesystem::path assoc_w_obj = assoc_w / obj_id.str();
        boost::filesystem::create_directories ( assoc_w_obj );

        boost::filesystem::path assoc_w_obj_postTrack =
            assoc_w_obj / "postTrack";
        boost::filesystem::create_directories ( assoc_w_obj_postTrack );

        for ( const auto& assoc : obj_assocw.second ) {
            writeImage ( assoc_w_obj_postTrack, assoc.first, assoc.second );
        }
    }
}

void EMFusion::writeHuberWeights ( const boost::filesystem::path& p ) {
    boost::filesystem::path huber_w = p / "huber_weights";
    boost::filesystem::create_directories ( huber_w );

    boost::filesystem::path huber_w_bg = huber_w / "bg";
    boost::filesystem::create_directories ( huber_w_bg );

    for ( const auto& huber : bg_huberWeights ) {
        writeImage ( huber_w_bg, huber.first, huber.second );
    }

    for ( const auto& obj_huber : obj_huberWeights ) {
        std::stringstream obj_id;
        obj_id << obj_huber.first;
        boost::filesystem::path huber_w_obj = huber_w / obj_id.str();
        boost::filesystem::create_directories ( huber_w_obj );

        for ( const auto& huber : obj_huber.second ) {
            writeImage ( huber_w_obj, huber.first, huber.second );
        }
    }
}

void EMFusion::writeTrackWeights ( const boost::filesystem::path& p ) {
    boost::filesystem::path track_w = p / "track_weights";
    boost::filesystem::create_directories ( track_w );

    boost::filesystem::path track_w_bg = track_w / "bg";
    boost::filesystem::create_directories ( track_w_bg );

    for ( const auto& track : bg_trackWeights ) {
        writeImage ( track_w_bg, track.first, track.second );
    }

    for ( const auto& obj_track : obj_trackWeights ) {
        std::stringstream obj_id;
        obj_id << obj_track.first;
        boost::filesystem::path track_w_obj = track_w / obj_id.str();
        boost::filesystem::create_directories ( track_w_obj );
        for ( const auto& track : obj_track.second ) {
            writeImage ( track_w_obj, track.first, track.second );
        }
    }
}

void EMFusion::writeFgProbs ( const boost::filesystem::path& p ) {
    boost::filesystem::path fg_probs = p / "fg_probs";
    boost::filesystem::create_directories ( fg_probs );

    for ( const auto& fg_prob : obj_fgProbs ) {
        std::stringstream obj_id;
        obj_id << fg_prob.first;
        boost::filesystem::path fg_prob_obj = fg_probs / obj_id.str();
        boost::filesystem::create_directories ( fg_prob_obj );
        for ( const auto& fg : fg_prob.second ) {
            writeImage ( fg_prob_obj, fg.first, fg.second );
        }
    }
}

void EMFusion::writeMeshes ( const boost::filesystem::path& p ) {
    // Write final mesh for each volume in main directory
    auto mesh = background.getMesh();
    writeMesh ( mesh, ( p / "mesh_bg.ply" ).string() );

    for ( const auto& mesh : meshes ) {
        std::stringstream filename;
        filename << "mesh_" << mesh.first << ".ply";
        writeMesh ( mesh.second, ( p / filename.str() ).string() );
    }

    // Write frame meshes to folder
    boost::filesystem::path frame_meshes_dir = p / "frame_meshes";
    boost::filesystem::create_directories ( frame_meshes_dir );

    for ( const auto& bg_mesh : frame_meshes ) {
        boost::filesystem::path bg_mesh_path = frame_meshes_dir / "bg";
        boost::filesystem::create_directories ( bg_mesh_path );
        std::stringstream filename;
        filename << std::setfill ( '0' ) << std::setw ( 4 ) << bg_mesh.first
                 << ".ply";
        writeMesh ( bg_mesh.second,
                    ( bg_mesh_path / filename.str() ).string() );
    }

    for ( const auto& obj_mesh : frame_obj_meshes ) {
        std::stringstream obj_id;
        obj_id << obj_mesh.first;
        boost::filesystem::path obj_mesh_path = frame_meshes_dir / obj_id.str();
        boost::filesystem::create_directories ( obj_mesh_path );
        for ( const auto& mesh : obj_mesh.second ) {
            std::stringstream filename;
            filename << std::setfill ( '0' ) << std::setw ( 4 ) << mesh.first
                     << ".ply";
            writeMesh ( mesh.second,
                        ( obj_mesh_path / filename.str() ).string() );
        }
    }
}

void EMFusion::writeTSDFs ( const boost::filesystem::path& p ) {
    boost::filesystem::path tsdfs_dir = p / "tsdfs";
    boost::filesystem::create_directories ( tsdfs_dir );

    cv::Mat bg_tsdf = background.getTSDF();
    writeVolume ( ( tsdfs_dir / "bg_tsdf.bin" ).string(), bg_tsdf,
                  background.getVolumeRes(), background.getVoxelSize() );

    for ( const auto& tsdf : tsdfs ) {
        std::stringstream filename;
        filename << "tsdf_" << tsdf.first << ".bin";
        writeVolume ( ( tsdfs_dir / filename.str() ).string(),
                      tsdf.second, meta[tsdf.first].first,
                      meta[tsdf.first].second );
    }

    for ( const auto& weights : intWeights ) {
        std::stringstream filename;
        filename << "weights_" << weights.first << ".bin";
        writeVolume ( ( tsdfs_dir / filename.str() ).string(),
                      weights.second, meta[weights.first].first,
                      meta[weights.first].second );
    }

    for ( const auto& fg : fgProbs ) {
        std::stringstream filename;
        filename << "fgProbs_" << fg.first << ".bin";
        writeVolume ( ( tsdfs_dir / filename.str() ).string(),
                      fg.second, meta[fg.first].first, meta[fg.first].second );
    }

}

std::map<int, std::map<int, cv::Affine3f>> EMFusion::addPoseOffsets (
        const std::map<int, std::map<int, cv::Affine3f>>& poses,
        const std::map<int, std::map<int, cv::Vec3f>>& offsets ) {
    std::map<int, std::map<int, cv::Affine3f>> cleanedPoses;
    for ( const auto& obj_poses : poses ) {
        cv::Vec3f cumOffset = cv::Vec3f::all ( 0.f );
        for ( const auto& pose : obj_poses.second ) {
            if ( offsets.count ( obj_poses.first ) &&
                    offsets.at ( obj_poses.first ).count ( pose.first ) )
                cumOffset -= offsets.at ( obj_poses.first ).at ( pose.first );
            cleanedPoses[obj_poses.first][pose.first] =
                pose.second.translate ( pose.second.rotation() * cumOffset );
        }
    }

    return cleanedPoses;
}

void EMFusion::writePoseFile ( const std::string& filename,
                               const std::map<int, cv::Affine3f>& poses ) {
    std::ofstream pose_file ( filename );
    for ( const auto& pose : poses ) {
        Sophus::Matrix4f se3_mat;
        cv2eigen ( pose.second.matrix, se3_mat );
        Sophus::SE3f se3_pose ( se3_mat );
        pose_file << pose.first << " " << se3_pose.translation().x()
                  << " " << se3_pose.translation().y() << " "
                  << se3_pose.translation().z() << " "
                  << se3_pose.unit_quaternion().x() << " "
                  << se3_pose.unit_quaternion().y() << " "
                  << se3_pose.unit_quaternion().z() << " "
                  << se3_pose.unit_quaternion().w() << std::endl;
    }
    pose_file.close();
}

void EMFusion::writeImage ( const boost::filesystem::path& path, int id,
                            const cv::Mat& image ) {
    std::stringstream filename;
    filename << std::setfill ( '0' ) << std::setw ( 4 ) << id << ".png";
    imwrite ( ( path / filename.str() ).string(), image );
}

void EMFusion::writeMesh ( const cv::viz::Mesh& mesh,
                           const std::string& filename ) {
    FILE* file = fopen ( filename.c_str(), "w" );
    if ( !file )
        throw std::runtime_error ( "Could not write ply file: " + filename );

    // write statistics
    fprintf ( file, "ply\n" );
    fprintf ( file, "format ascii 1.0\n" );
    fprintf ( file, "element vertex %d\n", mesh.cloud.cols );
    fprintf ( file, "property float x\n" );
    fprintf ( file, "property float y\n" );
    fprintf ( file, "property float z\n" );
    fprintf ( file, "property float nx\n" );
    fprintf ( file, "property float ny\n" );
    fprintf ( file, "property float nz\n" );
    fprintf ( file, "element face %d\n", mesh.polygons.cols / 4 );
    fprintf ( file, "property list uchar int vertex_index\n" );
    fprintf ( file, "end_header\n" );

    // write vertices
    for ( int i = 0; i < mesh.cloud.cols; ++i ) {
        cv::Vec3f v = mesh.cloud.at<cv::Vec3f> ( 0, i );
        cv::Vec3f n = mesh.normals.at<cv::Vec3f> ( 0, i );
        fprintf ( file, "%f %f %f %f %f %f\n", v[0], v[1], v[2], n[0], n[1],
                  n[2] );
    }

    // write triangles
    for ( int i = 0; i < mesh.polygons.cols; i += 4 )
        fprintf ( file, "%d %d %d %d\n",
                  mesh.polygons.at<int> ( 0, i ),
                  mesh.polygons.at<int> ( 0, i+1 ),
                  mesh.polygons.at<int> ( 0, i+2 ),
                  mesh.polygons.at<int> ( 0, i+3 ) );

    fclose ( file );
}

void EMFusion::writeVolume ( const std::string& filename, const cv::Mat& vol,
                             const cv::Vec3i& resolution, float voxelSize ) {
    size_t elemSize = vol.elemSize();
    std::ofstream ofile ( filename, std::ios::binary );
    ofile.write ( ( char * ) resolution.val, 3 * sizeof ( int ) );
    ofile.write ( ( char * ) &elemSize, sizeof ( size_t ) );
    ofile.write ( ( char * ) &voxelSize, sizeof ( float ) );
    ofile.write ( ( char * ) vol.data, vol.total() * elemSize );
    ofile.close();
    if ( !ofile.good() )
        std::cerr << "Error writing " << filename << std::endl;
}

}
