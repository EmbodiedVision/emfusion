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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>

#include "EMFusion/core/EMFusion.h"
#include "EMFusion/utils/RGBDReader.h"
#include "EMFusion/utils/ImageReader.h"
#include "EMFusion/utils/TUMRGBDReader.h"

namespace po = boost::program_options;

namespace cv {
template<typename T>
std::vector<T> getValues ( const std::vector<std::string>& values ) {
    std::vector<T> tvalues;

    for ( const auto& val : values ) {
        std::vector<std::string> splits;
        boost::split ( splits, val, boost::is_any_of ( ", " ) );
        for ( const auto& str : splits ) {
            std::stringstream ss ( str );
            copy ( std::istream_iterator<T> ( ss ),
                   std::istream_iterator<T>(), back_inserter ( tvalues ) );
            if ( !ss.eof() )
                throw po::validation_error (
                    po::validation_error::invalid_option_value
                );
        }
    }
    return tvalues;
}

template<typename T>
void validate ( boost::any& v, const std::vector<std::string>& values,
                cv::Size_<T>*, int ) {
    std::vector<T> tvalues = getValues<T> ( values );

    if ( tvalues.size() != 2 )
        throw po::validation_error (
            po::validation_error::invalid_option_value
        );

    cv::Size_<T> size ( tvalues[0], tvalues[1] );
    v = size;
}

template<typename T, int cn>
void validate ( boost::any& v, const std::vector<std::string>& values,
                cv::Vec<T, cn>*, int ) {
    std::vector<T> tvalues = getValues<T> ( values );

    if ( tvalues.size() != cn )
        throw po::validation_error (
            po::validation_error::invalid_option_value
        );

    cv::Vec<T, cn> vec ( &tvalues[0] );
    v = vec;
}

template<typename T>
void validate ( boost::any& v, const std::vector<std::string>& values,
                cv::Affine3<T>*, int ) {
    std::vector<T> tvalues = getValues<T> ( values );

    if ( tvalues.size() != 3 )
        throw po::validation_error (
            po::validation_error::invalid_option_value
        );

    cv::Affine3<T> vec = cv::Affine3<T>().translate (
                             cv::Vec<T,3> ( &tvalues[0] ) );
    v = vec;
}
}

void main_loop ( const emf::Params& globalParams,
                 const std::unique_ptr<RGBDReader>& reader,
                 const std::string& exportpath = "",
                 bool exp_frame_meshes = false, bool exp_vols = false,
                 const std::string& maskpath = "",
                 const bool background = false,
                 const bool show_3d_vis = false ) {
    emf::EMFusion emf ( globalParams );

    if ( exportpath.length() ) {
        emf.setupOutput ( exp_frame_meshes, exp_vols );
    }
    if ( maskpath.length() ) {
        emf.usePreprocMasks ( maskpath );
    }

    cv::viz::Viz3d* window = NULL;

    if ( show_3d_vis && !background ) {
        window = new cv::viz::Viz3d ( "EM-Fusion: 3D view" );
        if ( background ) {
            window->setOffScreenRendering();
            window->getScreenshot();
        } else {
            window->spinOnce ( 1 );
        }
        window->setCamera ( cv::viz::Camera ( globalParams.intr,
                                              globalParams.frameSize ) );
        window->setViewerPose (
            cv::Affine3d().translate ( cv::Vec3f ( 0, 0, -1 ) ) );
        window->setWindowSize ( cv::Size ( 1024, 768 ) );
    }

    int64 prevTime = cv::getTickCount();

    int frameCount = 0;

    while ( reader->moreFrames() ) {

        RGBD frame = reader->getNextFrame();

        cv::Mat depth = frame.getDepth();

        cv::Mat depth_tmp, depth_show;
        double maxDepth = 3.0;
        cv::minMaxLoc ( depth, NULL, &maxDepth );
        depth.convertTo ( depth_tmp, CV_8U, 255.0/maxDepth );

        applyColorMap ( depth_tmp, depth_show, cv::COLORMAP_PARULA );

        emf.processFrame ( frame );

        cv::Mat rendered = cv::Mat::zeros ( frame.getSize(), CV_8UC3 );
        cv::Mat masks = cv::Mat::zeros ( frame.getSize(), CV_8UC3 );
        emf.render ( rendered, window );

        if ( !background ) {
            if ( show_3d_vis )
                window->spinOnce ( 1 );

            emf.getLastMasks ( masks );

            cv::Mat visin, visout, vis;
            hconcat ( frame.getRGB(), depth_show, visin );
            hconcat ( masks, rendered, visout );
            vconcat ( visin, visout, vis );

            int64 newTime = cv::getTickCount();
            putText ( vis, cv::format (
                          "FPS: %2d press P to pause, Q to quit",
                          ( int ) ( cv::getTickFrequency()
                                    / ( newTime - prevTime ) ) ),
                      cv::Point ( 0, vis.rows-1 ), cv::FONT_HERSHEY_SIMPLEX, 1,
                      cv::Scalar ( 0, 255, 255 ) );
            prevTime = newTime;


            namedWindow ( "EM-Fusion", cv::WINDOW_NORMAL );
            resizeWindow ( "EM-Fusion", frame.getSize() );
            imshow ( "EM-Fusion", vis );


            char key = cv::waitKey ( 1 );
            if ( key == 'p' || key == 'P' ) {
                key = cv::waitKey ( 0 );
            }
            if ( key == 'q' || key == 'Q' ) {
                break;
            }
        }

        ++frameCount;
    }

    if ( window ) {
        window->close();
        delete window;
        window = NULL;
    }

    if ( exportpath.length() ) {
        std::cout << "Writing outputs to: " << exportpath << std::endl;
        emf.writeResults ( exportpath );
    } else if ( !background ) {
        std::cout << "Finished processing, press any key to end the program!"
                  << std::endl;
        cv::waitKey();
    }
    std::cout << "Program ended successfully!" << std::endl;
}

int main ( int argc, char **argv ) {
    po::options_description options ( "EM-Fusion: Dynamic tracking and Mapping"
                                      " from RGB-D data" );

    po::options_description required ( "One of these options is required" );
    required.add_options ()
    ( "tumdir,t", po::value<std::string>(),
      "Directory containing RGB-D data in the TUM format" )
    ( "dir,d", po::value<std::string>(),
      "Directory containing color and depth images" )
    ;

    po::options_description dir_options ( "Possibly needed when using"
                                          " \"--dir\" above" );
    dir_options.add_options()
    ( "colordir", po::value<std::string>()->default_value ( "colour" ),
      "Subdirectory containing color images named Color*.png. Needed if"
      " different from \"colour\"" )
    ( "depthdir", po::value<std::string>()->default_value ( "depth" ),
      "Subdirectory containing depth images named Depth*.exr. Needed if"
      " different from \"depth\"" )
    ;

    po::options_description optional ( "Optional inputs" );
    optional.add_options ()
    ( "help,h", "Print this help" )
    ( "exportdir,e", po::value<std::string>(), "Directory for storing results" )
    ( "export-frame-meshes", po::bool_switch(),
      "Whether to export meshes for every frame. Needs a lot of RAM if there "
      "are many objects." )
    ( "export-volumes", po::bool_switch(),
      "Whether to output TSDF volumes with weights and foreground "
      "probabilities. Needs a lot of RAM if there are many objects." )
    ( "background", po::bool_switch(),
      "Whether to run this program without live output (without -e it won't"
      "be possible to examine results)" )
    ( "3d-vis", po::bool_switch(),
      "Whether to show 3D visualizations with meshes and bounding boxes"
    )
    ( "configfile,c", po::value<std::string>(),
      "Path to a configuration file containing experiment parameters" )
    ( "maskdir,m", po::value<std::string>(),
      "Directory containing preprocessed Mask R-CNN results" )
    ;

    options.add ( required ).add ( dir_options ).add ( optional );
    po::variables_map result;
    po::store ( po::parse_command_line ( argc, argv, options ), result );

    if ( result.count ( "help" ) || ! ( result.count ( "tumdir" ) ||
                                        result.count ( "dir" ) ) ) {
        std::cout << options << std::endl;
        exit ( 1 );
    }

    emf::Params globalParams; // configuration for global model
    if ( result.count ( "configfile" ) ) {
        std::string configpath = result["configfile"].as<std::string>();
        po::options_description params ( "Params" );
        params.add_options()
        ( "Params.frameSize",
          po::value<cv::Size> ( &globalParams.frameSize )->multitoken() )

        ( "Params.intr.fx",
          po::value<float> ( &globalParams.intr ( 0, 0 ) ) )
        ( "Params.intr.fy",
          po::value<float> ( &globalParams.intr ( 1, 1 ) ) )
        ( "Params.intr.cx",
          po::value<float> ( &globalParams.intr ( 0, 2 ) ) )
        ( "Params.intr.cy",
          po::value<float> ( &globalParams.intr ( 1, 2 ) ) )

        ( "Params.bilateral_sigma_depth",
          po::value<float> ( &globalParams.bilateral_sigma_depth ) )
        ( "Params.bilateral_sigma_spatial",
          po::value<float> ( &globalParams.bilateral_sigma_spatial ) )
        ( "Params.bilateral_kernel_size",
          po::value<int> ( &globalParams.bilateral_kernel_size ) )

        ( "Params.globalVolumeDims",
          po::value<cv::Vec3i> ( &globalParams.globalVolumeDims )->multitoken()
        )
        ( "Params.globalVoxelSize",
          po::value<float> ( &globalParams.globalVoxelSize ) )
        ( "Params.globalRelTruncDist",
          po::value<float> ( &globalParams.globalRelTruncDist ) )
        ( "Params.objVolumeDims",
          po::value<cv::Vec3i> ( &globalParams.objVolumeDims )->multitoken() )
        ( "Params.objRelTruncDist",
          po::value<float> ( &globalParams.objRelTruncDist ) )

        ( "Params.volumePose",
          po::value<cv::Affine3f> ( &globalParams.volumePose )->multitoken() )

        ( "Params.volPad",
          po::value<float> ( &globalParams.volPad ) )

        ( "Params.maxTrackingIter",
          po::value<int> ( &globalParams.maxTrackingIter ) )

        ( "Params.maskRCNNFrames",
          po::value<int> ( &globalParams.maskRCNNFrames ) )

        ( "Params.existenceThresh",
          po::value<float> ( &globalParams.existenceThresh ) )

        ( "Params.volIOUThresh",
          po::value<float> ( &globalParams.volIOUThresh ) )

        ( "Params.matchIOUThresh",
          po::value<float> ( &globalParams.matchIOUThresh ) )

        ( "Params.distanceThresh",
          po::value<float> ( &globalParams.distanceThresh ) )

        ( "Params.visibilityThresh",
          po::value<int> ( &globalParams.visibilityThresh ) )

        ( "Params.assocThresh",
          po::value<float> ( &globalParams.assocThresh ) )

        ( "Params.boundary",
          po::value<int> ( &globalParams.boundary ) )

        ( "Params.tsdfParams.tau",
          po::value<float> ( &globalParams.tsdfParams.tau ) )
        ( "Params.tsdfParams.eps1",
          po::value<float> ( &globalParams.tsdfParams.eps1 ) )
        ( "Params.tsdfParams.eps2",
          po::value<float> ( &globalParams.tsdfParams.eps2 ) )
        ( "Params.tsdfParams.nu_init",
          po::value<float> ( &globalParams.tsdfParams.nu_init ) )

        ( "Params.tsdfParams.huberThresh",
          po::value<float> ( &globalParams.tsdfParams.huberThresh ) )
        ( "Params.tsdfParams.maxTSDFWeight",
          po::value<float> ( &globalParams.tsdfParams.maxTSDFWeight ) )

        ( "Params.tsdfParams.assocSigma",
          po::value<float> ( &globalParams.tsdfParams.assocSigma ) )
        ( "Params.tsdfParams.alpha",
          po::value<float> ( &globalParams.tsdfParams.alpha ) )
        ( "Params.tsdfParams.uniPrior",
          po::value<float> ( &globalParams.tsdfParams.uniPrior ) )

        ( "Params.ignore_person",
          po::value<bool> ( &globalParams.ignore_person ) )

        ( "Params.MaskRCNNParams.FILTER_CLASSES",
          po::value<std::vector<std::string>> ( &globalParams.FILTER_CLASSES ) )
        ( "Params.MaskRCNNParams.STATIC_OBJECTS",
          po::value<std::vector<std::string>> ( &globalParams.STATIC_OBJECTS ) )
        ;

        po::variables_map cfg;
        po::store ( po::parse_config_file<char> ( configpath.c_str(), params ),
                    cfg );
        po::notify ( cfg );
    }

    std::unique_ptr<RGBDReader> reader;
    bool readerReady = false;

    if ( result.count ( "tumdir" ) ) {
        std::string dirpath = result["tumdir"].as<std::string>();
        if ( dirpath.length() ) {
            dirpath += "/";

            reader = std::unique_ptr<RGBDReader> (
                         new TUMRGBDReader ( dirpath ) );

            readerReady = true;
        }
    }

    if ( result.count ( "dir" ) ) {
        std::string dirpath = result["dir"].as<std::string>();
        if ( dirpath.length() ) {
            dirpath += "/";

            reader = std::unique_ptr<RGBDReader> (
                         new ImageReader ( dirpath,
                                           result["colordir"].as<std::string>(),
                                           result["depthdir"].as<std::string>()
                                         ) );

            readerReady = true;

            std::ifstream calibstr ( dirpath + "calibration.txt" );
            if ( calibstr.is_open() ) {
                calibstr >> globalParams.intr ( 0, 0 )
                         >> globalParams.intr ( 1, 1 )
                         >> globalParams.intr ( 0, 2 )
                         >> globalParams.intr ( 1, 2 );

                calibstr >> globalParams.frameSize.width
                         >> globalParams.frameSize.height;
            }
            calibstr.close();
        }
    }

    std::string expath = "";
    if ( result.count ( "exportdir" ) &&
            result["exportdir"].as<std::string>().length() ) {
        expath = result["exportdir"].as<std::string>();
    }
    std::string maskpath = "";
    if ( result.count ( "maskdir" ) &&
            result["maskdir"].as<std::string>().length() ) {
        maskpath = result["maskdir"].as<std::string>();
    }

    if ( readerReady ) {
        reader->init();

        main_loop ( globalParams, reader, expath,
                    result["export-frame-meshes"].as<bool>(),
                    result["export-volumes"].as<bool>(), maskpath,
                    result["background"].as<bool>(),
                    result["3d-vis"].as<bool>() );
    }
}
