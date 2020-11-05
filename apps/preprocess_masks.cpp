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
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/opencv.hpp>

#include "EMFusion/core/data.h"
#include "EMFusion/core/MaskRCNN.h"
#include "EMFusion/utils/RGBDReader.h"
#include "EMFusion/utils/ImageReader.h"
#include "EMFusion/utils/TUMRGBDReader.h"

namespace po = boost::program_options;

int main ( int argc, char **argv ) {

    po::options_description options ( "Preprocess RGB-D datasets with"
                                      " Mask R-CNN" );

    po::options_description required ( "One input option and the output option"
                                       " is required" );
    required.add_options ()
    ( "tumdir,t", po::value<std::string>(),
      "Directory containing RGB-D data in the TUM format" )
    ( "dir,d", po::value<std::string>(),
      "Directory containing color and depth images" )
    ( "maskdir,m", po::value<std::string>(),
      "Where to store the generated masks" )
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
    ( "configfile,c", po::value<std::string>(),
      "Path to a configuration file containing experiment parameters" )
    ;

    options.add ( required ).add ( dir_options ).add ( optional );
    po::variables_map result;
    po::store ( po::parse_command_line ( argc, argv, options ), result );

    if ( result.count ( "help" ) || ! ( result.count ( "tumdir" ) ||
                                        result.count ( "dir" ) )
            || ! result.count ( "maskdir" ) ) {
        std::cout << options << std::endl;
        exit ( 0 );
    }

    emf::Params globalParams;
    if ( result.count ( "configfile" ) ) {
        std::string configpath = result["configfile"].as<std::string>();
        po::options_description params ( "Params" );
        params.add_options()
        ( "Params.maskRCNNFrames",
          po::value<int> ( &globalParams.maskRCNNFrames ) )

        ( "Params.MaskRCNNParams.FILTER_CLASSES",
          po::value<std::vector<std::string>> ( &globalParams.FILTER_CLASSES ) )
        ( "Params.MaskRCNNParams.STATIC_OBJECTS",
          po::value<std::vector<std::string>> ( &globalParams.STATIC_OBJECTS ) )
        ;

        po::variables_map cfg;
        po::store ( po::parse_config_file<char> ( configpath.c_str(), params,
                    true ), cfg );
        po::notify ( cfg );
    }

    std::unique_ptr<RGBDReader> reader;
    bool readerReady = false;

    if ( result.count ( "tumdir" ) ) {
        std::string dirpath = result["tumdir"].as<std::string>();
        if ( dirpath.length() ) {
            dirpath += "/";
            TUMRGBDReader* tumReader = new TUMRGBDReader ( dirpath );

            reader = std::unique_ptr<RGBDReader> ( tumReader );

            readerReady = true;
        }
    }

    if ( result.count ( "dir" ) ) {
        std::string dirpath = result["dir"].as<std::string>();
        if ( dirpath.length() ) {
            dirpath += "/";
            ImageReader* imReader =
                new ImageReader ( dirpath, result["colordir"].as<std::string>(),
                                  result["depthdir"].as<std::string>() );

            reader = std::unique_ptr<RGBDReader> ( imReader );

            readerReady = true;
        }
    }

    if ( readerReady ) {
        std::string maskdir = result["maskdir"].as<std::string>();
        boost::filesystem::path maskpath ( maskdir );
        if ( !boost::filesystem::exists ( maskpath ) )
            boost::filesystem::create_directories ( maskpath );
        reader->init();
        emf::MaskRCNN maskrcnn ( globalParams.FILTER_CLASSES,
                                 globalParams.STATIC_OBJECTS );

        int frameCount = 0;
        while ( reader->moreFrames() ) {
            RGBD frame = reader->getNextFrame();

            if ( frameCount % globalParams.maskRCNNFrames == 0 ) {
                std::cout << "Processing frame " << frameCount << std::endl;
                cv::Mat rgb ( frame.getRGB() );
                std::stringstream filename;
                filename << maskdir << "/" << "Mask" << std::setw ( 4 )
                         << std::setfill ( '0' ) << frameCount << ".plk";

                maskrcnn.preprocess ( rgb, filename.str() );
            }
            frameCount++;
        }
    }
}
