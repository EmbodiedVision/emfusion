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
#include "EMFusion/utils/ImageReader.h"

ImageReader::ImageReader ( std::string basepath_,
                           std::string colordir_,
                           std::string depthdir_ )
    : RGBDReader ( basepath_ ), colorpath ( basepath_ + colordir_ ),
      depthpath ( basepath_ + depthdir_ ) {
}


void ImageReader::init() {
    std::cout << "Reading from " << path << std::endl;
    numFrames = countFiles();
    minBufferSize = frameRate;

    startBufferedRead();
}

int ImageReader::countFiles() {
    fs::path color ( colorpath );
    fs::path depth ( depthpath );

    if ( fs::is_directory ( color ) && fs::is_directory ( depth ) ) {
        fs::directory_iterator end;

        int rgbs = 0, depths = 0;

        for ( fs::directory_iterator iter ( color ); iter != end;
                ++iter )
            if ( iter->path().extension().string() == ".png" ) {
                ++rgbs;
            }
        for ( fs::directory_iterator iter ( depth ); iter != end;
                ++iter )
            if ( iter->path().extension().string() == ".exr" ) {
                ++depths;
            }

        if ( rgbs != depths ) {
            throw std::runtime_error (
                "Different number of rgb and depth files!" );
        }

        currBufferIndex = 0;
        std::stringstream rgbstr;
        std::stringstream depthstr;
        rgbstr << colorpath << "/Color" << std::setfill ( '0' )
               << std::setw ( 4 ) << currBufferIndex << ".png";
        depthstr << depthpath << "/Depth" << std::setfill ( '0' )
                 << std::setw ( 4 ) << currBufferIndex << ".exr";
        fs::path rgbfile ( rgbstr.str() );
        fs::path depthfile ( depthstr.str() );

        while ( ! ( fs::exists ( rgbfile ) && fs::exists ( depthfile ) ) ) {
            ++currBufferIndex;

            if ( currBufferIndex >= rgbs )
                throw std::runtime_error ( "Could not find starting index!" );

            rgbstr.str ( std::string() );
            depthstr.str ( std::string() );

            rgbstr << colorpath << "/Color" << std::setfill ( '0' )
                   << std::setw ( 4 ) << currBufferIndex << ".png";
            depthstr << depthpath << "/Depth" << std::setfill ( '0' )
                     << std::setw ( 4 ) << currBufferIndex << ".exr";
            rgbfile = fs::path ( rgbstr.str() );
            depthfile = fs::path ( depthstr.str() );
        }
        currFrame = currBufferIndex;

        return rgbs;
    } else {
        throw std::runtime_error ( "Could not read color or depth dir!" );
    }
}

RGBD ImageReader::readFrame ( int index ) {
    std::stringstream rgbs;
    std::stringstream depths;
    rgbs << colorpath << "/Color" << std::setfill ( '0' ) << std::setw ( 4 )
         << index << ".png";
    depths << depthpath << "/Depth" << std::setfill ( '0' ) << std::setw ( 4 )
           << index << ".exr";

    cv::Mat depth = cv::imread ( depths.str(), cv::IMREAD_UNCHANGED );
    if ( depth.type() != CV_32FC1 ) {
        throw std::invalid_argument ( "Unsupported depth-files: "
                                      + cv::typeToString ( depth.type() ) );
    }

    depth.setTo ( 0, depth > 100 );

    return RGBD ( cv::imread ( rgbs.str() ), depth );
}
