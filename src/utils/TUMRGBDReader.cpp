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
#include "EMFusion/utils/TUMRGBDReader.h"

TUMRGBDReader::TUMRGBDReader ( std::string path_ ) : RGBDReader ( path_ ),
    associationFile ( path_ + "associations.txt" ) {
}

void TUMRGBDReader::init() {
    std::cout << "Reading from " << path << std::endl;
    readFileAssociations ( associationFile, rgbFileNames, depthFileNames );

    numFrames = rgbFileNames.size();

    startBufferedRead();
}

void TUMRGBDReader::readFileAssociations ( const std::string& filename,
        std::vector<std::string>& rgbNames,
        std::vector<std::string>& depthNames ) {
    // parse associations.txt
    std::ifstream assocFile;
    assocFile.open ( filename, std::ios::in );

    if ( assocFile.fail() )
        throw std::runtime_error ( "Could not open association file!" );

    double startTime = 0, endTime = 0;
    std::string lineStr;
    std::vector<std::string> entryStrs;
    bool rgbFirst = true;

    if ( !assocFile.eof() ) {
        // read in line
        getline ( assocFile, lineStr );

        // split line at blanks
        boost::split ( entryStrs, lineStr, boost::is_any_of ( "\t " ) );
        if ( entryStrs.size() == 4 ) {
            if ( boost::starts_with ( entryStrs[1], "rgb/" ) ) {
                rgbNames.push_back ( entryStrs[1] );
                depthNames.push_back ( entryStrs[3] );
                startTime = stod ( entryStrs[0] );
            } else {
                rgbFirst = false;
                rgbNames.push_back ( entryStrs[3] );
                depthNames.push_back ( entryStrs[1] );
                startTime = stod ( entryStrs[0] );
            }
        }
    }

    while ( !assocFile.eof() ) {
        // read in line
        getline ( assocFile, lineStr );

        // split line at blanks
        boost::split ( entryStrs, lineStr, boost::is_any_of ( "\t " ) );
        if ( entryStrs.size() == 4 ) {
            if ( rgbFirst ) {
                rgbNames.push_back ( entryStrs[1] );
                depthNames.push_back ( entryStrs[3] );
            } else {
                rgbFirst = false;
                rgbNames.push_back ( entryStrs[3] );
                depthNames.push_back ( entryStrs[1] );
            }
            endTime = stod ( entryStrs[0] );
        }
    }
    frameRate = rgbNames.size() / ( endTime - startTime );
    minBufferSize = std::round ( frameRate );
}

RGBD TUMRGBDReader::readFrame ( int index ) {
    cv::Mat rgb = cv::imread ( path + rgbFileNames[index] );
    cv::Mat depth_tmp = cv::imread ( path + depthFileNames[index],
                                     cv::IMREAD_UNCHANGED );
    cv::Mat depth;

    // Scale depth to metric scale (factor 5000 from TUM website)
    depth_tmp.convertTo ( depth, CV_32FC1, 1/5000.f );
    return RGBD ( rgb, depth );
}
