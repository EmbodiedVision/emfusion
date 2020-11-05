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

#include <fstream>
#include <math.h>

#include <thread>
#include <condition_variable>
#include <mutex>

#include <boost/algorithm/string.hpp>

#include "EMFusion/utils/RGBDReader.h"

/**
 * Reader for TUM RGBD dataset. Assumes file associations.txt in the
 * dataset folder.
 */
class TUMRGBDReader : public RGBDReader {
public:
    /**
     * Create new reader object.
     *
     * @param path_ the path to the dataset folder containing associations.txt
     *              and rgb and depth folders
     */
    TUMRGBDReader ( std::string path_ );

    void init() override;

private:
    RGBD readFrame ( int index ) override;

    /**
     * Read association.txt file to get names of matching rgb and depth images.
     *
     * @param filename full path of associations.txt
     * @param rgbNames output vector of rgb filenames
     * @param depthNames output vector of depth filenames
     */
    void readFileAssociations ( const std::string& filename,
                                std::vector<std::string>& rgbNames,
                                std::vector<std::string>& depthNames );

    std::string associationFile;

    std::vector<std::string> rgbFileNames;
    std::vector<std::string> depthFileNames;
};
