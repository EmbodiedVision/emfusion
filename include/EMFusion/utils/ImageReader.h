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

#include <boost/filesystem.hpp>

#include "EMFusion/utils/RGBDReader.h"

namespace fs = boost::filesystem;

/**
 * Reader for image files (e.g. Co-Fusion dataset)
 */
class ImageReader : public RGBDReader {
public:
    /**
     * Create a new reader using a path for the base folder and relative paths
     * to subdirectories containing indexed color and depth files.
     *
     * @param basepath_ path to the base directory
     * @param colordir_ path to the color directory (subdirectory of basepath_)
     * @param depthdir_ path to the depth directory (subdirectory of basepath_)
     */
    ImageReader ( std::string basepath_, std::string colordir_,
                  std::string depthdir_ );

    void init() override;

private:
    RGBD readFrame ( int index ) override;

    /**
     * Count the number of files in the dataset (and check if the number of
     * depth and color frames is the same).
     *
     * @return the number of files in the dataset.
     */
    int countFiles();

    std::string colorpath;
    std::string depthpath;
};
