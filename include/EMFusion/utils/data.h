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

/**
 * Class for storing RGBD frames.
 */
class RGBD {
public:
    /**
     * Default constructor for empty frame.
     */
    RGBD() {}

    /**
     * Constructor for new RGBD frame.
     *
     * @param _rgb RGB data to be stored
     * @param _depth depth data to be stored
     */
    RGBD ( cv::Mat _rgb, cv::Mat _depth ) : rgb ( _rgb ), depth ( _depth ) {
        assert ( _rgb.size == _depth.size );
    }

    /**
     * Get const reference to stored RGB frame.
     */
    const cv::Mat& getRGB() const {
        return rgb;
    }

    /**
     * Get const reference to stored depth frame.
     */
    const cv::Mat& getDepth() const {
        return depth;
    }

    /**
     * Get frame size.
     */
    cv::Size getSize() const {
        return rgb.size();
    }

private:
    cv::Mat rgb;
    cv::Mat depth;
};
