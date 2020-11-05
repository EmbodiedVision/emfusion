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

#include <condition_variable>
#include <thread>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "EMFusion/utils/data.h"

/**
 * Base class for reading RGBD datasets.
 */
class RGBDReader {
public:
    /**
     * Destroy object and stop buffered reader thread.
     */
    virtual ~RGBDReader ();

    /**
     * Initialize the reader (check for number of input files, start buffered
     * reading thread.
     */
    virtual void init() = 0;

    /**
     * Get next RGBD frame.
     *
     * @return the next RGBD frame
     */
    virtual RGBD getNextFrame();

    /**
     * Get total number of frames in dataset.
     *
     * @return the number of frames in the dataset
     */
    int getNumFrames();

    /**
     * Check if there are more frames in the dataset.
     *
     * @return boolean indicating whether more frames can be read.
     */
    bool moreFrames();

    /**
     * Get average framerate of current dataset.
     *
     * @return the framerate of the current dataset.
     */
    double getFrameRate();

protected:
    /**
     * Constructor initializing the current reader with a path and setting the
     * current frame number to -1.
     *
     * @param path_ the path to the dataset
     */
    RGBDReader ( std::string path_ );

    /**
     * Start the buffered reader thread.
     */
    void startBufferedRead();

    /**
     * Stop the buffered reader thread.
     */
    void stopBufferedRead();

    /**
     * Loop reading frames from disk in separate thread.
     */
    void readerLoop();

    /**
     * Implementation for reading files (including specific patterns for this
     * type of dataset.
     */
    void readerImpl();

    /**
     * Read a single RGBD frame for an index from disk.
     *
     * @param index the frame index to be read.
     * @return the read frame
     */
    virtual RGBD readFrame ( int index ) = 0;

    int32_t numFrames;
    double frameRate = 30;
    int32_t currFrame;
    std::string path;
    std::queue<RGBD> frames;

    std::thread readerThread;
    std::mutex readerMutex, bufferMutex;
    std::condition_variable readerCondition, bufferCondition;
    bool readerActive;
    int32_t minBufferSize;
    int32_t currBufferIndex = 0;
};
