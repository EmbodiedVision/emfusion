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
#include "EMFusion/utils/RGBDReader.h"

RGBDReader::RGBDReader ( std::string path_ ) : path ( path_ ) {
    currFrame = -1;
}

RGBDReader::~RGBDReader() {
    stopBufferedRead();
}

RGBD RGBDReader::getNextFrame() {
    if ( readerActive ) {
        readerCondition.notify_one();
    }
    ++currFrame;

    if ( currFrame < 0 ) return RGBD();

    assert ( readerActive || !frames.empty() );

    bool bufferFail = false;
    // The buffering thread might not be fast enough...
    std::unique_lock<std::mutex> lock ( bufferMutex );
    while ( frames.empty() ) {
        bufferCondition.wait ( lock );
        bufferFail = true;
    }
    // Show info for this case...
    if ( bufferFail ) {
        std::cout << "INFO: Buffering slower than processing." << std::endl;
    }

    RGBD frame = frames.front();
    frames.pop();
    return frame;
}

int RGBDReader::getNumFrames() {
    return numFrames;
}

bool RGBDReader::moreFrames() {
    return currFrame + 1 < numFrames;
}

double RGBDReader::getFrameRate() {
    return frameRate;
}

void RGBDReader::startBufferedRead() {
    // Start a new thread. We will use the readerActive variable to tell it when
    // to stop.
    readerActive = true;
    readerThread = std::thread ( &RGBDReader::readerLoop, this );
}

void RGBDReader::stopBufferedRead() {
    while ( readerActive || !readerThread.joinable() ) {
        // Tell the thread to stop, notify it in case it's waiting because the
        // buffer is large enough.
        readerActive = false;
        readerCondition.notify_one();
    }
    readerThread.join();
}

void RGBDReader::readerLoop() {
    std::cout << "File reader thread started with id: "
              << std::this_thread::get_id() << std::endl;

    std::unique_lock<std::mutex> lock ( readerMutex );
    while ( readerActive ) {
        readerImpl();
        if ( readerActive ) {
            // Wait for main thread's notification. Continue only if reader
            // should shut down or buffer gets too small.
            readerCondition.wait ( lock, [this]() {
                return !readerActive || frames.size() < minBufferSize;
            } );
        }
    }
}

void RGBDReader::readerImpl() {
    // Load in chunks of at least half the minimum buffer size
    for ( unsigned i = 0; i < minBufferSize / 2 && currBufferIndex < numFrames;
            ++i, ++currBufferIndex ) {
        frames.push ( readFrame ( currBufferIndex ) );
    }
    // Notify main thread that might be waiting for frames
    bufferCondition.notify_one();
    // If all frames were read, shut down reader thread.
    if ( currBufferIndex == numFrames )
        readerActive = false;
}
