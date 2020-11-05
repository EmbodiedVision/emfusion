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

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <opencv2/opencv.hpp>

namespace emf {

/**
 * Class for loading and executing Python interface to Mask R-CNN.
 */
class MaskRCNN {
public:
    /**
     * Constructor
     */
    MaskRCNN ( const std::vector<std::string>& FILTER_CLASSES,
               const std::vector<std::string>& STATIC_OBJECTS );

    /**
     * Destructor
     */
    ~MaskRCNN();

    /**
     * Run Mask R-CNN on current frame.
     *
     * @param frame input RGB frame
     * @param bounding_boxes output bounding boxes
     * @param segmentation output segmentation
     * @param scores output class scores
     *
     * @return number of instances
     */
    int execute ( const cv::Mat& frame, std::vector<cv::Rect>& bounding_boxes,
                  std::vector<cv::Mat>& segmentation,
                  std::vector<std::vector<double>>& scores );

    /**
     * Run Mask R-CNN on current frame and save to given file.
     *
     * @param frame input RGB frame
     * @param filename output filename
     */
    void preprocess ( const cv::Mat& frame, const std::string& filename );

    /**
     * Load preprocessed mask data from file.
     *
     * @param filename filename to load data from.
     */
    int loadPreprocessed ( const std::string& filename,
                           std::vector<cv::Rect>& bounding_boxes,
                           std::vector<cv::Mat>& segmentation,
                           std::vector<std::vector<double>>& scores );

    /**
     * Visualize Mask R-CNN segmentation.
     *
     * @param vis output visualization
     * @param rgb input rgb image
     * @param numInstances number of mask instances
     * @param bounding_boxes bounding boxes for instances
     * @param segmentation masks for instances
     * @param scores class scores for instances
     */
    static void visualize ( cv::Mat& vis, const cv::Mat& rgb, int numInstances,
                            const std::vector<cv::Rect>& bounding_boxes,
                            const std::vector<cv::Mat>& segmentation,
                            const std::vector<std::vector<double>>& scores );

    /**
     * Get class name for class ID.
     *
     * @param id class id.
     *
     * @return class name.
     */
    static std::string getClassName ( int id );

private:
    /**
     * Load Mask R-CNN Python module.
     */
    void* initialize ( const std::vector<std::string>& FILTER_CLASSES,
                       const std::vector<std::string>& STATIC_OBJECTS );

    /**
     * Get bounding boxes from last Mask R-CNN run.
     *
     * @param pBoxes python object containing the bounding boxes
     * @param bounding_boxes OpenCV output vector
     *
     * @return number of instances.
     */
    int getBoundingBoxes ( PyObject* pBoxes,
                           std::vector<cv::Rect>& bounding_boxes );

    /**
     * Get masks from last Mask R-CNN run.
     *
     * @param pBoxes python object containing the masks
     * @param bounding_boxes OpenCV output vector
     *
     * @return number of instances.
     */
    int getSegmentation ( PyObject* pSegmentation,
                          std::vector<cv::Mat>& segmentation );

    /**
     * Get class scores from last Mask R-CNN run.
     *
     * @param pBoxes python object containing the class scores
     * @param bounding_boxes OpenCV output vector
     *
     * @return number of instances.
     */
    int getScores ( PyObject* pScores,
                    std::vector<std::vector<double>>& scores );

    static constexpr char const *moduleName = "maskrcnn";
    static constexpr char const *execName = "execute";
    static constexpr char const *preprocName = "preprocess";
    static constexpr char const *loadPreprocName = "load_preprocessed";
    PyObject *pModule, *pExec, *pPreproc, *pLoadPreproc;

    static const std::string classNames[];
};

}
