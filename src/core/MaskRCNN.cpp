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
#include "EMFusion/core/MaskRCNN.h"

namespace emf {

const std::string MaskRCNN::classNames[] = {
    "BG", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

MaskRCNN::MaskRCNN ( const std::vector<std::string>& FILTER_CLASSES,
                     const std::vector<std::string>& STATIC_OBJECTS ) {
    initialize ( FILTER_CLASSES, STATIC_OBJECTS );
}

MaskRCNN::~MaskRCNN() {
    Py_XDECREF ( pExec );
    Py_XDECREF ( pPreproc );
    Py_XDECREF ( pLoadPreproc );
    Py_DECREF ( pModule );
}

void* MaskRCNN::initialize ( const std::vector<std::string>& FILTER_CLASSES,
                             const std::vector<std::string>& STATIC_OBJECTS ) {
    size_t nameLength;
    wchar_t *progName = Py_DecodeLocale ( moduleName, &nameLength );
    Py_SetProgramName ( progName );
    Py_Initialize();

    std::string fileName ( moduleName );
    fileName += ".py";
    wchar_t const *argv2[] = { Py_DecodeLocale ( fileName.c_str(),
                               &nameLength )
                             };

    PySys_SetArgv ( 1, const_cast<wchar_t**> ( argv2 ) );

    pModule = PyImport_ImportModule ( moduleName );
    if ( pModule == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        throw std::runtime_error ( "Could not open module maskrcnn." );
    }

    PyObject *pFilterClasses = PyList_New ( FILTER_CLASSES.size() );
    for ( int i = 0; i < FILTER_CLASSES.size(); ++i ) {
        PyList_SetItem ( pFilterClasses, i,
                         PyUnicode_FromString ( FILTER_CLASSES[i].c_str() ) );
    }
    PyDict_SetItemString ( PyModule_GetDict ( pModule ), "FILTER_CLASSES",
                           pFilterClasses );
    PyObject *pStaticObjects = PyList_New ( STATIC_OBJECTS.size() );
    for ( int i = 0; i < STATIC_OBJECTS.size(); ++i ) {
        PyList_SetItem ( pStaticObjects, i,
                         PyUnicode_FromString ( STATIC_OBJECTS[i].c_str() ) );
    }
    PyDict_SetItemString ( PyModule_GetDict ( pModule ), "STATIC_OBJECTS",
                           pStaticObjects );

    import_array();

    pExec = PyObject_GetAttrString ( pModule, execName );
    if ( pExec == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        throw std::runtime_error ( "Could not find function execute." );
    }
    pPreproc = PyObject_GetAttrString ( pModule, preprocName );
    if ( pExec == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        throw std::runtime_error ( "Could not find function preprocess." );
    }
    pLoadPreproc = PyObject_GetAttrString ( pModule, loadPreprocName );
    if ( pExec == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        throw std::runtime_error ( "Could not find "
                                   "function load_preprocessed." );
    }

    return 0;
}

int MaskRCNN::getBoundingBoxes ( PyObject *pBoxes,
                                 std::vector<cv::Rect>& bounding_boxes ) {
    if ( !PySequence_Check ( pBoxes ) )
        throw std::runtime_error ( "Boxes should be a sequence!" );
    Py_ssize_t num_boxes = PySequence_Length ( pBoxes );
    bounding_boxes.reserve ( num_boxes );
    for ( int i = 0; i < num_boxes; ++i ) {
        PyObject *pBox = PySequence_GetItem ( pBoxes, i );
        assert ( PySequence_Check ( pBox ) );
        Py_ssize_t num_coords = PySequence_Length ( pBox );
        assert ( num_coords == 4 );

        int x[2], y[2];
        for ( int j = 0; j < 2; ++j ) {
            PyObject* px = PySequence_GetItem ( pBox, 2*j );
            PyObject* py = PySequence_GetItem ( pBox, 2*j + 1 );
            assert ( PyLong_Check ( px ) && PyLong_Check ( py ) );

            x[j] = PyLong_AsLong ( px );
            y[j] = PyLong_AsLong ( py );

            Py_DECREF ( px );
            Py_DECREF ( py );
        }

        bounding_boxes.push_back (
            cv::Rect ( y[0], x[0], y[1]-y[0], x[1]-x[0] ) );
        Py_DECREF ( pBox );
    }

    return num_boxes;
}

int MaskRCNN::getSegmentation ( PyObject *pSegmentation,
                                std::vector<cv::Mat>& segmentation ) {
    if ( !PySequence_Check ( pSegmentation ) )
        throw std::runtime_error ( "Segmentation should be a sequence!" );
    Py_ssize_t n = PySequence_Length ( pSegmentation );
    segmentation.resize ( n );

    for ( int i = 0; i < n; ++i ) {
        PyObject* pMask = PySequence_GetItem ( pSegmentation, i );
        PyArrayObject* pMaskArray = ( PyArrayObject* ) ( pMask );
        unsigned char* pData = ( unsigned char* ) PyArray_GETPTR1 ( pMaskArray,
                               0 );
        npy_intp h = PyArray_DIM ( pMaskArray, 0 );
        npy_intp w = PyArray_DIM ( pMaskArray, 1 );

        cv::Mat ( h, w, CV_8UC1, pData ).copyTo ( segmentation[i] );
        Py_DECREF ( pMask );
    }

    return n;
}

int MaskRCNN::getScores ( PyObject* pScores,
                          std::vector<std::vector<double> >& scores ) {
    if ( !PySequence_Check ( pScores ) )
        throw std::runtime_error ( "Scores should be a sequence!" );
    Py_ssize_t n = PySequence_Length ( pScores );
    scores.resize ( n );

    for ( int i = 0; i < n; ++i ) {
        PyObject *pScoresDetect = PySequence_GetItem ( pScores, i );
        assert ( PySequence_Check ( pScoresDetect ) );
        // TODO: Use actual number of classes dynamically?
        assert ( PySequence_Length ( pScoresDetect ) == 81 );

        scores[i].reserve ( 81 );
        for ( int j = 0; j < 81; ++j ) {
            PyObject *pScore = PySequence_GetItem ( pScoresDetect, j );
            assert ( PyFloat_Check ( pScore ) );

            double score = PyFloat_AsDouble ( pScore );
            scores[i].push_back ( score );
            Py_DECREF ( pScore );
        }
        Py_DECREF ( pScoresDetect );
    }

    return n;
}

int MaskRCNN::execute ( const cv::Mat& frame,
                        std::vector<cv::Rect>& bounding_boxes,
                        std::vector<cv::Mat>& segmentation,
                        std::vector<std::vector<double>>& scores ) {
    npy_intp dims[3] = { frame.rows, frame.cols, frame.channels() };
    PyObject *pInput = PyArray_SimpleNewFromData ( 3, dims, NPY_UINT8,
                       frame.data );

    PyObject *pValue = PyObject_CallFunctionObjArgs ( pExec, pInput, NULL );
    if ( pValue == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        fprintf ( stderr, "Maskrcnn function call returned NULL!" );
        return -1;
    }

    if ( !PyTuple_Check ( pValue ) || PyTuple_Size ( pValue ) != 3 )
        throw std::runtime_error ( "Maskrcnn function did not return a tuple"
                                   " or a tuple of the wrong size!" );

    PyObject *pBoxes = PyTuple_GetItem ( pValue, 0 );
    PyObject *pSegmentation = PyTuple_GetItem ( pValue, 1 );
    PyObject *pScores = PyTuple_GetItem ( pValue, 2 );

    int numInstances = getBoundingBoxes ( pBoxes, bounding_boxes );
    int numMasks = getSegmentation ( pSegmentation, segmentation );
    int numScoreInstances = getScores ( pScores, scores );
    assert ( numInstances == numScoreInstances && numInstances == numMasks );

    Py_XDECREF ( pValue );

    return numInstances;
}

void MaskRCNN::preprocess ( const cv::Mat& frame,
                            const std::string& filename ) {
    npy_intp dims[3] = { frame.rows, frame.cols, frame.channels() };
    PyObject *pImage = PyArray_SimpleNewFromData ( 3, dims, NPY_UINT8,
                       frame.data );
    PyObject *pFilename = PyUnicode_FromString ( filename.c_str() );

    PyObject_CallFunctionObjArgs ( pPreproc, pImage, pFilename, NULL );

    Py_DECREF ( pFilename );
    if ( PyErr_Occurred() )
        PyErr_Print();
}

int MaskRCNN::loadPreprocessed ( const std::string& filename,
                                 std::vector<cv::Rect>& bounding_boxes,
                                 std::vector<cv::Mat>& segmentation,
                                 std::vector<std::vector<double> >& scores ) {
    PyObject *pFilename = PyUnicode_FromString ( filename.c_str() );

    PyObject *pValue =
        PyObject_CallFunctionObjArgs ( pLoadPreproc, pFilename, NULL );
    if ( pValue == NULL ) {
        if ( PyErr_Occurred() )
            PyErr_Print();
        fprintf ( stderr, "Maskrcnn function call returned NULL!" );
        return -1;
    }

    if ( !PyTuple_Check ( pValue ) || PyTuple_Size ( pValue ) != 3 )
        throw std::runtime_error ( "Maskrcnn function did not return a tuple"
                                   " or a tuple of the wrong size!" );

    PyObject *pBoxes = PyTuple_GetItem ( pValue, 0 );
    PyObject *pSegmentation = PyTuple_GetItem ( pValue, 1 );
    PyObject *pScores = PyTuple_GetItem ( pValue, 2 );

    int numInstances = getBoundingBoxes ( pBoxes, bounding_boxes );
    int numMasks = getSegmentation ( pSegmentation, segmentation );
    int numScoreInstances = getScores ( pScores, scores );
    assert ( numInstances == numScoreInstances && numInstances == numMasks );

    Py_DECREF ( pFilename );
    Py_XDECREF ( pValue );

    return numInstances;
}

void MaskRCNN::visualize ( cv::Mat& vis, const cv::Mat& rgb,
                           const int numInstances,
                           const std::vector<cv::Rect>& bounding_boxes,
                           const std::vector<cv::Mat>& segmentation,
                           const std::vector<std::vector<double>>& scores ) {
    const unsigned char colors[31][3] = {
        {0, 0, 0},       {0, 0, 255},     {255, 0, 0},    {0, 255, 0},
        {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
        {167, 96, 61},   {79, 0, 105},    {0, 255, 246},  {61, 123, 140},
        {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
        {131, 131, 0},   {0, 255, 149},   {96, 0, 43},    {246, 131, 17},
        {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
        {0, 43, 96},     {158, 114, 140}, {79, 184, 17},  {158, 193, 255},
        {149, 158, 123}, {255, 123, 175}, {158, 8, 0}
    };
    auto getColor = [&colors] ( unsigned index ) -> cv::Vec3b {
        return ( index == 255 ) ? cv::Vec3b ( 255, 255, 255 )
        : ( cv::Vec3b ) colors[index % 31];
    };

    rgb.copyTo ( vis );

    for ( int i = 0; i < numInstances; ++i )
        vis.setTo ( getColor ( i+1 ), segmentation[i] );

    cv::addWeighted ( rgb, .5, vis, .5, 0.0, vis );

    for ( int i = 0; i < numInstances; ++i ) {
        rectangle ( vis, bounding_boxes[i], getColor ( i + 1 ) );
        int classId = distance ( scores[i].begin(),
                                 max_element ( scores[i].begin(),
                                               scores[i].end() ) );
        double score = scores[i][classId];
        std::stringstream textStream;
        textStream << classNames[classId] << ": "
                   << std::setprecision ( 2 ) << score;
        putText ( vis, textStream.str(), bounding_boxes[i].tl(),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar ( 0, 255, 255 ) );
    }
}

std::string MaskRCNN::getClassName ( int id ) {
    return classNames[id];
}

}
