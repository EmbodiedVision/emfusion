##
## This file is part of EM-Fusion.
##
## Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
## Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
## For more information see <https://emfusion.is.tue.mpg.de>.
## If you use this code, please cite the respective publication as
## listed on the website.
##
## EM-Fusion is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## EM-Fusion is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with EM-Fusion.  If not, see <https://www.gnu.org/licenses/>.
##

cmake_minimum_required(VERSION 3.0)

# OpenCV
find_package(OpenCV REQUIRED)

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
link_directories(${Boost_LIBRARY_DIRS})

# Threads
find_package(Threads)

add_library(rgbdreaders SHARED
    utils/RGBDReader.cpp
    utils/TUMRGBDReader.cpp
    utils/ImageReader.cpp
)

target_include_directories(rgbdreaders PUBLIC
    "${Boost_INCLUDE_DIRS}"
)

target_link_libraries(rgbdreaders
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)


find_package(PythonLibs)
IF(PYTHON_INCLUDE_DIRS)
message("Found Python: include: ${PYTHON_INCLUDE_DIRS},"
        " libs: ${PYTHON_LIBRARIES}")

add_library(maskrcnn SHARED core/MaskRCNN.cpp)

target_include_directories(maskrcnn PUBLIC
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(maskrcnn
    ${PYTHON_LIBRARIES}
    ${OpenCV_LIBS}
)
ENDIF()

## CUDA
find_package(CUDA)

find_package(Sophus)
find_package(Eigen3)

IF(CUDA_FOUND AND Eigen3_DIR)
cuda_add_library(emfusion SHARED
    core/EMFusion.cpp
    core/cuda/EMFusion.cu
    core/TSDF.cpp
    core/cuda/TSDF.cu
    core/ObjTSDF.cpp
    core/cuda/ObjTSDF.cu
)

target_include_directories(emfusion PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "${EIGEN3_INCLUDE_DIR}"
    "${CMAKE_PREFIX_PATH}/include"
)

target_link_libraries(emfusion
    ${PYTHON_LIBRARIES}
    ${OpenCV_LIBS}
    maskrcnn
)
ENDIF()
