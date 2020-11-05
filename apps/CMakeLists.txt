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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Boost
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
link_directories(${Boost_LIBRARY_DIRS})

find_package(CUDA)
## Eigen3
find_package(Eigen3)
IF (CUDA_FOUND AND Eigen3_DIR)
add_executable(EM-Fusion
    EM-Fusion.cpp
)

target_include_directories(EM-Fusion PUBLIC
    "${CUDA_INCLUDE_DIRS}"
    "${CMAKE_PREFIX_PATH}/include"
    "${Boost_INCLUDE_DIRS}"
    "${EIGEN3_INCLUDE_DIR}"
)

target_link_libraries(EM-Fusion
    rgbdreaders
    emfusion
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
configure_file("maskrcnn.in.py" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/maskrcnn.py")
ENDIF ()

find_package(PythonLibs)
IF(PYTHON_INCLUDE_DIRS)
add_executable(preprocess_masks
    preprocess_masks.cpp
)

target_include_directories(preprocess_masks PUBLIC
    "${CMAKE_PREFIX_PATH}/include"
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(preprocess_masks
    rgbdreaders
    maskrcnn
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
ENDIF()
