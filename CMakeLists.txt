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
project(EM-Fusion)

include_directories("${PROJECT_SOURCE_DIR}/include/")

if (NOT MASKRCNN_ROOT_DIR)
    set(MASKRCNN_ROOT_DIR "${PROJECT_SOURCE_DIR}/external/Mask_RCNN")
endif()
if (NOT MASKRCNN_VENV_DIR)
    set(MASKRCNN_VENV_DIR "${MASKRCNN_ROOT_DIR}/venv")
endif()

add_subdirectory( src )
add_subdirectory( apps )
