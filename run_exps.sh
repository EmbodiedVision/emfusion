#!/bin/bash
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

# Usage: ./run_exps.sh <path/to/cofusion> <path/to/tum> <output/path>

cd build
./EM-Fusion -d $1/car4-full  --depthdir depth_noise -c ../config/default.cfg --background -e $3/co-fusion/car4
./EM-Fusion -d $1/room4-full --depthdir depth_noise -c ../config/room4.cfg   --background -e $3/co-fusion/room4

for ds in {sitting,walking}_{static,xyz,halfsphere}; do
	./EM-Fusion -t $2/freiburg3/$ds -c ../config/tum.cfg --background -e $3/tum/f3_$ds
done

