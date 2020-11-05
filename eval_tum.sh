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

# Usage: ./eval_tum.sh <path/to/results> <path/to/tum-rgbd-dataset>

TUM_EVAL_SCRIPTS=/is/sg/mstrecke/code/rgbd_benchmark_tools/scripts

cd $1/tum

for ds in {sitting,walking}_{static,xyz,halfsphere}; do
	cd f3_$ds
	awk 'FNR==NR{a[NR]=$3;next}{$1=a[FNR]}1' "$2/freiburg3/$ds/associations.txt" poses-cam.txt > poses-cam-ts.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_ate.py" --plot ate_cam.pdf --verbose "$2/freiburg3/$ds/groundtruth.txt" poses-cam-ts.txt > ate.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_rpe.py" --plot rpe_cam.pdf --fixed_delta --verbose "$2/freiburg3/$ds/groundtruth.txt" poses-cam-ts.txt > rpe.txt
	cd ..
done
