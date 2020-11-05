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

# Usage: ./eval_co-fusion.sh <path/to/results> <path/to/co-fusion>

CONVERT_POSES=/is/sg/mstrecke/code/dataset-tools/Release/bin/convert_poses
TUM_EVAL_SCRIPTS=/is/sg/mstrecke/code/rgbd_benchmark_tools/scripts

cd "$1/co-fusion"

cd car4
declare -A gt_obj
declare -A first_frame

gt_obj[2]="gt-truck-1"
first_frame[2]=90
gt_obj[4]="gt-car-2"
first_frame[4]=270

for obj in {2,4}; do
	$CONVERT_POSES --frame ${first_frame[$obj]} --object poses-$obj-corrected.txt --camera poses-cam.txt --gtobject "$2/car4-full/trajectories/${gt_obj[$obj]}.txt" --gtcamera "$2/car4-full/trajectories/gt-cam-0.txt" --out poses-$obj-mapped-origin.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_ate.py" --verbose --plot ate_$obj.pdf "$2/car4-full/trajectories/${gt_obj[$obj]}.txt" poses-$obj-mapped-origin.txt > ate_$obj.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_rpe.py" --verbose --fixed_delta --plot rpe_$obj.pdf "$2/car4-full/trajectories/${gt_obj[$obj]}.txt" poses-$obj-mapped-origin.txt > rpe_$obj.txt
done

python "$TUM_EVAL_SCRIPTS/evaluate_ate.py" --verbose --plot ate_cam.pdf "$2/car4-full/trajectories/gt-cam-0.txt" poses-cam.txt > ate_cam.txt
python "$TUM_EVAL_SCRIPTS/evaluate_rpe.py" --verbose --fixed_delta --plot rpe_cam.pdf "$2/car4-full/trajectories/gt-cam-0.txt" poses-cam.txt > rpe_cam.txt
cd ..

cd room4
declare -A gt_obj
declare -A first_frame

gt_obj[1]="gt-ship-1"
first_frame[1]=540
gt_obj[2]="gt-ship-1"
first_frame[2]=600
gt_obj[3]="gt-ship-1"
first_frame[3]=600
gt_obj[4]="gt-poses-horse-3"
first_frame[4]=690
gt_obj[5]="gt-car-2"
first_frame[5]=780

for obj in {1..5}; do
	$CONVERT_POSES --frame ${first_frame[$obj]} --object poses-$obj-corrected.txt --camera poses-cam.txt --gtobject "$2/room4-full/trajectories/${gt_obj[$obj]}.txt" --gtcamera "$2/room4-full/trajectories/gt-cam-0.txt" --out poses-$obj-mapped-origin.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_ate.py" --verbose --plot ate_$obj.pdf "$2/room4-full/trajectories/${gt_obj[$obj]}.txt" poses-$obj-mapped-origin.txt > ate_$obj.txt
	python "$TUM_EVAL_SCRIPTS/evaluate_rpe.py" --verbose --fixed_delta --plot rpe_$obj.pdf "$2/room4-full/trajectories/${gt_obj[$obj]}.txt" poses-$obj-mapped-origin.txt > rpe_$obj.txt
done

python "$TUM_EVAL_SCRIPTS/evaluate_ate.py" --verbose --plot ate_cam.pdf "$2/room4-full/trajectories/gt-cam-0.txt" poses-cam.txt > ate_cam.txt
python "$TUM_EVAL_SCRIPTS/evaluate_rpe.py" --verbose --fixed_delta --plot rpe_cam.pdf "$2/room4-full/trajectories/gt-cam-0.txt" poses-cam.txt > rpe_cam.txt
cd ..

