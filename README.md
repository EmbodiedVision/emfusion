# EM-Fusion: Dynamic Object-Level SLAM With Probabilistic Data Association

![Overview Image](images/teaser.png)

This repository provides source code for EM-Fusion accompanying the following publication:

*Michael Strecke and Joerg Stuecker, "**EM-Fusion: Dynamic Object-Level SLAM With
Probabilistic Data Association**"*  
*Presented at the **IEEE/CVF International Conference on Computer Vision (ICCV)
2019**, Seoul, Korea*

Please see the [project page](https://emfusion.is.tue.mpg.de/) for details.

If you use the source code provided in this repository for your research, please cite the corresponding publication as:
```
@InProceedings{strecke2019_emfusion,
  author      = {Michael Strecke and Joerg Stueckler},
  booktitle   = {2019 {IEEE}/{CVF} International Conference on Computer Vision ({ICCV})},
  title       = {{EM-Fusion}: Dynamic Object-Level {SLAM} With Probabilistic Data Association},
  year        = {2019},
  month       = {oct},
  publisher   = {{IEEE}},
  doi         = {10.1109/iccv.2019.00596},
}
```

## Getting started

### 0. Install dependencies

Our code requires the following dependencies to be present on your system:

* [CUDA](https://developer.nvidia.com/cuda-zone) tested with version 10.0
* [CMake](https://cmake.org/)
* [boost](https://www.boost.org/) (modules system, filesystem, program_options
  with development headers)
* Python C API headers for calling python code from C++ and numpy as system-wide
  installation
* [Eigen](http://eigen.tuxfamily.org/)
* [VTK](https://vtk.org/) for enabling the build of the OpenCV VIZ module for 3D
  visualizations
* [OpenCV](https://opencv.org/) with CUDA support (needs building from sources
  with the OpenCV-contrib modules) tested with version 4.3.0
* [Sophus](https://github.com/strasdat/Sophus)
* [GTK 3 development package](https://www.gtk.org/) to make the OpenCV GUI 
  compatible with Mask R-CNN

For CUDA, follow the setup instructions at the link above for your platform.

On Ubuntu 18.04, CMake, boost, Python, Eigen, and VTK can be installed from the
default package repositories:
```bash
apt-get update && apt-get install cmake \
                libboost-system1.65-dev \
                libboost-filesystem1.65-dev \
                libboost-program-options1.65-dev \
                python3-dev python3-numpy virtualenv \
                libeigen3-dev \
                libvtk7-dev \
                libgtk-3-dev
```

OpenCV and Sophus need to be built from source. By using
`-DCMAKE_INSTALL_PREFIX=<path/to/install>` you can install these
libraries to a custom path. If you do so, set `CMAKE_PREFIX_PATH` accordingly
when running the CMake configuration for our code.

Recommended CMake call for building OpenCV (disables some unnecessary parts):
```bash
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib_path>/modules \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DWITH_CUDA=ON \
      -DCUDA_GENERATION=Auto \
      <opencv_path>
```

### 1. Set up Mask R-CNN

Clone the [Mask R-CNN repository](https://github.com/matterport/Mask_RCNN) to
some location (`<Mask_RCNN_DIR>` in the following) and create and set up the
virtual environment for Mask R-CNN in `<Mask_RCNN_DIR>/venv`:

```bash
cd <Mask_RCNN_DIR>
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate 
```

Afterwards, follow the instructions for setting up Mask R-CNN
[here](https://github.com/matterport/Mask_RCNN#installation). Make sure to
install a TensorFlow version <2.0. In order to achieve this, edit the
`requirements.txt` file and change the line `tensorflow>=1.3.0` to
`tensorflow>=1.3.0,<2.0`, as well as `keras>=2.0.8` to `keras>=2.0.8,<2.4.0`
(keras 2.4.0 dropped support for Tensorflow 1.x, see [here](https://github.com/keras-team/keras/releases)).
For GPU support in TensorFlow, you might want to point it to a specific version
compatible with you CUDA installation (see [here](https://www.tensorflow.org/install/gpu?hl=en)
for details). You also need to install `pycocotools` via `pip install pycocotools`.

Afterwards, you can leave the virtual environment with `deactivate` and return
to the base directory of this repository.

Pretrained weights for Mask R-CNN will be downloaded automatically upon the
first run of EM-Fusion or the preprocessing.

### 2. Build EM-Fusion

After installing all dependencies mentioned above and setting up Mask R-CNN,
download the EM-Fusion code to a separate directory. In that directory, create a build directory, configure the build using CMake, and build the
project:

```bash
mkdir build
cd build
cmake .. -DMASKRCNN_ROOT_DIR=<Mask_RCNN_DIR>
make -j$(nproc)
```

Depending on where you installed the dependencies, you might want to append 
`-DCMAKE_PREFIX_PATH=<path/to/install>` to the `cmake` call.

If you virtualenv for Mask R-CNN is not in `<Mask_RCNN_DIR>/venv`, append
`-DMASKRCNN_VENV_DIR=<path/to/venv>` to the `cmake` call above.

### 3. Running the code

Change to the `build` folder for running the code (since the `maskrcnn.py` file
is placed there this should be the working directory for running the code).

The main executable is called `EM-Fusion` and can be called with the following
options:
```
$ ./EM-Fusion -h
EM-Fusion: Dynamic tracking and Mapping from RGB-D data:

One of these options is required:
  -t [ --tumdir ] arg       Directory containing RGB-D data in the TUM format
  -d [ --dir ] arg          Directory containing color and depth images

Possibly needed when using "--dir" above:
  --colordir arg (=colour)  Subdirectory containing color images named 
                            Color*.png. Needed if different from "colour"
  --depthdir arg (=depth)   Subdirectory containing depth images named 
                            Depth*.exr. Needed if different from "depth"

Optional inputs:
  -h [ --help ]             Print this help
  -e [ --exportdir ] arg    Directory for storing results
  --export-frame-meshes     Whether to export meshes for every frame. Needs a 
                            lot of RAM if there are many objects.
  --export-volumes          Whether to output TSDF volumes with weights and 
                            foreground probabilities. Needs a lot of RAM if 
                            there are many objects.
  --background              Whether to run this program without live output 
                            (without -e it won'tbe possible to examine results)
  --3d-vis                  Whether to show 3D visualizations with meshes and 
                            bounding boxes
  -c [ --configfile ] arg   Path to a configuration file containing experiment 
                            parameters
  -m [ --maskdir ] arg      Directory containing preprocessed Mask R-CNN 
                            results
```

The `-t` and `-d` flags load data stored in the format of the
[TUM RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)
and the [Co-Fusion dataset](https://github.com/martinruenz/co-fusion),
respectively.

`--colordir` or `--depthdir` are needed when the color or depth subfolders of
files in the Co-Fusion format are not named `colour` and `depth`, respectively.
(E.g. for the synthetic sequences, `--depthdir depth_noise` was used in our
experiments.)

The `-e` flag can be used for storing results to a folder (will be overwritten 
if it exists). We provide options for optionally writing meshes for each frame
and the final TSDF volumes to disk. Since these are buffered in RAM until the
program ends, this might take a lot of memory.

`--background` allows running the program without direct visual output (no X
server) required. You should save results with `-e` to examine them.

`--3d-vis` enables 3D visualization. This creates a window that displays meshes
generated by marching cubes and the bounding boxes of objects. This slows down
the processing since marching cubes is run on every object in every frame.
This flag cannot be combined with `--background` and will be ignored if you try
to do so anyways.

The `-c` flag allows giving experiment-specific parameters in a config file.
Check [data.h](include/EMFusion/core/data.h) for
documentation of the parameters and [config/](config/) for example config files.
[config/default.cfg](config/default.cfg) specifies default parameters that are
also set automatically in [data.h](include/EMFusion/core/data.h) and can be used
as a reference guide on how to specify parameters.

The `-m` flag allows loading preprocessed masks. (See below.)

#### 3.1 Preprocessing with Mask R-CNN (optional)

By default, the `EM-Fusion` executable will attempt to run Mask R-CNN
sequentially as was done in the paper experiments. Since Tensorflow takes around
4-5GB of GPU memory for this, you might run into issues with available GPU
memory. To avoid this, we provide the program `preprocess_masks`, which lets you
preprocess and save the Mask R-CNN results in a format readable for `EM-Fusion`.

```
$ ./preprocess_masks -h
Preprocess RGB-D datasets with Mask R-CNN:

One input option and the output option is required:
  -t [ --tumdir ] arg       Directory containing RGB-D data in the TUM format
  -d [ --dir ] arg          Directory containing color and depth images
  -m [ --maskdir ] arg      Where to store the generated masks

Possibly needed when using "--dir" above:
  --colordir arg (=colour)  Subdirectory containing color images named 
                            Color*.png. Needed if different from "colour"
  --depthdir arg (=depth)   Subdirectory containing depth images named 
                            Depth*.exr. Needed if different from "depth"

Optional inputs:
  -h [ --help ]             Print this help
  -c [ --configfile ] arg   Path to a configuration file containing experiment 
                            parameters
```

Most parameters are the same as for `EM-Fusion`. The `-m` flag is now the output
folder and required for running this program. The only parameters from the
configuration file that are used in this program are the Mask R-CNN frequency
and the options `FILTER_CLASSES` and `STATIC_OBJECTS` that allow to only process
some classes or disregard some classes as static.

#### 3.2 Reducing volume resolution (optional)

If you still run out of GPU memory after following the previous section, the
main source for savings in GPU memory consumption is the background TSDF volume
resolution. Try reducing it in the config file (remember to adjust the voxel
size accordingly so the overall volume dimension stays the same). This change
might influence the results in terms of level of detail for the background
volume and tracking accuracies.

## Reproducing paper results
For reproducing the results from the paper, you can run the code on sequences 
from the [Co-Fusion dataset](https://github.com/martinruenz/co-fusion) and the
[TUM RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset).
The following configurations were used in the paper's experiments:
* For `car4-full`, we used the default configuration
* For `room4-full`, the volume size and starting pose needed to be adjusted as
  given in [room4.cfg](config/room4.cfg)
* For the real-world scenes from Co-Fusion, the pattern on a wall caused
  spurious `'umbrella'` detections that we excluded as static objects in
  [co-fusion-real.cfg](config/co-fusion-real.cfg)
* For the robust background tracking experiment on the  TUM RGB-D benchmark, we
  only detect `'person'` objects and disable their visualization in the rendered
  output as set up in [tum.cfg](config/tum.cfg)

A more detailed guide on how to run EM-Fusion can be found [here](EXAMPLE.md).

We provide scripts to automatically reproduce paper results consisting of the following parts:
### Run EM-Fusion
[run_exps.sh](run_exps.sh) runs EM-Fusion on the Co-Fusion and TUM RGB-D datasets.

For this to work, you will need to download the synthetic archives `car4-full.tar.gz`
and `room4-full.tar.gz` from [this site](https://github.com/martinruenz/co-fusion#synthetic-sequences).
Extract both archives to a folder (`CO-FUSION_FOLDER` in the following) in
subdirectories "car4-full" and "room4-full", respectively.

Furthermore, download the TUM RGB-D dataset from [here](https://vision.in.tum.de/data/datasets/rgbd-dataset)
(at least the `fr3/sitting*` and `fr3/walking*` scenes) and place it in a folder
(`TUM_FOLDER` in the following). The `fr3/{sitting,walking}*` scenes should be
placed in `TUM_FOLDER/freiburg3/`. Also make sure to run [the association script](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools)
for each scene like this:
```bash
python associate.py rgb.txt depth.txt > associations.txt
```

You can then execute the following to run EM-Fusion and write the results to an
`OUTPUT_FOLDER`:
```bash
./run_exps.sh $CO-FUSION_FOLDER $TUM_FOLDER $OUTPUT_FOLDER
```

### Evaluate results on Co-Fusion
For evaluating performance on the Co-Fusion datasets, you will need the program
`convert_poses` from the [dataset-tools repository](https://github.com/martinruenz/dataset-tools)
and the `evaluate_ate.py` and `evaluate_rpe.py` script from [here](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/).
Make sure to set the correct locations for these in [eval_co-fusion.sh](eval_co-fusion.sh).

Then, after running EM-Fusion with the above script, you can run
```bash
./eval_co-fusion.sh $OUTPUT_FOLDER $CO-FUSION_FOLDER
```
to evaluate EM-Fusion's performance.

### Evaluate on TUM RGB-D dataset
Adapt the path for the evaluation scripts from above in [eval_tum.sh](eval_tum.sh).

Then, you can run
```bash
./eval_tum.sh $OUTPUT_FOLDER $TUM_FOLDER
```
to evaluate EM-Fusion for robust static background tracking.

### Output folder structure
```
|-co-fusion
|  |-car4
|  |   |-multiple folders: containing weights/etc used for figures in the paper
|  |   |-mesh*.ply: final mesh models for all objects
|  |   |-poses*.txt: files for the internal object poses
|  |   |-poses*-corrected.txt: files containing poses corrected for possbly resized objects
|  |   |-poses*-mapped-origin.txt: aligned poses from convert_poses
|  |   |-{ate,rpe}*.txt: files containing the numerical errors for the objects
|  |   `-{ate,rpe}*.pdf: visualizations of the errors
|  `-room4 with the same structure as car4
`-tum
   `-f3_* folders for each dataset
        |-folders as above
        |-poses-files as above
        |-poses-cam-ts.txt with frame numbers replaced by timestamps
        `-ate and rpe results and visualizations for the camera trajectory
```

## License
EM-Fusion has been developed at the [Embodied Vision Group](https://ev.is.mpg.de) at the Max Planck Institute for Intelligent Systems, Germany. The open-source version is licensed under the [GNU General Public License v3 (GPLv3)](LICENSE).

For commercial inquiries, please send email to [ev-license@tue.mpg.de](mailto:ev-license@tue.mpg.de).
