# Example for running and evaluating EM-Fusion

## 1. Download dynamic scene datasets

Go to the [Co-Fusion repository](https://github.com/martinruenz/co-fusion#synthetic-sequences)
and download one or more of the sequences. For the synthetic sequences, we used
the *.tar.gz archives.

The real-world scenes are provided in the *.klg format, so you need to convert
them first. A tool for doing this can be found [here](https://github.com/martinruenz/dataset-tools/tree/master/convert_klg).
After building this tool with CMake (together with the rest of the repository),
we used
```bash
convert_klg -i <path/to/klg> -o <output/path> -frames -sub -png
```
to generate folders for the real-world scenes. Please check
`build/bin/convert_klg -h` for what the options do. We additionally renamed the
`color` subfolder to `colour` to match the synthetic sequences.

## 2. Run EM-Fusion

### 2.0 (Optional) Preprocess masks
In order to preprocess masks, run
```bash
preprocess_masks -d <dataset/path> [--depthdir depth_noise] [-c <path/to/config>] -m <output/mask/path>
```
The program output will contain a lot of "Buffering failure." outputs. This is
normal since we do not process every frame and just skip over non-mask frames.
Thus, the buffering thread cannot always keep up pre-loading the next frame.

The `--depthdir` argument is only needed for the synthetic scenes since the
depth subfolder has this name.

The `-c` option is needed e.g. for the robust tracking experiment with the
TUM-RGBD-scenes where Mask R-CNN is instructed to only detect persons.

### 2.1 Run EM-Fusion

Go to the `build` folder of EM-Fusion and run the EM-Fusion executable.

This is an example running EM-Fusion with preprocessed masks (the car4-full
dataset extracted to `~/co-fusion-datasets/car4-full/` and preprocessed masks in
a subfolder called `preproc_masks`). The output will contain more of the
TensorFlow output seen in `preprocess_masks` if Mask R-CNN is run on-the-fly
from EM-Fusion.

```bash
./EM-Fusion -d ~/co-fusion-datasets/car4-full/ --depthdir depth_noise \
            -c ../config/default.cfg \
            -m ~/co-fusion-datasets/car4-full/preproc_masks/ \
            --3d-vis
Reading from /home/streckus/co-fusion-datasets/car4-full//
Buffer thread started with id: 139871750289152
Created new Object with ID: 1
Created new Object with ID: 2
Created new Object with ID: 3
Created new Object with ID: 4
Deleting Object 4 because it is not visible!
Finished processing, press any key to end the program!
Program ended successfully!
```

While the program is running, you will see the following windows:

| Main window                  | 3D visualization         |
| ---------------------------- | ------------------------ |
| ![Output](images/output.png) | ![3D](images/3d-vis.png) |
| This is the main window. It will be visible if not disabled with `--background`. When this window is active, you can pause the processing by pressing "P" or cancel the program with "Q". When the commandline output says "press any key to end the program", this window should be active for the program to register the key. | This is the 3D visualization window showing object and background meshes with their bounding boxes. It will only be visible when the `--3d-vis` flag is given. |

# 3. Evaluate EM-Fusion

If you export the results of EM-Fusion with `-e`, you can evaluate the poses
numerically. For dynamic scene datasets, EM-Fusion has to "guess" the object
centers, thus the object coordinate systems might not be well aligned with the
ground truth. The authors of Co-Fusion provide a program to convert these poses:
[convert_poses](https://github.com/martinruenz/dataset-tools/tree/master/convert_poses).
The second bullet point (**How to compare an object-trajectory of your non-static SLAM-method with ground-truth data?**)
[here](https://github.com/martinruenz/dataset-tools#howtos) explains how to use
it.

After converting the poses by using the first frame in the object trajectory as
a reference, we evaluated numerical accuracy by scripts from the
[TUM RGB-D benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset),
namely [evaluate_ate.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py)
and [evaluate_rpe.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_rpe.py).
