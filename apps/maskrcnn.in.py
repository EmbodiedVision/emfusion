#
# This file is part of EM-Fusion.
#
# Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
# Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
# For more information see <https://emfusion.is.tue.mpg.de>.
# If you use this code, please cite the respective publication as
# listed on the website.
#
# EM-Fusion is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EM-Fusion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EM-Fusion.  If not, see <https://www.gnu.org/licenses/>.
#

import os
# Mask_RCNN root directory (reative to current file)
ROOT_DIR = os.path.abspath('@MASKRCNN_ROOT_DIR@')
ve_path = os.path.join('@MASKRCNN_VENV_DIR@', 'bin', 'activate_this.py')
exec(open(ve_path).read(), {'__file__': ve_path});

import sys
import numpy as np
import pickle


FILTER_CLASSES = []
STATIC_OBJECTS = []

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

config = None
model = None
utils = None


def init():
    global config
    global model
    global utils
    global FILTER_CLASSES
    global STATIC_OBJECTS

    FILTER_CLASSES = [class_names.index(x) for x in FILTER_CLASSES]
    STATIC_OBJECTS = [class_names.index(x) for x in STATIC_OBJECTS]

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    # Import Mask_RCNN
    sys.path.append(ROOT_DIR)
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from samples.coco import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to pre-trained coco model
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch sizSe to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Since we are running tracking code on the GPU as well, we don't
    # want tensorflow to use the entire GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True)


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def clip_boxes(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = np.split(window, 4)
    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    # Clip
    y1 = np.maximum(np.minimum(y1, wy2), wy1)
    x1 = np.maximum(np.minimum(x1, wx2), wx1)
    y2 = np.maximum(np.minimum(y2, wy2), wy1)
    x2 = np.maximum(np.minimum(x2, wx2), wx1)
    clipped = np.concatenate([y1, x1, y2, x2], axis=1)
    return clipped


def refine_proposals(proposals, class_ids, deltas, window):
    # Class-specific bounding box shifts.
    deltas_specific = deltas[0, np.arange(class_ids.shape[0]), class_ids]

    # Apply bounding box transformations
    # Shape: [N, (y1, x1, y2, x2)]
    refined_rois = utils.apply_box_deltas(
        proposals[0], deltas_specific * config.BBOX_STD_DEV)
    refined_rois = clip_boxes(refined_rois, window)
    return refined_rois


def filter_rois(refined_rois, class_ids, class_scores):
    # Remove boxes classified as background
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0]
        keep = np.intersect1d(keep, conf_keep)

    # Apply per-class non-max suppression
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_boxes = refined_rois[keep]
    unique_pre_nms_class_ids = np.unique(pre_nms_class_ids)

    nms_keep = []
    for class_id in unique_pre_nms_class_ids:
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                                pre_nms_scores[ixs],
                                                config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)

    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    return keep


def filter_fusion(masks):
    keep = []
    h, w = masks.shape[:2]
    for i in np.arange(masks.shape[2]):
        if np.count_nonzero(masks[:,:,i]) < 50 * 50:
            continue
        nonzeros = np.where(masks[:,:,i])
        keep = np.append(keep, i).astype(np.int32())

    return keep


def generate_result(bounding_boxes, segmentation, class_scores):
    h, w = segmentation.shape[:2]
    n = bounding_boxes.shape[0]

    class_ids = np.argmax(class_scores, axis=1)

    exported_masks = []
    exported_boxes = []
    exported_scores = []

    for m in range(n):
        if ((len(FILTER_CLASSES) == 0 or class_ids[m] in FILTER_CLASSES)
                and class_ids[m] not in STATIC_OBJECTS):
            exported_masks.append(segmentation[:,:,m])
            exported_boxes.append(bounding_boxes[m,:].tolist())
            exported_scores.append(class_scores[m,:].tolist())

    return exported_boxes, exported_masks, exported_scores


def execute(image):
    if (config is None or model is None):
        init()
    molded_image, image_meta, windows = model.mold_inputs([image])

    window = utils.norm_boxes(windows, molded_image[0].shape[:2])[0]
    anchors = model.get_anchors(molded_image[0].shape)
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,)
                              + anchors.shape)
    dects, probs, deltas, masks, proposals, _, _ = model.keras_model.predict(
                                            [molded_image, image_meta, anchors],
                                            verbose=0)

    # Class ID, and score per proposal
    class_ids = np.argmax(probs[0], axis=1)
    class_scores = probs[0, np.arange(class_ids.shape[0]), class_ids]

    refined_rois = refine_proposals(proposals, class_ids, deltas, window)

    keep = filter_rois(refined_rois, class_ids, class_scores)

    class_scores = probs[0][keep]

    # Detections are in different order than the indices in the "keep" array.
    # Thus, we need to identify the matching detections for correct ordering of
    # class_scores.
    bounding_boxes = utils.denorm_boxes(dects[0, :len(keep), :4],
                                        molded_image[0].shape[:2])
    roi_boxes = utils.denorm_boxes(refined_rois[keep],
                                   molded_image[0].shape[:2])

    perm = []
    for i in np.arange(len(keep)):
        perm = np.append(perm, np.where(np.all(roi_boxes == bounding_boxes[i],
                                               axis=1))[0][0]).astype(np.int32)

    class_scores = class_scores[perm]

    bounding_boxes, _, _, segmentation = model.unmold_detections(
                            dects[0], masks[0], image.shape,
                            molded_image[0].shape, windows[0])

    keep_fusion = filter_fusion(segmentation)

    return generate_result(bounding_boxes[keep_fusion],
                           segmentation[:,:,keep_fusion],
                           class_scores[keep_fusion])


def preprocess(image, filename):
    if (config is None or model is None):
        init()
    result = execute(image)
    with open(filename, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def load_preprocessed(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
