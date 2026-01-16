# python interpreter searches these subdirectories for modules
import sys

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly, label2order

# kalman filter
from KalmanFilter_multi_rot_10 import convert_bbox_to_z, convert_x_to_bbox, KalmanBoxTracker, Tracker, Object, xywh2xyxy
from CategoryFilter_multi import CategoryTracker
import numpy as np
import time

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        # hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        #        '2C99A8', '00C2FF', '344593', '6473FF', '00128EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        hex = ('3839ff', 'afb0ff', '1e6fff', 'a5c5ff', '2ed3d3', 'abeded', '924633', '3DDB86', '31961a', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '00128EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'fefe01') # last is the yellow
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def linear_assignment(cost_matrix):
    try:
        import lap # linear assignment problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i>=0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cat = int(categories[i]) if categories is not None else 0

        id = int(identities[i]) if identities is not None else 0

        color = compute_color_for_labels(id)

        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def re_associate_detections_to_trackers(detections, trackers, objects, trackers_topleft, detections_topleft, iou_threshold = 0.3):
    """
       Assigns detections to tracked object (both represented as bounding boxes)
       Input:
       detections: the detections
       trackers: the trackers object (list type)
       iou_threshold: the base condition to get a match
       Returns matches list
       """
    if (len(trackers) == 0):
        return np.empty((0, 6)), [], [], [], []

    # trackers bbox extraction
    tks_bbox = np.empty((0, 6))
    for index in range(len(trackers)):
        if len(trackers[index].cur_bbox) > 0:
            tks_bbox = np.vstack((tks_bbox, trackers[index].cur_bbox)) # TODO: need to refine later, need to be a bbox type: (n, 5)

    # iou_matrix = iou_batch(xywh2xyxy(dets), xywh2xyxy(pred))
    iou_matrix = iou_batch(xywh2xyxy(detections), xywh2xyxy(tks_bbox)) # here the trackers are the bbox, not the tracker object

    # we can rebuilt a cost matrix to replace the iou_matrix later.

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # filter out matched with low IOU
    matches = []
    matched_bbox = []
    matched_tracker = []
    matched_object = []
    matched_tracker_topleft = []
    matched_topleft = []
    for m in matched_indices:
        print('iou: ', iou_matrix[m[0], m[1]], 'matched bbox: ', m[0], 'matched filter: ', m[1], 'num of filter: ', len(trackers))
        if (iou_matrix[m[0], m[1]] >= iou_threshold):
            print('matched bbox: ', m[0], ', matched filter: ', m[1], ', num of filter: ', len(trackers),
                  ', num of trackers_topleft: ', len(trackers_topleft),
                  ', num of detections_topleft: ', len(detections_topleft))
            matches.append(m.reshape(1, 2))
            matched_bbox.append(detections[m[0]])
            matched_tracker.append(trackers[m[1]])
            matched_object.append(objects[m[1]])
            matched_tracker_topleft.append(trackers_topleft[m[1]])
            # matched_topleft.append(detections_topleft[m[1]])
            matched_topleft.append(detections_topleft[m[0]])

    if (len(matches) == 0):
        matched_bbox = np.empty((0, 6), dtype=int)
    else:
        matched_bbox = np.array(matched_bbox)

    # split the matched bbox and the tracker respectively
    # matched_bbox = detections[matches[0, 0]]
    # matched_tracker = trackers[matches[0, 1]]

    return matched_bbox, matched_tracker, matched_object, matched_tracker_topleft, matched_topleft

def TopLeftIndexTrack(dets_to_topleft, threshold_count=8):
    """
    Basic tracking plan: count the closest index number
    The index is valid only when it appear continiously
    If it always keep changing, return -1 which will treat the center position
    We will plot the topleft position only when the prediction is 1, 2, 3, 4
    The continously count >= threshold is valid.
    """
    # tracked_topleft3 = dets_to_topleft3[-1, 0] if dets_to_topleft3.shape[0] > 0 else 1
    # tracked_topleft3 = TopLeftIndexTrack(dets_to_topleft3)
    # set a array to count the index information
    pass

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initial a tracker
    tracker1 = Tracker() # track tool1m
    tracker2 = Tracker() # track tool2m
    tracker3 = Tracker() # track tool3m
    object1 = Object(num=25)
    object2 = Object(num=25)
    object3 = Object(num=25)
    tracker1_TopLeft = CategoryTracker()
    tracker2_TopLeft = CategoryTracker()
    tracker3_TopLeft = CategoryTracker()
    # frame count
    frame_count = 1
    raw_topleft1 = []
    raw_topleft2 = []
    raw_topleft3 = []
    topleft1 = []
    topleft2 = []
    topleft3 = []
    count = []

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # colors = [[196, 173, 4], [253, 238, 123], [0, 200, 67], [99, 249, 138], [194, 33, 251], [214, 108, 251]] # filter bbox color + trajectory
    colors = Colors()  # create instance for 'from utils.plots import colors'
    yolobbox_colors = [[0, 0, 0], [0, 0, 0], [0, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 255]]

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    t0 = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        # pred: list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        if (frame_count == 89):
            print('Debug!')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            det[:, 5] = torch.squeeze(label2order(theta=det[:, 4], label=det[:, 5]))
            # pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                # det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xywht, topleft, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        xywht = torch.unsqueeze(torch.tensor(xywht, device=device), dim=0)
                        pred_poly = rbox2poly(xywht)  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                        pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)[0,:]
                        # annotator.poly_label(pred_poly, top_left=None, label=label, color=colors(c+6, True))

            dets_to_track1 = np.empty((0, 6))
            dets_to_track2 = np.empty((0, 6))
            dets_to_track3 = np.empty((0, 6))

            dets_to_topleft1 = np.empty((0, 2))
            dets_to_topleft2 = np.empty((0, 2))
            dets_to_topleft3 = np.empty((0, 2))

            # Pass the detection to the Kalman filter
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for cx, cy, w, h, theta, topleft, conf, detclass in det.cpu().detach().numpy():
                if int(detclass) == 1: # 0:
                    dets_to_track1 = np.vstack((dets_to_track1, np.array([cx, cy, w, h, theta, conf])))
                    dets_to_topleft1 = np.vstack((dets_to_topleft1, np.array([theta, topleft], dtype=int)))
                if int(detclass) == 3: # 2:
                    dets_to_track2 = np.vstack((dets_to_track2, np.array([cx, cy, w, h, theta, conf])))
                    dets_to_topleft2 = np.vstack((dets_to_topleft2, np.array([theta, topleft], dtype=int)))
                if int(detclass) == 5: # 4:
                    dets_to_track3 = np.vstack((dets_to_track3, np.array([cx, cy, w, h, theta, conf])))
                    dets_to_topleft3 = np.vstack((dets_to_topleft3, np.array([theta, topleft], dtype=int)))

            # print('Input into SORT:\n', dets_to_track, '\n')
            # Run the Kalman filter
            # need to left one observation (choose the best fit one)
            # If the main instrument is detected for the first time, we need to initialize the Kalman filter
            m1, tracked_dets1 = tracker1.update(dets_to_track1)
            if m1 != -1:
                _, tracked_topleft1 = tracker1_TopLeft.update(tracked_dets1[0, 4], dets_to_topleft1)
            tracker1.update_bbox_heatmap(tracked_dets1)
            object1.update(tracked_dets1)

            print('track2 prob: ', dets_to_track2[:, -1])
            m2, tracked_dets2 = tracker2.update(dets_to_track2)
            if m2 != -1:
                _, tracked_topleft2 = tracker2_TopLeft.update(tracked_dets2[0, 4], dets_to_topleft2)
            tracker2.update_bbox_heatmap(tracked_dets2)
            object2.update(tracked_dets2)

            m3, tracked_dets3 = tracker3.update(dets_to_track3)
            if m3 != -1:
                _, tracked_topleft3 = tracker3_TopLeft.update(tracked_dets3[0, 4], dets_to_topleft3)
            tracker3.update_bbox_heatmap(tracked_dets3)
            object3.update(tracked_dets3)

            # TODO: Need to match the other unmatched detected objects
            # Here we regard they only as the binary labels
            m_all = [m1, m2, m3]
            left_matches = np.empty((0, 6))
            left_trackers = []
            left_object = []
            left_matches_topleft = np.empty((0, 2))
            left_trackers_topleft = []
            if m1 == -1:
                left_matches = np.vstack((left_matches, dets_to_track1))
                left_trackers.append(tracker1)
                left_object.append(object1)

                left_matches_topleft = np.vstack((left_matches_topleft, dets_to_topleft1))
                left_trackers_topleft.append(tracker1_TopLeft)
            else:
                left_matches = np.vstack((left_matches, np.delete(dets_to_track1, m1, axis=0)))
                left_matches_topleft = np.vstack((left_matches_topleft, np.delete(dets_to_topleft1, m1, axis=0)))

            if m2 == -1:
                left_matches = np.vstack((left_matches, dets_to_track2))
                left_trackers.append(tracker2)
                left_object.append(object2)

                left_matches_topleft = np.vstack((left_matches_topleft, dets_to_topleft2))
                left_trackers_topleft.append(tracker2_TopLeft)
            else:
                left_matches = np.vstack((left_matches, np.delete(dets_to_track2, m2, axis=0)))
                left_matches_topleft = np.vstack((left_matches_topleft, np.delete(dets_to_topleft2, m2, axis=0)))

            if m3 == -1:
                left_matches = np.vstack((left_matches, dets_to_track3))
                left_trackers.append(tracker3)
                left_object.append(object3)

                left_matches_topleft = np.vstack((left_matches_topleft, dets_to_topleft3))
                left_trackers_topleft.append(tracker3_TopLeft)
            else:
                left_matches = np.vstack((left_matches, np.delete(dets_to_track3, m3, axis=0)))
                left_matches_topleft = np.vstack((left_matches_topleft, np.delete(dets_to_topleft3, m3, axis=0)))

            # TODO: Need to match the other unmatched detected objects
            print('left matches: ', left_matches, 'len of left tracker: ', len(left_trackers))
            if (frame_count == 1000): # 514
                raw_img = im0.copy()
                heatmap1 = tracker1.bbox_area / np.max(tracker1.bbox_area)
                # must convert to type unit8
                heatmap1 = np.uint8(255 * heatmap1)
                heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)

                superimposed_img1 = heatmap1 * 0.6 + raw_img * 0.3
                cv2.imwrite('./inference/' + str(frame_count).zfill(6) + '_tracker1.jpg', superimposed_img1)

                raw_img = im0.copy()
                heatmap2 = tracker2.bbox_area / np.max(tracker2.bbox_area)
                # must convert to type unit8
                heatmap2 = np.uint8(255 * heatmap2)
                heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)

                superimposed_img2 = heatmap2 * 0.6 + raw_img * 0.3
                cv2.imwrite('./inference/' + str(frame_count).zfill(6) + '_tracker2.jpg', superimposed_img2)

                raw_img = im0.copy()
                heatmap3 = tracker3.bbox_area / np.max(tracker3.bbox_area)
                # must convert to type unit8
                heatmap3 = np.uint8(255 * heatmap3)
                heatmap3 = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)

                superimposed_img3 = heatmap3 * 0.6 + raw_img * 0.3
                cv2.imwrite('./inference/' + str(frame_count).zfill(6) + '_tracker3.jpg', superimposed_img3)
                print('Debug!')
            matched_bbox, matched_tracker, matched_object, matched_tracker_topleft, matched_topleft = re_associate_detections_to_trackers(left_matches, left_trackers, left_object, left_trackers_topleft, left_matches_topleft, iou_threshold=0.3)
            print('len of matched tracker: ', len(matched_tracker))
            for i in range(len(matched_bbox)):
                m_temp, tracked_dets_temp = matched_tracker[i].update(matched_bbox[i].reshape((1, 6)))
                matched_tracker[i].update_bbox_heatmap(tracked_dets_temp)
                matched_object[i].update(tracked_dets_temp) # BUG here, the order is not consistent, SOLVED~
                if m_temp == 1:
                    _, _ = matched_tracker_topleft[i].update(tracked_dets_temp[0, 4], matched_topleft[i])

            # object.update(tracked_dets[0, :4] if len(tracked_dets) > 0 else np.empty((0, 5))) # TODO

            # print('Output from SORT:\n',tracked_dets,'\n')

            print(f'{s}Done. ({time_sync() - t1:.3f}s, {1/(time_sync() - t2):.2f}fps)')

            # use the final tracked theta for calculation

            if (save_img or view_img) and len(tracked_dets1) > 0:  # Add bbox to image + Add trajectory to image
                score = 0 if m1 == -1 else 1
                label = f'{names[int(1)]} {score:1d}'
                tracked_poly1 = rbox2poly(np.expand_dims(tracked_dets1[0, :5], axis=0))[0,:]  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                # _, tracked_topleft1 = tracker1_TopLeft.update(tracked_dets1[0, 4], dets_to_topleft1)
                tracked_topleft1 = tracker1_TopLeft.return_category
                raw_tl1 = dets_to_topleft1[0, 1] if dets_to_topleft1.shape[0] > 0 else 0
                annotator.poly_label(tracked_poly1, tracked_topleft1, label, color=colors(int(0), True), topleft_color=colors(int(-1), True))
                object1.plot(im0, color=colors(int(1), True))
            else:
                tracked_topleft1 = 0
                raw_tl1 = 0
            if (save_img or view_img) and len(tracked_dets2) > 0:  # Add bbox to image + Add trajectory to image
                score = 0 if m2 == -1 else 1
                label = f'{names[int(3)]} {score:1d}'
                tracked_poly2 = rbox2poly(np.expand_dims(tracked_dets2[0, :5], axis=0))[0,:]  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                # _, tracked_topleft2 = tracker2_TopLeft.update(tracked_dets2[0, 4], dets_to_topleft2)
                tracked_topleft2 = tracker2_TopLeft.return_category
                raw_tl2 = dets_to_topleft2[0, 1] if dets_to_topleft2.shape[0] > 0 else 0
                annotator.poly_label(tracked_poly2, tracked_topleft2, label, color=colors(int(2), True), topleft_color=colors(int(-1), True))
                object2.plot(im0, color=colors(int(3), True))
            else:
                tracked_topleft2 = 0
                raw_tl2 = 0
            if (save_img or view_img) and len(tracked_dets3) > 0:  # Add bbox to image + Add trajectory to image
                score = 0 if m3 == -1 else 1
                label = f'{names[int(5)]} {score:1d}'
                tracked_poly3 = rbox2poly(np.expand_dims(tracked_dets3[0, :5], axis=0))[0,:]  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                # _, tracked_topleft3 = tracker3_TopLeft.update(tracked_dets3[0, 4], dets_to_topleft3)
                tracked_topleft3 = tracker3_TopLeft.return_category
                raw_tl3 = dets_to_topleft3[0, 1] if dets_to_topleft3.shape[0] > 0 else 0
                annotator.poly_label(tracked_poly3, tracked_topleft3, label, color=colors(int(4), True), topleft_color=colors(int(-1), True))
                object3.plot(im0, color=colors(int(5), True))
            else:
                tracked_topleft3 = 0
                raw_tl3 = 0

            topleft1.append(tracked_topleft1)
            topleft2.append(tracked_topleft2)
            topleft3.append(tracked_topleft3)

            raw_topleft1.append(raw_tl1)
            raw_topleft2.append(raw_tl2)
            raw_topleft3.append(raw_tl3)
            count.append(frame_count)

            # Stream results
            if view_img:
                frame_count_ = f'Frame {frame_count:d}'
                # cv2.putText(im0, frame_count_, (100, 50), 0, 2, [225, 255, 255], thickness=3,
                #             lineType=cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                frame_count += 1

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        save_path.replace('.avi', '.mp4')
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    save_np_path = save_path.replace('.MP4', '.npz')
    save_np_path = save_path.replace('.avi', '.npz')
    np.savez(save_np_path, count=np.array(count), topleft1=np.array(topleft1), topleft2=np.array(topleft2), topleft3=np.array(topleft3),
             raw_topleft1=np.array(raw_topleft1), raw_topleft2=np.array(raw_topleft2), raw_topleft3=np.array(raw_topleft3))

    vid_writer.release()
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.subplots()  # type(ax1) = axes
    ax2 = fig2.subplots()

    save_img_path = save_path.replace('.MP4', '.png')
    save_img_path = save_path.replace('.avi', '.png')
    ax1.plot(count, topleft1, 'r', label='tracker1')
    ax1.plot(count, topleft2, 'g', label='tracker2')
    ax1.plot(count, topleft3, 'b', label='tracker3')
    ax1.legend()
    ax1.set_xlabel('frame count')
    ax1.set_ylabel('topleft label')

    save_rawimg_path = save_path.replace('.MP4', '_raw.png')
    save_rawimg_path = save_path.replace('.avi', '_raw.png')
    ax2.plot(count, raw_topleft1, 'r', label='raw-tracker1')
    ax2.plot(count, raw_topleft2, 'g', label='raw-tracker2')
    ax2.plot(count, raw_topleft3, 'b', label='raw-tracker3')
    ax2.legend()
    ax2.set_xlabel('frame count')
    ax2.set_ylabel('topleft raw label')

    plt.show()
    # fig1.savefig(save_img_path)
    # fig2.savefig(save_rawimg_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5m_finetune/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='dataset/dataset_demo_rate1.0_split1024_gap200/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1024], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
