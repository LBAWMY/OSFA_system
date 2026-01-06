# python interpreter searches these subdirectories for modules
import sys

import numpy as np

sys.path.insert(0, '../yolov5')

import argparse
import os
import random
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

# yolov5
import sys
sys.path.append('../')
from yolov5.utils.datasets import LoadImages, LoadStreams, LoadRosImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.models.experimental import attempt_load
from yolov5.utils.plots import plot_one_box
from yolov5.utils.datasets import letterbox

# kalman filter
from KalmanFilter_multi import convert_bbox_to_z, convert_x_to_bbox, KalmanBoxTracker, Tracker, Object

# Ros topic image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import roslib

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# receieve the image and the predicted mask
img_LR_src = None
bridge = CvBridge()

def imageLeftRightCallback(img):
    global img_LR_src
    img_LR_src = bridge.imgmsg_to_cv2(img, "bgr8")

# Initialization: ROS
rospy.init_node("TD_node", anonymous=True)
# capture image
rospy.Subscriber("/camera1_2/usb_cam1_2/image_raw", Image, imageLeftRightCallback, queue_size=1, buff_size=2**24)
frequency = 50 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

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

def re_associate_detections_to_trackers(detections, trackers, objects, iou_threshold = 0.3):
    """
       Assigns detections to tracked object (both represented as bounding boxes)
       Input:
       detections: the detections
       trackers: the trackers object (list type)
       iou_threshold: the base condition to get a match
       Returns matches list
       """
    if (len(trackers) == 0):
        return np.empty((0, 5)), [], []

    # trackers bbox extraction
    tks_bbox = np.empty((0, 5))
    for index in range(len(trackers)):
        if len(trackers[index].cur_bbox) > 0:
            tks_bbox = np.vstack((tks_bbox, trackers[index].cur_bbox)) # TODO: need to refine later, need to be a bbox type: (n, 5)

    iou_matrix = iou_batch(detections, tks_bbox) # here the trackers are the bbox, not the tracker object

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
    for m in matched_indices:
        print('iou: ', iou_matrix[m[0], m[1]], 'matched bbox: ', m[0], 'matched filter: ', m[1], 'num of filter: ', len(trackers))
        if (iou_matrix[m[0], m[1]] >= iou_threshold):
            print('matched bbox: ', m[0], 'matched filter: ', m[1], 'num of filter: ', len(trackers))
            matches.append(m.reshape(1, 2))
            matched_bbox.append(detections[m[0]])
            matched_tracker.append(trackers[m[1]])
            matched_object.append(objects[m[1]])

    if (len(matches) == 0):
        matched_bbox = np.empty((0, 5), dtype=int)
    else:
        matched_bbox = np.array(matched_bbox)

    # split the matched bbox and the tracker respectively
    # matched_bbox = detections[matches[0, 0]]
    # matched_tracker = trackers[matches[0, 1]]

    return matched_bbox, matched_tracker, matched_object

def detect(opt, *args):
    out, source, weights, view_img, save_txt, imgsz, sort_max_age, sort_min_hits, sort_iou_thresh = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.tracker_max_age, opt.tracker_min_hits, opt.tracker_iou_thres
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    roscam = source.lower().endswith('image_raw')

    # Initial a tracker
    tracker1 = Tracker() # track tool1m
    tracker2 = Tracker() # track tool2m
    object1 = Object(num=30)
    object2 = Object(num=30)
    # frame count
    frame_count = 1

    # Directories
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # elif roscam:
    #     dataset = LoadRosImages(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[196, 173, 4], [253, 238, 123], [0, 200, 67], [99, 249, 138], [194, 33, 251], [214, 108, 251]] # filter bbox color + trajectory
    yolobbox_colors = [[0, 0, 0], [0, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 255]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # for path, img, im0s, vid_cap in dataset:
    path = None
    vid_cap = None
    count = 1
    while not rospy.is_shutdown():
        try:
            with torch.no_grad():

                print(type(img_LR_src))
                W, H = img_LR_src.shape[:2]
                L_Img = img_LR_src[:, :int(H/2), :]
                R_Img = img_LR_src[:, int(H/2):, :]

                # Padded resize
                img = letterbox(R_Img, 640, 32)[0]

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                # img = L_Img
                im0s = R_Img.copy()

                # left, right, R1, Q = undistort_stereo(L_Img,R_Img)

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    # if webcam:  # batch_size >= 1
                    #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    # else:
                    #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    s = ''
                    im0 = im0s

                    # p = Path(p)  # to Path
                    # save_path = str(Path(out) / p.name)  # img.jpg
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=yolobbox_colors[int(cls)], line_thickness=3)

                    dets_to_track1 = np.empty((0, 5))
                    dets_to_track2 = np.empty((0, 5))

                    # Pass the detection to the Kalman filter
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        if int(detclass) == 1:
                            dets_to_track1 = np.vstack((dets_to_track1, np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 3:
                            dets_to_track2 = np.vstack((dets_to_track2, np.array([x1, y1, x2, y2, conf])))

                    # print('Input into SORT:\n', dets_to_track, '\n')
                    # Run the Kalman filter
                    # need to left one observation (choose the best fit one)
                    # If the main instrument is detected for the first time, we need to initialize the Kalman filter
                    m1, tracked_dets1 = tracker1.update(dets_to_track1)
                    tracker1.update_bbox_heatmap(tracked_dets1)
                    object1.update(tracked_dets1)

                    print('track2 prob: ', dets_to_track2[:, -1])
                    m2, tracked_dets2 = tracker2.update(dets_to_track2)
                    tracker2.update_bbox_heatmap(tracked_dets2)
                    object2.update(tracked_dets2)

                    # TODO: Need to match the other unmatched detected objects
                    # Here we regard they only as the binary labels
                    m_all = [m1, m2]
                    left_matches = np.empty((0, 5))
                    left_trackers = []
                    left_object = []
                    if m1 == -1:
                        left_matches = np.vstack((left_matches, dets_to_track1))
                        left_trackers.append(tracker1)
                        left_object.append(object1)
                    else:
                        left_matches = np.vstack((left_matches, np.delete(dets_to_track1, m1, axis=0)))

                    if m2 == -1:
                        left_matches = np.vstack((left_matches, dets_to_track2))
                        left_trackers.append(tracker2)
                        left_object.append(object2)
                    else:
                        left_matches = np.vstack((left_matches, np.delete(dets_to_track2, m2, axis=0)))

                    # TODO: Need to match the other unmatched detected objects
                    print('left matches: ', left_matches, 'len of left tracker: ', len(left_trackers))
                    if (len(left_trackers) == 2):
                        cv2.imshow('results2', R_Img)
                        print('Debug!')
                    matched_bbox, matched_tracker, matched_object = re_associate_detections_to_trackers(left_matches, left_trackers, left_object, iou_threshold = 0.3)
                    print('len of matched tracker: ', len(matched_tracker))
                    for i in range(len(matched_bbox)):
                        m_temp, tracked_dets_temp = matched_tracker[i].update(matched_bbox[i].reshape((1, 5)))
                        matched_tracker[i].update_bbox_heatmap(tracked_dets_temp)
                        matched_object[i].update(tracked_dets_temp) # BUG here, the order is not consistent, SOLVED~

                    # object.update(tracked_dets[0, :4] if len(tracked_dets) > 0 else np.empty((0, 5))) # TODO

                    # print('Output from SORT:\n',tracked_dets,'\n')
                    t2 = time_synchronized()
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    if (save_img or view_img) and len(tracked_dets1) > 0:  # Add bbox to image + Add trajectory to image
                        score = 0 if m1 == -1 else 1
                        label = f'{names[int(1)]} {score:1d}'
                        plot_one_box(tracked_dets1[:4], im0, label=label, color=colors[int(0)], line_thickness=3)
                        object1.plot(im0, color=colors[int(1)])
                    if (save_img or view_img) and len(tracked_dets2) > 0:  # Add bbox to image + Add trajectory to image
                        score = 0 if m2 == -1 else 1
                        label = f'{names[int(3)]} {score:1d}'
                        plot_one_box(tracked_dets2[:4], im0, label=label, color=colors[int(2)], line_thickness=3)
                        object2.plot(im0, color=colors[int(3)])
                    # time.sleep(0.5)
                    # Stream results
                    if view_img:
                        frame_count_ = f'Frame {frame_count:d}'
                        cv2.putText(im0, frame_count_, (40, 40), 0, 1, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        cv2.imshow('results', im0)
                        cv2.waitKey(1)  # 1 millisecond
                        frame_count += 1

                    # count += 1
                    # if count % 2 == 0:
                    #     cv2.imshow('results1', im0)
                    # if count % 2 == 1:
                    #     cv2.imshow('results2', im0)
                    # Save results (image with detections)
                    # if save_img:
                    #     if dataset.mode == 'image':
                    #         cv2.imwrite(save_path, im0)
                    #     else:  # 'video' or 'stream'
                    #         if vid_path != save_path:  # new video
                    #             vid_path = save_path
                    #             if isinstance(vid_writer, cv2.VideoWriter):
                    #                 vid_writer.release()  # release previous video writer
                    #             if vid_cap:  # video
                    #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #             else:  # stream
                    #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #                 save_path += '.mp4'
                    #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #         vid_writer.write(im0)

        except TypeError:
            print('no data from camera')
            continue
        except AttributeError:
            print('no data from camera')
            continue
        except cv2.error:
            print('cv2 error')
            continue
        # except IndexError:
        #     print('Index error: no matching!')
        #     continue
        else:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    # Tracker settings
    parser.add_argument('--tracker-max-age', type=int, default=5,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--tracker-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--tracker-iou-thres', type=float, default=0.2,
                        help='intersection-over-union threshold between two frames for association')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        detect(opt)
