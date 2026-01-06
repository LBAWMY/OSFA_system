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
from KalmanFilter_multi import convert_bbox_to_z, convert_x_to_bbox, KalmanBoxTracker, Tracker, Object, KalmanDepthTracker

# Ros topic image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import roslib
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64, Int32, Bool
from yolo_bbox_msg.msg import Boundingbox

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# receieve the image and the predicted mask
img_LR_src = None
main_tool_id = 4
bbox_scale = 1.0 # from 0~2
mid_depth = 75 # mm
bridge = CvBridge()
pred_action_msg = Vector3()
pred_position_msg = Vector3()
depth_msg = Float64()

def imageLeftRightCallback(img):
    global img_LR_src
    img_LR_src = bridge.imgmsg_to_cv2(img, "bgr8")

def mainToolIdxCallback(idx):
    global main_tool_id
    main_tool_id = idx.data

def bboxScaleCallback(scale):
    global bbox_scale
    bbox_scale = scale.data

def midDepthCallback(depth):
    global mid_depth
    mid_depth = depth.data

# Initialization: ROS
rospy.init_node("TD_node", anonymous=True)
# capture image
# rospy.Subscriber("/camera1_2/usb_cam1_2/image_raw", Image, imageLeftRightCallback, queue_size=1, buff_size=2**24)
rospy.Subscriber("/camera1_2/usb_cam1_2/image_raw", Image, imageLeftRightCallback)
# acquire the main tool idx
rospy.Subscriber("/multitools/main_tool_id", Int32, mainToolIdxCallback)
# acquire the bbox scale
rospy.Subscriber("/customize_settings/scale_bbox", Float64, bboxScaleCallback)
# acquire the depth range
rospy.Subscriber("/customize_settings/mid_depth", Float64, midDepthCallback)
# publish the predicted action
pred_action_pub = rospy.Publisher("/track/pred_action", Vector3, queue_size=10)
# publish the current white/yellow bbox
whitebbox_refer_pub = rospy.Publisher("/visualize_bbox/white_bbox", Boundingbox, queue_size=10)
yellowbbox_refer_pub = rospy.Publisher("/visualize_bbox/yellow_bbox", Boundingbox, queue_size=10)
whetherbbox_visualize_pub = rospy.Publisher("/visualize_bbox/whether_visualize", Bool, queue_size=10)
white_bbox_msg = Boundingbox()
yellow_bbox_msg = Boundingbox()
whether_visualize_msg = Bool()
# publish the predicted position
NUM_TOOLS_TO_TRACK = 5
# main_tool_id = 1 # the second tool
pred_position_tools_pub = [rospy.Publisher("/track/pred_position_tool" + str(i), Vector3, queue_size=10) for i in range(NUM_TOOLS_TO_TRACK)]
# pred_position_main_pub = rospy.Publisher("/pred_position_main", Vector3, queue_size=10)
tool_depth_pub = rospy.Publisher("/track/main_tool_depth", Float64, queue_size=10)
frequency = 50 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

def undistort_stereo(img_left, img_right):
    h, w = img_left.shape[:2]

    cameraMatrixl = np.array([[662.2442132308511, 0, 344.9069167552053],
                              [0, 663.4465422680099, 244.9097816667816],
                              [0, 0, 1]
                              ])
    distCoeffsl = np.array([-0.3998422020995984, 0.1974903638339781, -0.000977499191867238, 0.001463321449655296, -0.07176023016948978])

    cameraMatrixr = np.array([[672.156450947434, 0, 337.1348007607834],
                              [0, 659.4031526687482, 233.1060205508514],
                              [0, 0, 1]
                              ])
    distCoeffsr = np.array([-0.4053127751351984, 0.2601311290555161, -0.001659039864156836, -0.001179532311658814, -0.09147876464963214])

    R = np.array([[0.9999207022053717, -0.0001611069341282794, 0.01259219383873574],
                  [7.297434460661222e-05, 0.9999755031114125, 0.006999132219243628],
                  [-0.0125930129778996, -0.006997658296401788, 0.9998962190160063]])
    T = np.array([[-4.021515245243654],[-0.1062506235283896],[0.08107672340610546]])

    new_size = (int(w*1.5), int(h*1.5))
    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    #     cv2.stereoRectify(cameraMatrixl, distCoeffsl, cameraMatrixr, distCoeffsr, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrixl, distCoeffsl, cameraMatrixr, distCoeffsr, (w, h), R, T, flags=1, alpha=1)
    map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrixl, distCoeffsl, R1, P1, (w, h), cv2.CV_16SC2)
    map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrixr, distCoeffsr, R2, P2, (w, h), cv2.CV_16SC2)
    result1 = cv2.remap(img_left, map1_1, map1_2, cv2.INTER_LINEAR)
    result2 = cv2.remap(img_right, map2_1, map2_2, cv2.INTER_LINEAR)
    return result1, result2, R1, R2, Q, map2_1

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

def SteroTempleteDepthCal(leftImg, l_bbox, rightImg, resize_factor, R1, Q):
    height, width = leftImg.shape[:2]
    size = (int(width*resize_factor), int(height*resize_factor))
    leftImg_ = cv2.resize(leftImg.copy(), size, interpolation = cv2.INTER_AREA)
    rightImg_ = cv2.resize(rightImg.copy(), size, interpolation = cv2.INTER_AREA)
    print('l_bbox: ', l_bbox, 'width: ', width, 'height: ', height)
    if (len(l_bbox) != 0 and min(l_bbox) > 0 and max([l_bbox[0], l_bbox[2]]) <= width and max([l_bbox[1], l_bbox[3]]) <= height):
        # print(l_bbox)
        l_box = [int(l_bbox[0]*resize_factor), int(l_bbox[1]*resize_factor), int(l_bbox[2]*resize_factor), int(l_bbox[3]*resize_factor)]
        # print(l_box)
        # print(leftImg_.shape)
        # print(rightImg_.shape)
        left_template = leftImg_[l_box[1]:l_box[3], l_box[0]:l_box[2], :]
        # print('left rect shape: ', left_template.shape)
        cv2.imshow('left template', left_template)
        d1 = (left_template - left_template.mean()) / left_template.std()
        ncc_value = np.zeros((1, leftImg_.shape[1]))
        # start_index = l_box[0]
        # end_index = np.minimum(start_index+100, right.shape[0])
        start_index = np.maximum(0, int(l_box[0]-100*resize_factor))
        end_index = l_box[0]
        print('start index: ', start_index, 'end index: ', end_index)
        for index in range(start_index, end_index):
            right_rect = rightImg_[l_box[1]:l_box[3], index:index+l_box[2]-l_box[0],:]
            # print(index)
            # print('right rect shape: ', right_rect.shape)
            d2 = (right_rect - right_rect.mean()) / right_rect.std()
            ncc_value[0, index] = np.sum(d1 * d2)

        u_l = l_box[0]
        v_l = l_box[1]

        u_r = np.argmax(ncc_value)
        v_r = l_box[1]

        visual_d = u_l - u_r
        pixel_coordinate = [u_l / resize_factor, v_l / resize_factor, visual_d / resize_factor, 1]
        pixel_coordinate = np.array(pixel_coordinate)
        pixel_coordinate.reshape([4, 1])
        xyz_rectified = np.dot(Q, pixel_coordinate)

        T_toLeftCam = np.zeros([4, 4], dtype=np.float32)
        T_toLeftCam[0:3, 0:3] = R1.T
        T_toLeftCam[3, 3] = 1
        xyz_rectified = np.dot(T_toLeftCam, xyz_rectified)

        xyz_rectified = xyz_rectified/xyz_rectified[3]
        tool_depth = xyz_rectified[2]
        # tool_depth = 1
        cv2.imshow('Matched RightImg', rightImg_[l_box[1]:l_box[3], u_r:u_r+l_box[2]-l_box[0],])
    else:
        print('no valid depth estimation ! The predicted bbox is out of the image.')
        tool_depth = -1

    return tool_depth

def SteroTempleteDepthCal_right(rightImg, r_bbox, leftImg, resize_factor, R2, Q):
    height, width = rightImg.shape[:2]
    size = (int(width*resize_factor), int(height*resize_factor))
    leftImg_ = cv2.resize(leftImg.copy(), size, interpolation = cv2.INTER_AREA)
    rightImg_ = cv2.resize(rightImg.copy(), size, interpolation = cv2.INTER_AREA)
    print('r_bbox: ', r_bbox, 'width: ', width, 'height: ', height)
    if (len(r_bbox) != 0 and min(r_bbox) > 0 and max([r_bbox[0], r_bbox[2]]) <= width and max([r_bbox[1], r_bbox[3]]) <= height):
        # print(l_bbox)
        r_box = [int(r_bbox[0]*resize_factor), int(r_bbox[1]*resize_factor), int(r_bbox[2]*resize_factor), int(r_bbox[3]*resize_factor)]
        print('int r_box', r_box)
        print(leftImg_.shape)
        print(rightImg_.shape)
        right_template = rightImg_[r_box[1]:r_box[3], r_box[0]:r_box[2], :]
        print('right rect shape: ', right_template.shape)
        cv2.imshow('right template', right_template)
        d1 = (right_template - right_template.mean()) / right_template.std()
        ncc_value = np.zeros((1, rightImg_.shape[1]))
        # start_index = l_box[0]
        # end_index = np.minimum(start_index+100, right.shape[0])
        start_index = r_box[2]
        end_index = np.minimum(int(r_box[2]+100*resize_factor), int(width*resize_factor))
        print('start index: ', start_index, 'end index: ', end_index)
        for index in range(start_index, end_index):
            left_rect = leftImg_[r_box[1]:r_box[3], index-(r_box[2]-r_box[0]):index,:]
            # print(index)
            # print('left rect shape: ', left_rect.shape)
            d2 = (left_rect - left_rect.mean()) / left_rect.std()
            ncc_value[0, index] = np.sum(d1 * d2)

        u_r = r_box[2]
        v_r = r_box[3]

        u_l = np.argmax(ncc_value)
        v_l = r_box[3]

        visual_d = u_l - u_r
        pixel_coordinate = [u_l / resize_factor, v_l / resize_factor, visual_d / resize_factor, 1]
        pixel_coordinate = np.array(pixel_coordinate)
        pixel_coordinate.reshape([4, 1])
        xyz_rectified = np.dot(Q, pixel_coordinate)

        T_toLeftCam = np.zeros([4, 4], dtype=np.float32)
        T_toLeftCam[0:3, 0:3] = R2.T
        T_toLeftCam[3, 3] = 1
        xyz_rectified = np.dot(T_toLeftCam, xyz_rectified)

        xyz_rectified = xyz_rectified/xyz_rectified[3]
        tool_depth = xyz_rectified[2]
        # tool_depth = 1
        cv2.imshow('Matched LeftImg', leftImg_[r_box[1]:r_box[3], u_l-(r_box[2]-r_box[0]):u_l,])
    else:
        print('no valid depth estimation ! The predicted bbox is out of the image.')
        tool_depth = -1

    return tool_depth

def RectErrorCal(pt, bbox):
    """
    Calculate the min distance from one point to a rect contour (outside the rect)
       Input:
       pt: the coordinates of a point (x, y)
       bbox: the rect points (top-left, bottom-right)
       Returns the error in x, y axis
    Here we seperate the area into 8 areas:
    1  |  2  | 3
    --------------
    4  |  0  | 5
    --------------
    6  /  7  / 8
    """
    pt_x = pt[0]
    pt_y = pt[1]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    if (pt_x > 0 and pt_x < x1) and (pt_y > 0 and pt_y < y1):
        x_error = pt_x - x1
        y_error = pt_y - y1
    elif (pt_x > x1 and pt_x < x2) and (pt_y > 0 and pt_y < y1):
        x_error = 0
        y_error = pt_y - y1
    elif (pt_x > x2) and (pt_y > 0 and pt_y < y1):
        x_error = pt_x - x2
        y_error = pt_y - y1
    elif (pt_x > 0 and pt_x < x1) and (pt_y > y1 and pt_y < y2):
        x_error = pt_x - x1
        y_error = 0
    # elif (pt_x > x1 and pt_x < x2) and (pt_y > y1 and pt_y < y2):
    #     x_error = 0
    #     y_error = 0
    elif (pt_x > x2) and (pt_y > y1 and pt_y < y2):
        x_error = pt_x - x2
        y_error = 0
    elif (pt_x > 0 and pt_x < x1) and (pt_y > y2):
        x_error = pt_x - x1
        y_error = pt_y - y2
    elif (pt_x > x1 and pt_x < x2) and (pt_y > y2):
        x_error = 0
        y_error = pt_y - y2
    elif (pt_x > x2) and (pt_y > y2):
        x_error = pt_x - x2
        y_error = pt_y - y2
    else:
        x_error = 0
        y_error = 0

    return x_error, y_error

def DepthActionCal(depth, minmax):
    """
    Calculate the [-1, 1] depth action with the setting bound,
    """
    reference = (minmax[1] - minmax[0])
    if depth <= 0 or (depth > minmax[0] and depth < minmax[1]):
        return 0.0, 0.0
    else:
        if depth <= minmax[0]:
            return (depth - minmax[0]), (depth - minmax[0]) / reference
        if depth >= minmax[1]:
            return (depth - minmax[1]), (depth - minmax[1]) / reference

def detect(opt, *args):
    out, source, weights, view_img, save_txt, imgsz, sort_max_age, sort_min_hits, sort_iou_thresh = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.tracker_max_age, opt.tracker_min_hits, opt.tracker_iou_thres
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    roscam = source.lower().endswith('image_raw')

    # Initial a tracker
    trackers = [Tracker() for _ in range(NUM_TOOLS_TO_TRACK)]
    objects = [Object() for _ in range(NUM_TOOLS_TO_TRACK)]
    # tracker1 = Tracker() # track tool1m
    # tracker2 = Tracker() # track tool2m
    # tracker3 = Tracker() # track tool3m
    # tracker4 = Tracker() # track tool4m
    # tracker5 = Tracker() # track tool5m
    # object1 = Object(num=10)
    # object2 = Object(num=10)
    # object3 = Object(num=10)
    # object4 = Object(num=10)
    # object5 = Object(num=10)

    main_tool_depth = KalmanDepthTracker() # filter the main instrument
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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    yolobbox_colors = colors
    # colors = [[196, 173, 4], [253, 238, 123], [0, 200, 67], [99, 249, 138], [194, 33, 251], [214, 108, 251]] # filter bbox color + trajectory
    # yolobbox_colors = [[0, 0, 0], [0, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 255]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # for path, img, im0s, vid_cap in dataset:
    path = None
    vid_cap = None
    count = 1
    is_out_rect = False
    # record the max distance when the tool is out of the bbox
    x_e_max = 0
    y_e_max = 0
    while not rospy.is_shutdown():
        try:
            with torch.no_grad():

                print(type(img_LR_src))
                W, H = img_LR_src.shape[:2]
                L_Img = img_LR_src[:, :int(H/2), :]
                R_Img = img_LR_src[:, int(H/2):, :]

                left, right, R1, R2, Q, map2_1 = undistort_stereo(L_Img,R_Img)

                # Padded resize
                imgR = letterbox(right, 640, 32)[0]
                imgL = letterbox(left, 640, 32)[0]

                # Convert
                imgR = imgR[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                imgR = np.ascontiguousarray(imgR)
                imgL = imgL[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                imgL = np.ascontiguousarray(imgL)

                # merge into one batch
                # img = np.stack((imgR, imgL), axis=0)
                img = imgL#imgR # use the right frame for detection
                print('img shape: ', imgR.shape)
                # img = L_Img
                # im0s = right.copy()
                im0s = left.copy()

                # left, right, R1, Q = undistort_stereo(L_Img,R_Img)

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                ###############################################################
                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()
                print('names: ', names)
                print('Prediction Done.', (t2 - t1), 's')
                print('pred shape: ', len(pred))

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

                    dets_to_track = [np.empty((0, 5)) for _ in range(NUM_TOOLS_TO_TRACK)]
                    # dets_to_track1 = np.empty((0, 5))
                    # dets_to_track2 = np.empty((0, 5))
                    # dets_to_track3 = np.empty((0, 5))
                    # dets_to_track4 = np.empty((0, 5))
                    # dets_to_track5 = np.empty((0, 5))

                    # Pass the detection to the Kalman filter
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        # dets_to_track[int(detclass) // 2 - 1] = np.vstack((dets_to_track[int(detclass) // 2 - 1], np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 2:
                            dets_to_track[0] = np.vstack((dets_to_track[0], np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 4:
                            dets_to_track[1] = np.vstack((dets_to_track[1], np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 6:
                            dets_to_track[2] = np.vstack((dets_to_track[2], np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 14:
                            dets_to_track[3] = np.vstack((dets_to_track[3], np.array([x1, y1, x2, y2, conf])))
                        if int(detclass) == 10:
                            dets_to_track[4] = np.vstack((dets_to_track[4], np.array([x1, y1, x2, y2, conf])))

                    # print('Input into SORT:\n', dets_to_track, '\n')
                    # Run the Kalman filter
                    # need to left one observation (choose the best fit one)
                    # If the main instrument is detected for the first time, we need to initialize the Kalman filter
                    ms = [-1] * NUM_TOOLS_TO_TRACK
                    tracked_dets = []
                    for i in range(NUM_TOOLS_TO_TRACK):
                        ms[i], tracked_det = trackers[i].update(dets_to_track[i])
                        tracked_dets.append(tracked_det)
                        trackers[i].update_bbox_heatmap(tracked_det)
                        objects[i].update(tracked_det)

                    # m1, tracked_dets1 = tracker1.update(dets_to_track1)
                    # tracker1.update_bbox_heatmap(tracked_dets1)
                    # object1.update(tracked_dets1)
                    #
                    # print('track2 prob: ', dets_to_track2[:, -1])
                    # m2, tracked_dets2 = tracker2.update(dets_to_track2)
                    # tracker2.update_bbox_heatmap(tracked_dets2)
                    # object2.update(tracked_dets2)
                    #
                    # m3, tracked_dets3 = tracker3.update(dets_to_track3)
                    # tracker3.update_bbox_heatmap(tracked_dets3)
                    # object3.update(tracked_dets3)
                    #
                    # m4, tracked_dets4 = tracker4.update(dets_to_track4)
                    # tracker4.update_bbox_heatmap(tracked_dets4)
                    # object4.update(tracked_dets4)
                    #
                    # m5, tracked_dets5 = tracker5.update(dets_to_track5)
                    # tracker5.update_bbox_heatmap(tracked_dets5)
                    # object5.update(tracked_dets5)

                    # TODO: Need to match the other unmatched detected objects
                    # Here we regard they only as the binary labels
                    # m_all = [m1, m2, m3, m4, m5]
                    left_matches = np.empty((0, 5))
                    left_trackers = []
                    left_object = []
                    for i, m in enumerate(ms):
                        if m == -1:
                            left_matches = np.vstack((left_matches, dets_to_track[i]))
                            left_trackers.append(trackers[i])
                            left_object.append(objects[i])
                        else:
                            left_matches = np.vstack((left_matches, np.delete(dets_to_track[i], m, axis=0)))

                    # if m2 == -1:
                    #     left_matches = np.vstack((left_matches, dets_to_track2))
                    #     left_trackers.append(tracker2)
                    #     left_object.append(object2)
                    # else:
                    #     left_matches = np.vstack((left_matches, np.delete(dets_to_track2, m2, axis=0)))
                    #
                    # if m3 == -1:
                    #     left_matches = np.vstack((left_matches, dets_to_track3))
                    #     left_trackers.append(tracker3)
                    #     left_object.append(object3)
                    # else:
                    #     left_matches = np.vstack((left_matches, np.delete(dets_to_track3, m3, axis=0)))
                    #
                    # if m4 == -1:
                    #     left_matches = np.vstack((left_matches, dets_to_track4))
                    #     left_trackers.append(tracker4)
                    #     left_object.append(object4)
                    # else:
                    #     left_matches = np.vstack((left_matches, np.delete(dets_to_track4, m4, axis=0)))
                    #
                    # if m5 == -1:
                    #     left_matches = np.vstack((left_matches, dets_to_track5))
                    #     left_trackers.append(tracker5)
                    #     left_object.append(object5)
                    # else:
                    #     left_matches = np.vstack((left_matches, np.delete(dets_to_track5, m5, axis=0)))

                    # TODO: Need to match the other unmatched detected objects
                    print('left matches: ', left_matches, 'len of left tracker: ', len(left_trackers))
                    # if (len(left_trackers) == 2):
                    #     cv2.imshow('results2', R_Img)
                    #     print('Debug!')
                    matched_bbox, matched_tracker, matched_object = re_associate_detections_to_trackers(left_matches, left_trackers, left_object, iou_threshold = 0.3)
                    print('len of matched tracker: ', len(matched_tracker))
                    for i in range(len(matched_bbox)):
                        m_temp, tracked_dets_temp = matched_tracker[i].update(matched_bbox[i].reshape((1, 5)))
                        matched_tracker[i].update_bbox_heatmap(tracked_dets_temp)
                        matched_object[i].update(tracked_dets_temp) # BUG here, the order is not consistent, SOLVED~

                    t3 = time_synchronized()
                    print(f'{s}Track Done. ({t3 - t2:.3f}s)')

                    # object.update(tracked_dets[0, :4] if len(tracked_dets) > 0 else np.empty((0, 5))) # TODO
                    # calculate the depth
                    # l_bbox = tracker2.cur_bbox[:4]
                    r_bbox = trackers[main_tool_id].cur_bbox[:4]
                    if trackers[main_tool_id].cur_bbox_flag is True:
                        # tool_depth = SteroTempleteDepthCal(leftImg=left, l_bbox=l_bbox, rightImg=right, resize_factor=0.8, R1=R1, Q=Q)
                        # print('------r_bbox width: ', r_bbox[2] - r_bbox[0], '------r_bbox height: ', r_bbox[3] - r_bbox[1])
                        scale = np.minimum(70.0/(r_bbox[2] - r_bbox[0]), 70.0/(r_bbox[3] - r_bbox[1]))
                        scale = np.minimum(scale, 1.0)
                        tool_depth = SteroTempleteDepthCal_right(rightImg=right, r_bbox=r_bbox, leftImg=left, resize_factor=scale, R2=R2, Q=Q)
                        # tool_depth = -1
                    else:
                        print('no valid depth estimation ! The current bbox is the previous one.')
                        tool_depth = -1

                    d1, filter_depth = main_tool_depth.update(tool_depth)
                    # filter_depth = -1
                    print('Filtered depth: ', filter_depth)
                    # print('Output from SORT:\n',tracked_dets,'\n')
                    t4 = time_synchronized()
                    print(f'{s}Depth Done. ({t4 - t3:.3f}s)')

                    for i in range(NUM_TOOLS_TO_TRACK):
                        if (save_img or view_img) and len(tracked_dets[i]) > 0:  # Add bbox to image + Add trajectory to image
                            score = 0 if ms[i] == -1 else 1
                            # label = f'{names[2*i+1]} {score:1d}' # since the toolXs,m the main is in 1, 3, 5, 7, 9
                            label = f'{names[2*i+2]} {score:1d}' # since the toolXs,m the main is in 2, 4, 6, 8, 10
                            plot_one_box(tracked_dets[i][:4], im0, label=label, color=colors[i], line_thickness=3)
                            objects[i].plot(im0, color=colors[i])
                            objects[i].plot_undistort(R_Img, color=colors[i], map=map2_1)

                    # if (save_img or view_img) and len(tracked_dets1) > 0:  # Add bbox to image + Add trajectory to image
                    #     score = 0 if m1 == -1 else 1
                    #     label = f'{names[int(1)]} {score:1d}'
                    #     plot_one_box(tracked_dets1[:4], im0, label=label, color=colors[int(0)], line_thickness=3)
                    #     object1.plot(im0, color=colors[int(0)])
                    #     object1.plot_undistort(R_Img, color=colors[int(1)], map=map2_1)
                    # if (save_img or view_img) and len(tracked_dets2) > 0:  # Add bbox to image + Add trajectory to image
                    #     score = 0 if m2 == -1 else 1
                    #     label = f'{names[int(3)]} {score:1d}'
                    #     plot_one_box(tracked_dets2[:4], im0, label=label, color=colors[int(1)], line_thickness=3)
                    #     object2.plot(im0, color=colors[int(1)])
                    #     object2.plot_undistort(R_Img, color=colors[int(3)], map=map2_1)
                    # if (save_img or view_img) and len(tracked_dets3) > 0:  # Add bbox to image + Add trajectory to image
                    #     score = 0 if m3 == -1 else 1
                    #     label = f'{names[int(5)]} {score:1d}'
                    #     plot_one_box(tracked_dets3[:4], im0, label=label, color=colors[int(2)], line_thickness=3)
                    #     object3.plot(im0, color=colors[int(2)])
                    #     object3.plot_undistort(R_Img, color=colors[int(3)], map=map2_1)
                    # if (save_img or view_img) and len(tracked_dets4) > 0:  # Add bbox to image + Add trajectory to image
                    #     score = 0 if m4 == -1 else 1
                    #     label = f'{names[int(7)]} {score:1d}'
                    #     plot_one_box(tracked_dets4[:4], im0, label=label, color=colors[int(3)], line_thickness=3)
                    #     object4.plot(im0, color=colors[int(3)])
                    #     object4.plot_undistort(R_Img, color=colors[int(3)], map=map2_1)
                    # if (save_img or view_img) and len(tracked_dets5) > 0:  # Add bbox to image + Add trajectory to image
                    #     score = 0 if m5 == -1 else 1
                    #     label = f'{names[int(9)]} {score:1d}'
                    #     plot_one_box(tracked_dets5[:4], im0, label=label, color=colors[int(4)], line_thickness=3)
                    #     object5.plot(im0, color=colors[int(4)])
                    #     object5.plot_undistort(R_Img, color=colors[int(3)], map=map2_1)

                    # reference control rect
                    # x1y1x2y2 = [120, 120, 520, 360]
                    bbox_w = 4
                    bbox_h = 3
                    interval_x = bbox_w * bbox_scale #3.2#3.5 # need < 8 parts
                    interval_y = bbox_h * bbox_scale #2.4#4.0
                    # inner_x = interval_x + 4.5
                    # inner_y = interval_y + 2.0
                    part = 40 # totally 16 parts, WIDTH:16*HEIGHT:12
                    interval_W = 16
                    interval_H = 12

                    x1y1x2y2 = [int(interval_x*part), int(interval_y*part), 640-int(interval_x*part), 480-int(interval_y*part)]
                    # x1y1x2y2_inner = [int(inner_x*part), int(inner_y*part), 640-int(inner_x*part), 480-int(inner_y*part)]

                    depth_minmax = [mid_depth - 25, mid_depth + 25] # middle: 75 -- [50, 100]
                    depth_minmax_inner = [mid_depth - 15, mid_depth + 15] # 75 -- [60, 90]
                    

                    # draw some notions to indicate whether it is located in the reference window
                    # the reference window need to be update
                    if is_out_rect is True:
                        # need to update the target bbox according to the reference position
                        # ADDing
                        # calculate the distance signal
                        # error: to a fixed bbox, need to get the max error to calculate the target position
                        try:
                            if len(objects[main_tool_id].undistorted_center) > 0 and trackers[main_tool_id].cur_bbox_flag is True:
                                x_e, y_e = RectErrorCal(objects[main_tool_id].undistorted_center[0,:], bbox=x1y1x2y2) # x1y1x2y2 is a fixed bbox
                                x_e = abs(x_e)
                                y_e = abs(y_e)
                                x_e_max = x_e if x_e > x_e_max else x_e_max
                                y_e_max = y_e if y_e > y_e_max else y_e_max
                                # fit the inner bbox automatically
                                Ex_min = 0.2 * part #* bbox_scale
                                Ex_max = (interval_W/4 - 1) * part #* bbox_scale
                                Ey_min = 0.2 * part #* bbox_scale
                                Ey_max = (interval_H/4 - 1) * part #* bbox_scale
                                # Linear
                                # scale_x = np.clip((x_e_max - Ex_min)/(Ex_max - Ex_min) * 0.9, 0.2, 0.9)
                                # scale_y = np.clip((y_e_max - Ey_min)/(Ey_max - Ey_min) * 0.9, 0.2, 0.9)
                                # Non-Linear
                                scale_x = np.clip((np.exp((x_e_max - Ex_min)*np.log(2)/(Ex_max - Ex_min)) - 1) * 0.9, 0.2, 0.9)
                                scale_y = np.clip((np.exp((y_e_max - Ey_min)*np.log(2)/(Ey_max - Ey_min)) - 1) * 0.9, 0.2, 0.9)

                                unified_scale = np.max((scale_x, scale_y))
                                if unified_scale > 0.35:
                                    whether_visualize_msg.data = True
                                print("DEBUG: unified_scale: ", unified_scale)
                                inner_x = interval_x + (interval_W/2 - interval_x) * unified_scale
                                inner_y = interval_y + (interval_H/2 - interval_y) * unified_scale

                                x1y1x2y2_inner = [int(inner_x*part), int(inner_y*part), 640-int(inner_x*part), 480-int(inner_y*part)]
                        except UnboundLocalError:
                            # if the x1y1x2y2_inner is not initialized
                            inner_x = interval_x + (interval_W/2 - interval_x) * 0.2
                            inner_y = interval_y + (interval_H/2 - interval_y) * 0.2
                            x1y1x2y2_inner = [int(inner_x*part), int(inner_y*part), 640-int(inner_x*part), 480-int(inner_y*part)]
                        dst_window = x1y1x2y2_inner
                        dst_depth = depth_minmax_inner
                    else: # start with the larger bbox as reference
                        x_e_max, y_e_max = 0, 0
                        whether_visualize_msg.data = False
                        # if the x1y1x2y2_inner is not initialized
                        inner_x = interval_x + (interval_W/2 - interval_x) * 0.2
                        inner_y = interval_y + (interval_H/2 - interval_y) * 0.2
                        x1y1x2y2_inner = [int(inner_x*part), int(inner_y*part), 640-int(inner_x*part), 480-int(inner_y*part)]
                        dst_window = x1y1x2y2
                        dst_depth = depth_minmax

                    # cv2.rectangle(im0, (x1y1x2y2[0], x1y1x2y2[1]), (x1y1x2y2[2], x1y1x2y2[3]), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    # cv2.rectangle(im0, (x1y1x2y2_inner[0], x1y1x2y2_inner[1]), (x1y1x2y2_inner[2], x1y1x2y2_inner[3]), (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                    cv2.rectangle(R_Img, (x1y1x2y2[0], x1y1x2y2[1]), (x1y1x2y2[2], x1y1x2y2[3]), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    cv2.rectangle(R_Img, (x1y1x2y2_inner[0], x1y1x2y2_inner[1]), (x1y1x2y2_inner[2], x1y1x2y2_inner[3]), (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                    if len(objects[main_tool_id].undistorted_center) > 0 and trackers[main_tool_id].cur_bbox_flag is True:
                        print("debug: ", objects[main_tool_id].undistorted_center, objects[main_tool_id].center)
                        if (objects[main_tool_id].undistorted_center[0, 0] > dst_window[0] and objects[main_tool_id].undistorted_center[0, 0] < dst_window[2]) \
                        and (objects[main_tool_id].undistorted_center[0, 1] > dst_window[1] and objects[main_tool_id].undistorted_center[0, 1] < dst_window[3]) \
                        and (filter_depth > dst_depth[0] and filter_depth < dst_depth[1]):
                            cv2.rectangle(im0, (0, 0), (640, 480), (0, 255, 0), 8) # already in the proper view
                            cv2.rectangle(R_Img, (0, 0), (640, 480), (0, 255, 0), 8) # already in the proper view
                            ctrl_mode = 'Zero'
                            x_error = y_error = z_error = 0
                            x_action = y_action = z_action = 0
                            is_out_rect = False # current is within reference bbox
                        else:
                            cv2.rectangle(im0, (0, 0), (640, 480), (0, 0, 255), 8)
                            # out of the proper view
                            if objects[main_tool_id].radius < 10:
                                objects[main_tool_id].ctrl_cnt += 1
                            else:
                                objects[main_tool_id].ctrl_cnt = 0

                            if objects[main_tool_id].ctrl_cnt > 30:
                                ctrl_mode = 'Fast'
                            else:
                                ctrl_mode = 'Slow'

                            # calculate the action command
                            x_error, y_error = RectErrorCal(objects[main_tool_id].undistorted_center[0,:], bbox=dst_window)
                            z_error, z_action = DepthActionCal(filter_depth, minmax=dst_depth)
                            x_action = x_error / 640
                            y_action = y_error / 480
                            is_out_rect = True # act as a flag: leave current the large bbox, we need to update the bbox

                    else:# no detection, need to keep static
                        ctrl_mode = 'Zero'
                        x_error = y_error = z_error = 0
                        x_action = y_action = z_action = 0
                        cv2.rectangle(im0, (0, 0), (640, 480), (0, 0, 255), 8)

                    # publish the control signals
                    x_action = np.clip(x_action * 10, a_min=-1, a_max=1)
                    y_action = np.clip(y_action * 10, a_min=-1, a_max=1)
                    z_action = np.clip(z_action, a_min=-1, a_max=1)
                    pred_action_msg = Vector3(-x_action, -y_action, -z_action)
                    pred_action_pub.publish(pred_action_msg)

                    # publish the white/yellow bbox info
                    height, width = R_Img.shape[:2]
                    white_bbox_msg.cx = (x1y1x2y2[0] + x1y1x2y2[2])/2/width
                    white_bbox_msg.cy = (x1y1x2y2[1] + x1y1x2y2[3])/2/height
                    white_bbox_msg.w = (x1y1x2y2[2] - x1y1x2y2[0])/width
                    white_bbox_msg.h = (x1y1x2y2[3] - x1y1x2y2[1])/height
                    whitebbox_refer_pub.publish(white_bbox_msg)
                    yellow_bbox_msg.cx = (x1y1x2y2_inner[0] + x1y1x2y2_inner[2])/2/width
                    yellow_bbox_msg.cy = (x1y1x2y2_inner[1] + x1y1x2y2_inner[3])/2/height
                    yellow_bbox_msg.w = (x1y1x2y2_inner[2] - x1y1x2y2_inner[0])/width
                    yellow_bbox_msg.h = (x1y1x2y2_inner[3] - x1y1x2y2_inner[1])/height
                    yellowbbox_refer_pub.publish(yellow_bbox_msg)
                    # publish whether to visulize bbox
                    whetherbbox_visualize_pub.publish(whether_visualize_msg)

                    # publish the tool depth info
                    tool_depth_msg = Float64(filter_depth / 1000) # convert mm into m
                    tool_depth_pub.publish(tool_depth_msg)

                    # publish the tool position info
                    pred_positions_msg = []
                    for i in range(NUM_TOOLS_TO_TRACK):
                        if len(objects[i].undistorted_center) > 0 and len(tracked_dets[i]) > 0:
                            pred_position_msg = Vector3(objects[i].undistorted_center[0, 0], objects[i].undistorted_center[0, 1], -1)
                        else:
                            pred_position_msg = Vector3(-1, -1, -1)
                        # pred_positions_msg.append(pred_position_msg)
                        pred_position_tools_pub[i].publish(pred_position_msg)


                    # if len(object1.undistorted_center) > 0 and len(tracked_dets1) > 0:
                    #     pred_position_aux_msg = Vector3(object1.undistorted_center[0, 0], object1.undistorted_center[0, 1], -1)
                    # else:
                    #     pred_position_aux_msg = Vector3(-1, -1, -1)
                    #
                    # if len(object2.undistorted_center) > 0 and len(tracked_dets2) > 0:
                    #     pred_position_main_msg = Vector3(object2.undistorted_center[0, 0], object2.undistorted_center[0, 1], filter_depth / 1000)
                    # else:
                    #     pred_position_main_msg = Vector3(-1, -1,filter_depth / 1000)
                    #
                    # pred_position_aux_pub.publish(pred_position_aux_msg)
                    # pred_position_main_pub.publish(pred_position_main_msg)

                    # if len(object2.center) > 0 and object2.radius < 10: # out of the bbox, need to select the movement mode: No movement / Slow / Fast, 30 is the hyperameters
                    #     object2.ctrl_cnt += 1
                    # elif len(object2.center) > 0 and object2.radius > 10: # out of the bbox
                    #     object2.ctrl_cnt = 0
                    #     ctrl_mode = 'Slow'

                    # time.sleep(0.5)
                    # Stream results
                    if view_img:
                        frame_count_ = f'Frame: {frame_count:d}'
                        tool_depth_ = f'Depth: {tool_depth:.2f}mm'
                        filter_tool_depth_ = f'Depth: {filter_depth:.2f}mm'
                        fps_count_ = f'FPS: {1/(t4 - t1):.1f}'
                        # traj_center_ = f'Center: {object2.radius:.1f}'
                        # ctrl_mode_ = 'Mode: ' + ctrl_mode
                        # x_error_ = f'dx: {x_error:.1f}'
                        # y_error_ = f'dy: {y_error:.1f}'
                        # z_error_ = f'dz: {z_error:.2f}'
                        # x_action_ = f'ax: {x_action:.2f}'
                        # y_action_ = f'ay: {y_action:.2f}'
                        # z_action_ = f'az: {z_action:.2f}'
                        cv2.putText(im0, frame_count_, (40, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        # cv2.putText(im0, tool_depth_, (260, 40), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        cv2.putText(im0, filter_tool_depth_, (260, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        cv2.putText(im0, fps_count_, (160, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        # cv2.putText(im0, traj_center_, (420, 40), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, ctrl_mode_, (540, 40), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)

                        # cv2.putText(im0, x_error_, (40, 60), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, y_error_, (160, 60), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, z_error_, (260, 60), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        #
                        # cv2.putText(im0, x_action_, (40, 80), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, y_action_, (160, 80), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, z_action_, (260, 80), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        # cv2.putText(im0, filter_tool_depth_, (320, 80), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)

                        cv2.putText(R_Img, frame_count_, (40, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        # cv2.putText(im0, tool_depth_, (260, 40), 0, 0.5, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        cv2.putText(R_Img, filter_tool_depth_, (260, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                        cv2.putText(R_Img, fps_count_, (160, 40), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)

                        cv2.imshow('results', im0)
                        cv2.imshow('R_Img', R_Img)
                        cv2.imshow('L_Img', L_Img)

                        cv2.imshow('R_Img_Dis', right)
                        cv2.imshow('L_Img_Dis', left)
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

        # except TypeError:
        #     print('no data from camera')
        #     continue
        # except AttributeError:
        #     print('no data from camera')
        #     continue
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
