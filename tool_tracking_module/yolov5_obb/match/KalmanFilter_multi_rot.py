import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
from make_cycle import make_circle


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

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    y[:, 4] = x[:, 4]
    # y[:, 5] = x[:, 5]
    return y

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    refer to: https://blog.51cto.com/u_15221047/2807354, dim_x: 7, dim_z: 4
    """
    count = 0

    def __init__(self):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=9, dim_z=5)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0]])
        # TODO
        self.kf.R[3:, 3:] *= 100.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities 1000
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.Initial_flag = False
        self.time_since_update = 0 # time interval from last updates
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0 # number of continuous detection
        self.age = 0

    def initialbbox(self, bbox):
        self.Initial_flag = True
        # self.kf.x[:4] = convert_bbox_to_z(bbox) # convert from x1y1x2y2 to xywh
        self.kf.x[:5] = bbox[:5].reshape((5, 1)) # bbox: already xywh+theta

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # self.kf.update(convert_bbox_to_z(bbox))
        self.kf.update(bbox[:5].reshape((5, 1)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0): # TO be clear
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append(convert_x_to_bbox(self.kf.x))
        self.history.append(self.kf.x[:5].reshape((1, 5)))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # return convert_x_to_bbox(self.kf.x)
        return self.kf.x[:5].reshape((1, 5))

class Tracker(object):
    def __init__(self, max_age=30, min_hits=20, iou_threshold=0.3, short_memory=20): #10
        """
        Parameters for Tracker
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = KalmanBoxTracker()
        self.frame_count = 0
        self.init_dect_count = 0
        self.init_min_hits = min_hits
        self.lambda1 = 0.5 # 0.0
        self.short_memory = short_memory
        self.bbox_memory = np.empty((self.short_memory, 4+1))  # store the bbox info
        self.bbox_index = 0
        h = 1080
        w = 1920
        self.bbox_area = np.zeros((h, w))
        self.cur_bbox = np.empty((0, 5+1))

    def update(self, dets=np.empty((0, 5+1))):
        """
        Parameters:
        'dets' - a numpy array of detection in the format [[x1, y1, x2, y2, theta, score], [x1,y1,x2,y2,theta,score],...]

        Ensure to call this method even frame has no detections. (pass np.empty((0, 6)))

        Returns a similar array, where the last column is object ID (replacing confidence score)

        NOTE: The number of objects returned may differ from the number of objects provided.
        """
        self.frame_count += 1
        if self.trackers.Initial_flag is False: # no initailization yet, keep waiting a detection
            if (len(dets) > 0 and np.min(dets[:, -1]) > 0.6): # have a detection 0.8
                self.init_dect_count += 1 # continuous detecting for a while
                if(self.init_dect_count > self.init_min_hits):
                    self.trackers.Initial_flag = True
                    # Need to choose the higher confidence ones as the initial one
                    high_conf_index = np.argmax(dets[:, -1])
                    dets_high_conf = dets[high_conf_index, :]
                    self.trackers.initialbbox(dets_high_conf)
                    return high_conf_index, dets_high_conf.reshape((1, -1))
                else:
                    return -1, np.empty((0, 5+1))
            else:
                # need to set some flag here
                self.init_dect_count = 0
                return -1, np.empty((0, 5+1))
        else: # already have a initial bbox, need to track this object
            pos = self.trackers.predict()[0] # the predicted pos
            pred_raw = [pos[0], pos[1], pos[2], pos[3], pos[4], 0] # format: [x, y, w, h, theta, conf]
            pred = np.array(pred_raw).reshape((1, -1))
            # Need to deal with several cases
            if (len(dets) > 0):
                # Have many detections, need to define a cost function based on confidence and IoU
                iou_matrix = iou_batch(xywh2xyxy(dets), xywh2xyxy(pred)) # TODO
                score_matrix = dets[:, -1].reshape(-1, 1)
                # assert the shape:
                assert iou_matrix.shape == score_matrix.shape
                cost_matrix = iou_matrix + self.lambda1 * score_matrix
                # if only only one detection has been found, we can use it as the observation
                high_conf_index = np.argmax(cost_matrix)
                obs = dets[high_conf_index, :]

                pred_bbox = xywh2xyxy(pred).reshape((-1,1)) # format: (6, 1) [x1, y1, x2, y2, theta, conf]
                obs_bbox = xywh2xyxy(obs.reshape((1,-1))).reshape((-1,1)) # format: (6, 1) [x1, y1, x2, y2, theta, conf]

                # observation indicator
                pred_z = convert_bbox_to_z(pred_bbox)
                obs_z = convert_bbox_to_z(obs_bbox)

                assert pred_z.shape == obs_z.shape
                dis = np.linalg.norm(pred_z[:2] - obs_z[:2])
                dis_x = np.abs(pred_z[0] - obs_z[0])
                dis_y = np.abs(pred_z[1] - obs_z[1])
                w_h_array = np.array([pred_bbox[2]-pred_bbox[0], pred_bbox[3]-pred_bbox[1], obs_bbox[2]-obs_bbox[0], obs_bbox[3]-obs_bbox[1]])
                # x_torrence = (w_h_array[0] + w_h_array[2]) / 2.0
                # y_torrence = (w_h_array[1] + w_h_array[3]) / 2.0
                x_torrence = w_h_array[0]
                y_torrence = w_h_array[1]
                max_torrence = np.max(w_h_array)
                # print('dis: ', dis, 'max_torrence: ', max_torrence)
                # TODO: Add the short memory here to have a boarder check
                # overlap_mem = np.sum(self.bbox_area[int(obs_bbox[1]):int(obs_bbox[3]), int(obs_bbox[0]):int(obs_bbox[2])] > 0.10) / self.bbox_area[int(obs_bbox[1]):int(obs_bbox[3]), int(obs_bbox[0]):int(obs_bbox[2])].size

                # rotated bbox for counting the short memory
                y, x = np.arange(int(obs_bbox[1]), int(obs_bbox[3])), np.arange(int(obs_bbox[0]), int(obs_bbox[2]))
                x_, y_ = np.meshgrid(x, y)
                y_mid, x_mid = (obs_bbox[1] + obs_bbox[3]) / 2, (obs_bbox[0] + obs_bbox[2]) / 2
                x_rot = np.cos(-obs_bbox[-1]) * (x_.ravel() - x_mid) - np.sin(-obs_bbox[-1]) * (y_.ravel() - y_mid) + x_mid
                x_rot = np.clip(x_rot, 0, 1919)
                y_rot = np.sin(-obs_bbox[-1]) * (x_.ravel() - x_mid) + np.cos(-obs_bbox[-1]) * (y_.ravel() - y_mid) + y_mid
                y_rot = np.clip(y_rot, 0, 1079)
                overlap_mem = np.sum(self.bbox_area[y_rot.astype(np.int), x_rot.astype(np.int)] > 0.10) / y_rot.size

                print('overlap mem: ', overlap_mem, 'prob: ', obs[-1])
                print('iou: ', iou_matrix[high_conf_index], 'dis_x: ', dis_x, 'dis_y:', dis_y)
                print('x_torrence: ', x_torrence, 'y_torrence: ', y_torrence)

                # Deal with below cases: High movement (cross large distance: need to fit some conditions (previous position+high confidence), the movement is approved)
                if (iou_matrix[high_conf_index] < 1e-3 and (dis_x > 1.0 * x_torrence or dis_y > 1.0 * y_torrence)):
                    if (overlap_mem < 0.2 and obs[-1] < 0.95) or (overlap_mem > 0.2  and overlap_mem < 0.6 and obs[-1] < 0.8) or (overlap_mem > 0.6 and obs[-1] < 0.6): # mem: 0.7
                        print('need track twice~')
                        if self.trackers.time_since_update >= self.max_age:
                            # if no detections in serveral frames, we need to re-initialazation
                            self.init_dect_count = 0
                            self.trackers.Initial_flag = False
                        return -1, np.array(pred)

                if (iou_matrix[high_conf_index] < 1e-3 and (dis_x > 5 * x_torrence or dis_y > 5 * y_torrence)):
                    if (overlap_mem < 0.9 and obs[-1] < 0.95): # mem: 0.7
                        print('need track twice~')
                        if self.trackers.time_since_update >= self.max_age:
                            # if no detections in serveral frames, we need to re-initialazation
                            self.init_dect_count = 0
                            self.trackers.Initial_flag = False
                        return -1, np.array(pred)

                # use the kalman filter to get the estimated value
                # deal with the cases with ambiguity
                # case1: 0 degree and 180 degree: [-90, 90)
                if obs[2] / obs[3] > 1.5: # 1.2
                    if np.abs(obs[4] + np.pi/2) < 10/180*np.pi: # close to -90, we get the close to 90 candidate
                        obs_candidate = np.array([obs[0], obs[1], obs[2], obs[3], obs[4] + np.pi, obs[5]])
                    elif np.abs(obs[4] - np.pi/2) < 10/180*np.pi: # close to 90, we get the close to -90 candidate
                        obs_candidate = np.array([obs[0], obs[1], obs[2], obs[3], obs[4] - np.pi, obs[5]])
                    else:
                        obs_candidate = np.array([obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]])
                else:
                    # case2: if the width is very close to height
                    # theta with 90+theta or 90-theta
                    if obs[4] > 0:
                        obs_candidate = np.array([obs[0], obs[1], obs[2], obs[3], obs[4] - np.pi/2, obs[5]])
                    else:
                        obs_candidate = np.array([obs[0], obs[1], obs[2], obs[3], obs[4] + np.pi/2, obs[5]])

                # select the near canditate as the update parameter
                obs_select = obs if np.abs(self.cur_bbox[0, 4] - obs[4]) < np.abs(self.cur_bbox[0, 4] - obs_candidate[4]) else obs_candidate

                # self.trackers.update(obs)
                self.trackers.update(obs_select)
                return high_conf_index, self.trackers.get_state() # shape: (1, 5)
            else:
                # Lose det for a while
                print('------------------------------------------------------------------------------------------------')
                if self.trackers.time_since_update >= self.max_age:
                    # if no detections in serveral frames, we need to re-initialazation
                    self.init_dect_count = 0
                    self.trackers.Initial_flag = False
                return -1, np.array(pred) # shape: (6, 1)

    def update_bbox_heatmap(self, bbox):
        """
        Pamameters:
        bbox: the current predicted bbox, need to store it
        Return: Based it to generate a new heatmap which contains the moving area in past steps
        """
        if len(bbox) > 0: # have a valid detection
            # self.bbox_index = self.bbox_index % self.short_memory
            # print(bbox)
            # print(bbox.shape)
            # self.bbox_memory[self.bbox_index, :] = bbox[:4]
            # self.bbox_z = convert_bbox_to_z(bbox[:4])
            # # self.pos[self.index, :] = self.bbox_z[:2].reshape((1, 2))
            # self.bbox_index += 1

            bbox = xywh2xyxy(bbox.reshape((1, -1))).reshape((-1, 1))
            self.cur_bbox = bbox.reshape((1, -1)) # cur_box should be shape (1, 6)

            # update the bbox_area
            self.bbox_area *= 0.99

            # calculate the rotated bbox area
            y, x = np.arange(int(bbox[1]), int(bbox[3])), np.arange(int(bbox[0]), int(bbox[2]))
            x_, y_ = np.meshgrid(x, y)
            y_mid, x_mid = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
            # np.cos(-bbox[-1]): the -1 here because the xy in image plane is different with the Cartesian coordinate system
            x_rot = np.cos(-bbox[-1]) * (x_.ravel() - x_mid) - np.sin(-bbox[-1]) * (y_.ravel() - y_mid) + x_mid
            x_rot = np.clip(x_rot, 0, 1919)
            y_rot = np.sin(-bbox[-1]) * (x_.ravel() - x_mid) + np.cos(-bbox[-1]) * (y_.ravel() - y_mid) + y_mid
            y_rot = np.clip(y_rot, 0, 1079)
            self.bbox_area[y_rot.astype(np.int), x_rot.astype(np.int)] += 1
            # self.bbox_area[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] += 1

class Object(object):
    def __init__(self, num=20):
        """
        Parameters for the object
        """
        self.num = num
        self.pos = np.zeros((self.num, 2)) # store the x, y coordinates
        self.bbox = np.zeros((self.num, 5)) # store the bbox info + theta
        self.index = 0
        self.count = 0

    def update(self, bbox):
        """
        Parameters:
            bbox: the lateset bbox info for the object
            return a full bbox info with a period of time and the current index
        """
        if len(bbox) > 0: # have a valid detection
            self.index = self.index % self.num
            # print(bbox)
            # print(bbox.shape)
            bbox = bbox.reshape((-1,)) # get (6 or 5, ) shape array
            self.bbox[self.index, :] = bbox[:5]
            self.bbox_z = bbox[:5].reshape((5, 1)) # convert_bbox_to_z(bbox[:4])
            self.pos[self.index, :] = self.bbox_z[:2].reshape((1, 2))
            self.index += 1
            self.count += 1

    def plot(self, img, color):
        """
        Parameters:
            img: the image where the lines will plot on.
        """
        if self.count >= 2:
            start_pt_index = self.index - 1
            # print('pos: ', self.pos)
            pt_num = self.count if self.count < len(self.pos) else len(self.pos)
            cv2.circle(img, (int(self.pos[start_pt_index, 0]), int(self.pos[start_pt_index, 1])), 4, color, 2)
            for index in range(pt_num - 1):
                start_pt = (int(self.pos[start_pt_index - index, 0]), int(self.pos[start_pt_index - index, 1]))
                end_pt = (int(self.pos[start_pt_index - index - 1, 0]), int(self.pos[start_pt_index - index - 1, 1]))
                # print('start_pt: ', start_pt, 'end_pt: ', end_pt)
                cv2.line(img, start_pt, end_pt, color, 2)

            points = self.pos[np.any(self.pos, axis=1)]
            output = make_circle(points)
            # center: output[:2], radius: output[2]
            cv2.circle(img, (int(output[0]), int(output[1])), int(output[2]), color, 2)
