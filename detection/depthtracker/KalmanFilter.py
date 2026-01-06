import numpy as np
from filterpy.kalman import KalmanFilter
import cv2


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
    """
    count = 0

    def __init__(self):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 100.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities 1000
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

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
        self.kf.x[:4] = convert_bbox_to_z(bbox)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class KalmanDepthTracker(object):

    def __init__(self, min_hits=20):
        """
        Initialises a tracker using initial depth.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.F = np.array(
            [[1.,1.],
             [0.,1.]])  # initial state (location and velocity)
        self.kf.H = np.array(
            [[1.,0.]])  # state transition matrix

        self.kf.R = 10.     # state uncertainty
        self.kf.P *= 1000.  # give high uncertainty to the unobservable initial velocities 1000
        self.kf.Q *= 0.01

        # define the depth interval
        self.min_depth, self.max_depth = 10.0, 300.0

        # some variables
        self.init_dect_count = 0
        self.init_min_hits = min_hits

        def initdepth(self, depth):
            if self.isvalid(depth) and self.Initial_flag is False:
                self.init_dect_count += 1
                if self.init_dect_count > self.init_min_hits:
                    self.Initial_flag = True
                    self.kf.x[:2] = np.array([[depth], [0]])
                else:
                    self.kf.x[:2] = np.empty((2,0))

        def isvalid(self, depth):
            if depth > self.min_depth and depth < self.max_depth:
                return True
            else:
                return False

        def update(self, depth):
            """
            initilization
            generation
            """
            if self.Initial_flag is False: # no initailization yet, keep waiting a detection
                if self.isvalid(depth):
                    self.init_dect_count += 1
                if self.init_dect_count > self.init_min_hits:
                    self.Initial_flag = True
                    self.kf.x[:2] = np.array([[depth], [0]])
                    return True, self.kf.x[:2]
                else:
                    # self.kf.x[:2] = np.empty((2,0))
                    return False, np.empty((2,0))
            else: # already initialized
                pred = self.predict()[0]
                if self.isvalid(depth):
                    self.time_since_update = 0
                    self.kf.update(depth)
                    return True, self.get_state()[0]
                else:
                    if self.time_since_update >= self.max_age:
                        # if no detections in serveral frames, we need to re-initialazation
                        self.trackers.Initial_flag = False
                    return False, pred

        def predict(self):
            """
            Advances the state vector and returns the predicted depth estimate.
            """
            self.kf.predict()
            self.age += 1
            if (self.time_since_update > 0):
                self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(self.kf.x)
            return self.history[-1]

    def get_state(self):
        """
        Returns the current depth estimate.
        """
        return self.kf.x


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
        self.lambda1 = 0.0
        self.short_memory = short_memory
        self.bbox_memory = np.empty((self.short_memory, 4))  # store the bbox info
        self.bbox_index = 0
        h = 1080
        w = 1920
        self.bbox_area = np.zeros((h, w))
        self.cur_bbox = np.empty((0, 5))

    def update(self, dets=np.empty((0, 5))):
        """
        Parameters:
        'dets' - a numpy array of detection in the format [[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]

        Ensure to call this method even frame has no detections. (pass np.empty((0,5)))

        Returns a similar array, where the last column is object ID (replacing confidence score)

        NOTE: The number of objects returned may differ from the number of objects provided.
        """
        self.frame_count += 1
        if self.trackers.Initial_flag is False: # no initailization yet, keep waiting a detection
            if (len(dets) > 0 and np.min(dets[:, -1]) > 0.8): # have a detection
                self.init_dect_count += 1 # continuous detecting for a while
                if(self.init_dect_count > self.init_min_hits):
                    self.trackers.Initial_flag = True
                    # Need to choose the higher confidence ones as the initial one
                    high_conf_index = np.argmax(dets[:, -1])
                    dets_high_conf = dets[high_conf_index, :]
                    self.trackers.initialbbox(dets_high_conf)
                    return high_conf_index, dets_high_conf
                else:
                    return -1, np.empty((0, 5))
            else:
                # need to set some flag here
                self.init_dect_count = 0
                return -1, np.empty((0, 5))
        else: # already have a initial bbox, need to track this object
            pos = self.trackers.predict()[0] # the predicted pos
            pred = [pos[0], pos[1], pos[2], pos[3], 0]
            # Need to deal with several cases
            if (len(dets) > 0):
                # Have many detections, need to define a cost function based on confidence and IoU
                iou_matrix = iou_batch(dets, pred)
                score_matrix = dets[:, -1].reshape(-1, 1)
                # assert the shape:
                assert iou_matrix.shape == score_matrix.shape
                cost_matrix = iou_matrix + self.lambda1 * score_matrix
                # if only only one detection has been found, we can use it as the observation
                high_conf_index = np.argmax(cost_matrix)
                obs = dets[high_conf_index, :]

                # observation indicator
                pred_z = convert_bbox_to_z(pred)
                obs_z = convert_bbox_to_z(obs)
                assert pred_z.shape == obs_z.shape
                dis = np.linalg.norm(pred_z[:2] - obs_z[:2])
                w_h_array = np.array([pred[2]-pred[0], pred[3]-pred[1], obs[2]-obs[0], obs[3]-obs[1]])
                max_torrence = np.max(w_h_array)
                # print('dis: ', dis, 'max_torrence: ', max_torrence)
                # TODO: Add the short memory here to have a boarder check
                overlap_mem = np.sum(self.bbox_area[int(obs[1]):int(obs[3]), int(obs[0]):int(obs[2])] > 0.2) / self.bbox_area[int(obs[1]):int(obs[3]), int(obs[0]):int(obs[2])].size
                print('overlap mem: ', overlap_mem)
                if iou_matrix[high_conf_index] < 1e-3 and dis > 0.3 * max_torrence and overlap_mem < 0.2 and obs[-1] < 0.8:
                    if self.trackers.time_since_update >= self.max_age:
                        # if no detections in serveral frames, we need to re-initialazation
                        self.trackers.Initial_flag = False
                    return -1, np.array(pred)

                # use the kalman filter to get the estimated value
                self.trackers.update(obs)
                return high_conf_index, self.trackers.get_state()[0]
            else:
                # Lose det for a while
                print('------------------------------------------------------------------------------------------------')
                if self.trackers.time_since_update >= self.max_age:
                    # if no detections in serveral frames, we need to re-initialazation
                    self.trackers.Initial_flag = False
                return -1, np.array(pred)

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
            self.cur_bbox = bbox

            # update the bbox_area
            self.bbox_area *= 0.99
            self.bbox_area[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] += 1

class Object(object):
    def __init__(self, num=20):
        """
        Parameters for the object
        """
        self.num = num
        self.pos = np.zeros((self.num, 2)) # store the x, y coordinates
        self.bbox = np.zeros((self.num, 4)) # store the bbox info
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
            self.bbox[self.index, :] = bbox[:4]
            self.bbox_z = convert_bbox_to_z(bbox[:4])
            self.pos[self.index, :] = self.bbox_z[:2].reshape((1, 2))
            self.index += 1
            self.count += 1

    def plot(self, img):
        """
        Parameters:
            img: the image where the lines will plot on.
        """
        if self.count >= 2:
            start_pt_index = self.index - 1
            # print('pos: ', self.pos)
            pt_num = self.count if self.count < len(self.pos) else len(self.pos)
            cv2.circle(img, (int(self.pos[start_pt_index, 0]), int(self.pos[start_pt_index, 1])), 4, (255, 255, 0), 4)
            for index in range(pt_num - 1):
                start_pt = (int(self.pos[start_pt_index - index, 0]), int(self.pos[start_pt_index - index, 1]))
                end_pt = (int(self.pos[start_pt_index - index - 1, 0]), int(self.pos[start_pt_index - index - 1, 1]))
                # print('start_pt: ', start_pt, 'end_pt: ', end_pt)
                cv2.line(img, start_pt, end_pt, (255, 0, 0), 4)







