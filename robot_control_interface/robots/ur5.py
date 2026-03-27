from backend.utils_robotics import Robotics, skew
import numpy as np

# basis parameters
d1 =  0.089159
a2 = -0.42500
a3 = -0.39225
d4 =  0.10915
d5 =  0.09465
d6 =  0.0823

# le2r = 0.290 #0.1850 # 0.2175 # The length between end-effector and RCM
lr2c = 0.040 # the length between the camera and RCM
ls2c = 0.30 # The laparoscope length

# Joint params
JOINT_SIZE = 6
JOINT_TYPE = np.array([0, 0, 0, 0, 0, 0])

# DH parameters
A      = np.array([0, 0, a2, a3, 0, 0])
ALPHA  = np.array([0, 90, 0, 0, 90, -90]) * np.pi/180
D_BASE = np.array([d1, 0, 0, d4, d5, d6])
Q_BASE = np.array([0, 0, 0, 0, 0, 0]) * np.pi/180 # only as initialization

# other parameters
bPr   = np.zeros((3, 1))
bRo   = np.zeros((3, 3))
bR0   = np.zeros((3, 3))
rJd   = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
rJe   = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
iRb   = np.zeros((3, 3))

# calibration parameters
# eTc = np.array([[-0.091098,	0.995769,	0.012086,	0.000091],
#                 [-0.038201,	0.008633,	-0.999233,	-0.264405],
#                 [-0.995109,	-0.091490,	0.037253,	0.017124],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# eTc = np.array([[0.046771,	0.998721,	0.019183,	-0.000686],
#                 [-0.039881,	0.017321,	-0.999054,	-0.227551],
#                 [-0.998109,	-0.047492,	0.039020,	0.016317],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# eTc = np.array([[-0.046126,	0.998785,	0.017367,	0.000215],
#                 [-0.038307,	0.015604,	-0.999144,	-0.230191],
#                 [-0.998201,	-0.046752,	0.037541,	0.017498],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# eTc = np.array([[-0.118324,	0.992975,	-0.000543,	-0.001646],
#                 [-0.046285,	-0.006062,	-0.998910,	-0.318658],
#                 [-0.991896,	-0.118170,	0.046678,	0.015604],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# eTc = np.array([[-0.110384,	-0.576939,	-0.809294,	-0.242559],
#                 [-0.046336,	-0.810397,	0.584046,	-0.138566],
#                 [-0.992808,	0.101969,	0.062722,	0.015338],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# curi
# eTc = np.array([[-0.002554,	-0.521297,	-0.853371,	-0.235067],
#                 [0.054779,	-0.852166,	0.520397,	-0.147810],
#                 [-0.998495,	-0.045417,	0.030733,	0.023136],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# cadaver2
# eTc = np.array([[-0.010087,	-0.506835,	-0.861984,	-0.269712],
#                 [-0.073432,	-0.859325,	0.506131,	-0.130470],
#                 [-0.997249,	0.068402,	-0.028550,	0.013702],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])
# cadaver3
eTc = np.array([[0.002131,	-0.525620,	-0.850717,	-0.275094],
                [0.105760,	-0.845829,	0.522865,	-0.128600],
                [-0.994389,	-0.091086,	0.053787,	0.005705],
                [0.000000,	0.000000,	0.000000,	1.000000]])

# new-holder
# eTc = np.array([[0.002415,	-0.514556,	-0.857453,	-0.240803],
#                 [0.005251,	-0.857438,	0.514561,	-0.164365],
#                 [-0.999983,	-0.005745,	0.000632,	0.006275],
#                 [0.000000,	0.000000,	0.000000,	1.000000]])

cTr = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -lr2c],
                [0.0, 0.0, 0.0, 1.0]])

cTs = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -ls2c],
                [0.0, 0.0, 0.0, 1.0]])

class UR5(object):
    """
    A UR5 robot
    """
    STEP_LIMITATION = 0.020 # 0.005
    K_IMAGE = 0.55

    def __init__(self, init_joint_positions):
        # Robotics calculation
        self._robotics = Robotics(JOINT_SIZE, JOINT_TYPE, A, ALPHA, D_BASE, Q_BASE)
        # Orginal rotation
        init_bTe = self._robotics.MFK(init_joint_positions)
        # previous case: defined for RCM
        # init_eTr = eTc.copy(); init_eTr[1, 3] = np.sign(eTc[1, 3]) * le2r; init_bTr = np.dot(init_bTe, init_eTr) # rcm point of the shaft
        # self.init_eTs = eTc.copy(); self.init_eTs[1, 3] = 0; self.init_bTs = np.dot(init_bTe, self.init_eTs) # start point of the shaft
        # current case: defined for RCM
        init_eTr = np.dot(eTc, cTr); init_bTr = np.dot(init_bTe, init_eTr) # rcm point of the shaft
        self.init_eTs = np.dot(eTc, cTs); self.init_bTs = np.dot(init_bTe, self.init_eTs) # start point of the shaft
        # init_eTr = self._robotics.MDH(A[JOINT_SIZE], ALPHA[JOINT_SIZE], d8, 0); init_bTr = np.dot(init_bTe, init_eTr)
        # init_eTt = self._robotics.MDH(A[JOINT_SIZE], ALPHA[JOINT_SIZE], d7, 0); init_bTt = np.dot(init_bTe, init_eTt)
        self.init_bTc = np.dot(init_bTe, eTc) # eTc: need to be calibrate in advance
        # init_bTc[0:3, 3] = init_bTt[0:3, 3] # to be confirmed ???
        # bRr = bTr[0: 3, 0: 3]
        # wRr = np.dot(self._wRb, bRr)
        # wRc = np.dot(wRr, self._rRc)
        self.bPc_init = self.init_bTc[0:3, 3]
        self.bRc_init = self.init_bTc[0:3, 0:3]
        self.bPr_init = init_bTr[0:3, 3]
        self.close_constraint = False
        self.out_constraint = False
        self.cVc_norm = np.array([0.0, 0.0, 0.0])

    def _init_global(self, init_joint_positions):
        """
        update the global camera position
        """
        bTe = self._robotics.MFK(init_joint_positions)
        self.init_bTc = np.dot(bTe, eTc)
        self.bPc_init = self.init_bTc[0:3, 3]
        self.bRc_init = self.init_bTc[0:3, 0:3]

    def cVc_to_deltaq(self, cVc: np.ndarray, homo_delta: np.ndarray, joint_positions: np.ndarray) -> np.ndarray:
        """
        Transfer the cVc into the joint velocity
        :param cVc: target velocity in camera frame
        :param joint_positions: current joint information
        :return: delta q
        """
        cVc = cVc.reshape((3, 1))
        self._homo_delta = homo_delta

        # set the boundary of the position command
        if np.linalg.norm(cVc) > self.STEP_LIMITATION:
            cVc = cVc / np.linalg.norm(cVc) * self.STEP_LIMITATION
        self.cVc_norm = cVc

        # Forward kinematics
        bTe = self._robotics.MFK(joint_positions)
        dis = np.linalg.norm(bTe[0:3, 3] - self.bPr_init)
        # print('distance: ', dis)
        # print('bTe: ', bTe[0:3, 3])
        # print('bPr: ', self.bPr_init)
        # eTr = eTc.copy(); eTr[1, 3] = np.sign(eTc[1, 3]) * le2r; bTr = np.dot(bTe, eTr)
        eTr = np.dot(eTc, cTr); bTr = np.dot(bTe, eTr)
        bTc = np.dot(bTe, eTc)
        # print('bTc: ', bTc)
        # eTr = self._robotics.MDH(A[JOINT_SIZE], ALPHA[JOINT_SIZE], d8, 0); bTr = np.dot(bTe, eTr)
        # eTt = self._robotics.MDH(A[JOINT_SIZE], ALPHA[JOINT_SIZE], d7, 0); bTt = np.dot(bTe, eTt)
        # bTc = np.dot(DH2UR(bTe), eTc) # eTc: need to be calibrate in advance
        # bTc[0:3, 3] = bTt[0:3, 3] # to be confirmed???
        bRr = bTr[0: 3, 0: 3]
        # wRr = np.dot(self._wRb, bRr)
        # wRc = np.dot(wRr, self._rRc)
        bRc = bTc[0:3, 0:3]

        # RCM feedback
        bTs = np.dot(bTe, self.init_eTs)
        lerror = np.linalg.norm(bTs[0:3, 3] - bTc[0:3, 3])
        # print('lerror', lerror)
        # print('r -> e', bTe[0:3, 3] - self.bPr_init)
        # print('r -> c', bTc[0:3, 3] - self.bPr_init)
        # print('length of r -> e', np.linalg.norm(bTe[0:3, 3] - self.bPr_init))
        # print('length of r -> c', np.linalg.norm(bTc[0:3, 3] - self.bPr_init))
        error = np.linalg.norm(np.cross(bTs[0:3, 3] - self.bPr_init, bTc[0:3, 3] - self.bPr_init)) / lerror
        derror = np.cross(np.cross(bTs[0:3, 3] - self.bPr_init, bTc[0:3, 3] - self.bPr_init), bTs[0:3, 3] - bTc[0:3, 3])
        print('lerror: ', lerror)
        print('direction_error_pre: ', derror, np.linalg.norm(derror))
        # if np.linalg.norm(derror) < 1e-4:
        #     derror = derror / np.linalg.norm(derror) * error
        # else:
        #     derror = np.zeros_like(derror)

        if np.linalg.norm(derror) < 1e-6:
            derror = np.zeros_like(derror)
        else:
            derror = derror / np.linalg.norm(derror) * error

        print('RCM error: ', error)
        print('direction_error_after: ', derror, np.linalg.norm(derror))
        print('RCM position: ', self.bPr_init)
        dv = np.dot(bRr.T, derror)
        kk = 5 # 0.3 * 50
        vx = -kk*dv[0]
        vy = -kk*dv[1]
        # vx = 0.0; vy = 0.0

        # Rotation
        R1 = self.bRc_init
        R2 = bRc
        xx = R1[0, 0] * R2[1, 0] - R1[1, 0] * R2[0, 0] + R1[0, 1] * R2[1, 1] - R1[1, 1] * R2[0, 1]
        yy = R1[0, 0] * R2[1, 1] - R1[1, 0] * R2[0, 1] - R1[0, 1] * R2[1, 0] + R1[1, 1] * R2[0, 0]
        dz = np.arctan(xx / yy)
        k1 = 10 #20.0
        k2 = 0.1
        wz = - dz * k1 * np.exp(-k2 * np.linalg.norm(self._homo_delta))

        # Pseudo Solution
        # Jd = np.dot(self._rRc,
        #             np.array([[0, 0, self._tip_offset, 0],
        #                       [0, -self._tip_offset, 0, 0],
        #                       [1, 0, 0, 0]]))
        # Je = np.dot(self._rRc,
        #             np.array([[0, 1, 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]]))
        #
        # rVr4 = np.dot(np.linalg.pinv(Jd), cVc) \
        #        + np.dot(np.dot((np.eye(4) - np.dot(np.linalg.pinv(Jd), Jd)), np.linalg.pinv(Je)),
        #                 np.array([[0], [0], [self._wz]]))
        # rVr = np.zeros((6, 1))
        # rVr[2: 6] = rVr4[0: 4]
        rTc = np.dot(np.linalg.pinv(bTr), bTc)
        cRr = rTc[0:3, 0:3].T
        rtc = rTc[0:3, 3]
        # print('rtc', rtc)
        rJd[0:3, 1:4] = skew(-rtc)
        cJd = np.dot(cRr, rJd)
        cJe = np.dot(cRr, rJe)
        pinv_cJd = np.linalg.pinv(cJd)
        pinv_cJe = np.linalg.pinv(cJe)
        # print('cJd: ', cJd)
        # print('cJe: ', cJe)
        # print('rvr_part1: ', np.dot(pinv_cJd, cVc))
        # print('wz: ', wz, - dz * k1)
        # print('rvr_part2_1: ', np.dot(pinv_cJe, np.array([[0], [0], [wz]])))
        rVr4 = np.dot(pinv_cJd, cVc) + np.dot(np.eye(4) - np.dot(pinv_cJd, cJd), np.dot(pinv_cJe, np.array([[0], [0], [wz]])))
        rVr = np.array([[vx], [vy], [rVr4[0, 0]], [rVr4[1, 0]], [rVr4[2, 0]], [rVr4[3, 0]]])

        rTe = np.linalg.pinv(eTr)
        rte = rTe[0:3, 3]
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = bRr
        Q[0:3, 3:6] = np.dot(bRr, skew(-rte))
        Q[3:6, 3:6] = bRr
        bVe = np.dot(Q, rVr)
        # print('rvr: ', rVr)
        # print('bve: ', bVe)
        if np.linalg.norm(bVe) < 1e-3:
            bVe = np.zeros_like(bVe)
        # print('velocity norm bVe: ', np.linalg.norm(bVe))
        # print('velocity bVe: ', bVe)
        # print('shape bVe: ', bVe.shape)
        # print('rvr4: ', rVr4)
        # print('vx: ', vx)
        # print('vy: ', vy)

        # Compute the Jocob matrix
        bJe = self._robotics.MDK(joint_positions)
        delta_q = np.dot(np.linalg.pinv(bJe), bVe)
        # Check whether it encounter the singularity
        J_ret = np.linalg.det(bJe)
        # print('***************** J_ret: ', J_ret)
        # print('***************** command delta_q: ', delta_q)
        # print('***************** command delta_q_max: ', np.max(delta_q), delta_q.shape)

        # eigen_value, eigen_vector = np.linalg.eig(bJe)
        # eigen_value_abs = np.abs(eigen_value)
        # print('***************** max/min eigen abs value: {}/{}, ratio: {}'.format(np.max(eigen_value_abs), np.min(eigen_value_abs), np.max(eigen_value_abs)/np.min(eigen_value_abs)))
        # calculate actual ecm joint angle
        # self.Joint_q += delta_q * self.dt
        # print(delta_q.shape)

        return delta_q

    def bVc_to_cVc(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        get the cVc to return to the initial position: self.init_bTc[0:3, 3]
        :param bVc:
        :param cVc:
        :return:
        """
        # Forward kinematics
        bTe = self._robotics.MFK(joint_positions)
        bTc = np.dot(bTe, eTc)
        # bec
        # bPc = bTc[0:3, 3]
        bVc = self.init_bTc[0:3, 3] - bTc[0:3, 3]
        print('bVc: ', bVc)
        print('len of bVc: ', np.linalg.norm(bVc))
        # transfer into cVc
        cTb = np.linalg.pinv(bTc)
        cVc = np.dot(cTb[0:3, 0:3], bVc)
        return cVc

    def get_bTc(self, joint_positions: np.ndarray):
        """
        :param joint_positions:
        :param flag:
        :return: the bTc position
        """
        bTe = self._robotics.MFK(joint_positions)
        bTc = np.dot(bTe, eTc)
        return bTc

    def rVc_to_cVc(self, joint_positions: np.ndarray, flag: bool, dir, workspace: float):
        """
        :param joint_positions:
        :param flag:
        :return:
        """
        bTe = self._robotics.MFK(joint_positions)
        bTc = np.dot(bTe, eTc)
        if flag is True:
            # self.target_rPc = rTc[0:3, 3].reshape((3, 1)) - 0.01*rTc[0:3, 2].reshape((3, 1))
            # self.target_bPc = np.dot(bTr[0:3, 0:3], self.target_rPc)
            self.target_bPc = bTc[0:3, 3] + dir * 0.005 * bTc[0:3, 2]
            # Add the constraint to forbit move out of RCM constraint 0428-2022
            vector_r2c = self.target_bPc - self.bPr_init
            direction = np.dot(bTc[0:3, 2], vector_r2c)
            self.out_constraint = direction < 0
            # Add another constraint to forbit move close
            self.close_constraint = False if workspace < 0 else np.linalg.norm(vector_r2c) > workspace
            if self.out_constraint or self.close_constraint: # the target position is out of RCM
                self.target_bPc = bTc[0:3, 3]
            flag = False
        # print('target_rPc:', self.target_rPc)
        print('target_bPc:', self.target_bPc)
        # rVc = self.target_rPc - rTc[0:3, 3].reshape((3, 1))
        bVc = self.target_bPc - bTc[0:3, 3]
        # transfer into cVc
        # cVc = np.dot(cTr[0:3,0:3], rVc)
        cTb = np.linalg.pinv(bTc)
        cVc = np.dot(cTb[0:3,0:3], bVc)
        return cVc, flag


def DH2UR(bTe: np.ndarray):
    return np.array([[-bTe[0,2], -bTe[0,0], -bTe[0,1], -bTe[0,3]],
                     [-bTe[1,2], -bTe[1,0], -bTe[1,1], -bTe[1,3]],
                     [+bTe[2,2], +bTe[2,0], +bTe[2,1], +bTe[2,3]],
                     [        0,         0,         0,         1]])
