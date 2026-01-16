from typing import List
import numpy as np

class Robotics(object):
    """
    CUHK T Stone Robotics Institute
    Robotics calculation
    """

    def __init__(self, joint_size: int, joint_type,
                 a: np.ndarray, alpha: np.ndarray, d: np.ndarray, theta: np.ndarray):
        self.JOINT_SIZE = joint_size
        self.A = a
        self.ALPHA = alpha
        self.D = d
        self.THETA = theta
        if len(joint_type) == 0:
            self.joint_type = np.zeros((joint_size))
        else:
            self.joint_type = joint_type

    # other functions
    def RotX(self, theta: float):
        ans = np.array([[1,              0,              0, 0],
                        [0, +np.cos(theta), -np.sin(theta), 0],
                        [0, +np.sin(theta), +np.cos(theta), 0],
                        [0,              0,              0, 1]])
        return ans

    def RotY(self, theta: float):
        ans = np.array([[+np.cos(theta), 0, +np.sin(theta), 0],
                        [             0, 1,              0, 0],
                        [-np.sin(theta), 0, +np.cos(theta), 0],
                        [             0, 0,              0, 1]])
        return ans

    def RotZ(self, theta: float):
        ans = np.array([[+np.cos(theta), -np.sin(theta), 0, 0],
                        [+np.sin(theta), +np.cos(theta), 0, 0],
                        [             0,              0, 1, 0],
                        [             0,              0, 0, 1]])
        return ans

    def RPY2Mat(self, psi: float, theta: float, phi: float):
        return np.dot(self.RotZ(phi), np.dot(self.RotY(theta), self.RotX(psi)))

    def Mat2RPY(self, mat: np.ndarray):
        psi = np.atan2(+mat[2, 1], -mat[2, 0])
        phi = np.atan2(+mat[1, 2], +mat[0, 2])
        theta = np.atan2(+mat[2, 1] / np.sin(psi), +mat[2, 2])
        return np.array([psi, theta, phi])

    def InvT(self, T: np.ndarray):
        R = T[0: 3, 0: 3]
        p = T[0: 3, 3]
        ans = np.zeros((4, 4))
        ans[0: 3, 0: 3] = R.transpose()
        ans[0: 3, 3] = -np.dot(R.transpose(), p)
        return ans

    # modify DH method (Creig's book)
    def A1(self, theta: float, d: float):
        ans = np.array([[+np.cos(theta), -np.sin(theta), 0, 0],
                        [+np.sin(theta), +np.cos(theta), 0, 0],
                        [0, 0, 1, d],
                        [0, 0, 0, 1]], dtype=np.float)
        return ans

    def A2(self, alpha: float, a: float):
        ans = np.array([[1, 0, 0, a],
                        [0, +np.cos(alpha), -np.sin(alpha), 0],
                        [0, +np.sin(alpha), +np.cos(alpha), 0],
                        [0, 0, 0, 1]], dtype=np.float)
        return ans

    # modify DH method (Craig's book)
    #
    # i-1         i
    #  +----------+  Oi
    #             |         i+1
    #             +----------+  Qi+1
    #
    def MDH(self, a: float, alpha: float, d: float, theta: float):
        return np.matmul(self.A2(alpha, a), self.A1(theta, d))

    def MFK(self, theta: np.ndarray, index: int = -1):
        if index == -1:
            index = self.JOINT_SIZE
        T = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float)
        for i in range(index):
            if self.joint_type[i] == 0:
                T = np.matmul(T, self.MDH(self.A[i], self.ALPHA[i], self.D[i], self.THETA[i] + theta[i]))
            else:
                T = np.matmul(T, self.MDH(self.A[i], self.ALPHA[i], self.D[i] + theta[i], self.THETA[i]))
        return T

    def MDK(self, theta: np.ndarray, index: int = -1):
        if index == -1:
            index = self.JOINT_SIZE
        Te = self.MFK(theta, index)
        T = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float)
        J = np.zeros((6, index))
        for k in range(0, index):
            if self.joint_type[k] == 0:
                T = np.matmul(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k], self.THETA[k] + theta[k]))
                J[0: 3, k] = np.cross(T[0: 3, 2], Te[0: 3, 3] - T[0: 3, 3])
                J[3: 6, k] = T[0: 3, 2]
            else:
                T = np.matmul(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k] + theta[k], self.THETA[k]))
                J[0: 3, k] = T[0: 3, 2]
        return J

    def MIK(self, Rt: np.ndarray, Pt: np.ndarray, q: np.ndarray, iterate_times: int = 20, verbose=False):
        q_copy = q.copy()
        count = 0
        while count < iterate_times:
            count = count + 1
            Tc = self.MFK(q)
            dv = Pt - Tc[0: 3, 3]
            dw = 0.5 * (np.cross(Tc[0: 3, 0], Rt[0: 3, 0])
                        + np.cross(Tc[0: 3, 1], Rt[0: 3, 1])
                        + np.cross(Tc[0: 3, 2], Rt[0: 3, 2]))
            dx = np.array([dv[0], dv[1], dv[2], dw[0], dw[1], dw[2]])
            if np.linalg.norm(dx) < 1e-5:
                break
            J = self.MDK(q)
            if abs(np.linalg.norm(J)) < 1e-5:
                print('[Robotics MIK] singularity.')
                return q_copy

            dq = np.dot(np.linalg.pinv(J), dx) * 0.5

            # method 1 set robot to 6 dof
            # dq[6] = 0
            q += dq.reshape(q.shape)

        if verbose:
            print('[Robotics MIK] iterates {} times.'.format(count))
        if verbose and count >= iterate_times:
            print(np.linalg.norm(dx))
            print('[Robotics MIK] iterates more than {} times.'.format(iterate_times))
        return q

    def OriErrAR(self, Rc: np.ndarray, Rt: np.ndarray):
        Re = np.matmul(Rc.T, Rt)
        e = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        eo = np.dot(Rc, e.T)
        return eo


def RotX(theta: float):
    ans = np.array([[1,              0,              0],
                    [0, +np.cos(theta), -np.sin(theta)],
                    [0, +np.sin(theta), +np.cos(theta)]])
    return ans


def RotY(theta: float):
    ans = np.array([[+np.cos(theta), 0, +np.sin(theta)],
                    [             0, 1,              0],
                    [-np.sin(theta), 0, +np.cos(theta)]])
    return ans


def RotZ(theta: float):
    ans = np.array([[+np.cos(theta), -np.sin(theta), 0],
                    [+np.sin(theta), +np.cos(theta), 0],
                    [             0,              0, 1]])
    return ans


def euler2mat(x: float, y: float, z: float):
    """ Convert Euler Angles to Rotation Matrix. """
    Rx = RotX(x)
    Ry = RotY(y)
    Rz = RotZ(z)
    return Rx.dot(Ry).dot(Rz)

def skew(w):
    ans = np.array([[0, -w[2], w[1]],
                 [w[2], 0, -w[0]],
                 [-w[1], w[0], 0]])
    return ans
