import numpy as np


def loop_index(index, loop_T):
    if index <= loop_T and index > 0:
        return index
    if index <= 0:
        return index + loop_T
    return index % loop_T


class CategoryTracker:
    def __init__(self, history=100, init_min_hits=15, count_threshold=15):
        self.history = history
        self.categoryhistory = np.zeros((1, history), dtype=int)
        self.cur_index = -1
        self.past_category = 0
        self.standby_category = 0
        self.init_min_hits = init_min_hits
        self.count_threshold = count_threshold
        self.ready_for_switch_cnt = 0
        self.latest_category = -1
        self.different = False
        self.Initial_flag = False
        self.frame_count = 0
        self.init_dect_count = 0
        self.return_res = -1
        self.return_category = 0

    def _result(self, result, category):
        self.return_res = result
        self.return_category = category
        return result, category

    def _category(self, theta, dets_topleft):
        if (np.abs(theta + np.pi / 2) < 10 / 180 * np.pi and
                np.abs(dets_topleft[0, 0] - np.pi / 2) < 10 / 180):
            return loop_index(int(dets_topleft[0, 1]) + 2, 4)
        if (np.abs(theta - np.pi / 2) < 10 / 180 * np.pi and
                np.abs(dets_topleft[0, 0] + np.pi / 2) < 10 / 180):
            return loop_index(int(dets_topleft[0, 1]) + 2, 4)
        return int(dets_topleft[0, 1])

    def update(self, theta=None, dets_topleft=np.empty((0, 2))):
        if not self.Initial_flag:
            if len(dets_topleft) == 0:
                self.init_dect_count = 0
                return self._result(-1, 0)
            self.latest_category = self._category(theta, dets_topleft)
            self.cur_index = (self.cur_index + 1) % self.history
            self.categoryhistory[0, self.cur_index] = self.latest_category
            self.past_category = self.latest_category
            if self.standby_category == self.latest_category:
                self.init_dect_count += 1
                if self.init_dect_count > self.init_min_hits:
                    self.Initial_flag = True
                    return self._result(1, self.latest_category)
            else:
                self.init_dect_count = 0
                self.standby_category = self.latest_category
            return self._result(-1, 0)

        if len(dets_topleft) == 0:
            self.ready_for_switch_cnt = 0
            self.Initial_flag = False
            self.different = False
            return self._result(-1, 0)

        self.latest_category = self._category(theta, dets_topleft)
        theta_adjust = theta - dets_topleft[0, 0]
        if np.abs(theta_adjust - np.pi / 2) < 10 / 180 * np.pi:
            self.latest_category = loop_index(int(dets_topleft[0, 1]) + 1, 4)
        elif np.abs(theta_adjust + np.pi / 2) < 10 / 180 * np.pi:
            self.latest_category = loop_index(int(dets_topleft[0, 1]) - 1, 4)

        self.cur_index = (self.cur_index + 1) % self.history
        self.categoryhistory[0, self.cur_index] = self.latest_category
        if not self.different:
            if self.past_category == self.latest_category:
                return self._result(1, self.latest_category)
            self.different = True
            self.ready_for_switch_cnt = 0
            self.standby_category = self.latest_category
            return self._result(2, self.past_category)

        if self.standby_category != self.latest_category:
            self.standby_category = self.latest_category
            self.ready_for_switch_cnt = 0
            return self._result(2, self.past_category)

        self.ready_for_switch_cnt += 1
        if self.ready_for_switch_cnt > self.count_threshold:
            self.different = False
            self.ready_for_switch_cnt = 0
            self.past_category = self.latest_category
            return self._result(1, self.latest_category)
        return self._result(2, self.past_category)
