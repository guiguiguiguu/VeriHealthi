#!/usr/bin/python
#
# Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
#
# 1. Redistributing the source code of this software is only allowed after
# receiving explicit, written permission from VeriSilicon. The copyright notice,
# this list of conditions and the following disclaimer must be retained in all
# source code distributions.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

import matplotlib.pyplot as plt
import numpy as np

def func_calculation(data,func_size,func):
    data_len = len(data)
    half_len = int((func_size-1)/2)
    func_data = np.zeros(data_len)
    for i in range(data_len):
        if i<=half_len:
            func_data[i] = func(data[0:2*i+1])
        else:
            if (i + half_len < data_len):
                func_data[i] = func(data[(i - half_len):(i + half_len+1)])
            else:
                func_data[i] = func(data[(2*i+1-data_len):data_len])
    return func_data

def remove_mean(a):
    return a-np.mean(a)

def if_a_in_B(a,B):
    B_len = len(B)
    i = 0
    r = 0
    while i < B_len:
        if B[i] == a:
            r = 1
            break
        i = i + 1
    return r

def delete_ith_a(a,i):
    a = np.array(a)
    a_len = len(a)
    if a_len <= 1 and i==0:
        return np.zeros(0)
    elif i < 0 or i > a_len - 1:
        return a
    else:
        if i == 0:
            return a[1:a_len]
        elif i == a_len-1:
            return a[0:a_len-1]
        else:
            return np.concatenate((a[0:i],a[i+1:a_len]))

class action_processor:
    def __init__(self,peak_num=125,valley_num=125):
        self.p_loc = np.zeros(peak_num,dtype=int)
        self.p_cnt = int(0)
        self.v_loc = np.zeros(valley_num,dtype=int)
        self.v_cnt = int(0)

    def find_possible_peak_valley(self,a):
        a_len = len(a)
        for i in range(1,a_len-1):
            if a[i] >= a[i-1] and a[i] > a[i+1]:
                self.p_loc[self.p_cnt] = i
                self.p_cnt = self.p_cnt + 1
            if a[i] <= a[i-1] and a[i] < a[i+1]:
                self.v_loc[self.v_cnt] = i
                self.v_cnt = self.v_cnt + 1
        self.p_loc = self.p_loc[0:self.p_cnt]
        self.v_loc = self.v_loc[0:self.v_cnt]
        return self

    def merge_close_pole(self,a,direction):
        if direction == 1:
            pole_locs = self.p_loc
            pole_cnt = self.p_cnt
            v_loc = self.v_loc
        elif direction == -1:
            pole_locs = self.v_loc
            pole_cnt = self.v_cnt
            v_loc = self.p_loc
        if pole_cnt > 1:
            i = 1
            while i < pole_cnt:
                r = 0
                j = pole_locs[i-1]
                while j < pole_locs[i]:
                    r = if_a_in_B(j,v_loc)
                    if r == 1:
                        break
                    j = j + 1
                if r == 0:
                    if a[pole_locs[i-1]]*direction > a[pole_locs[i]]*direction:
                        pole_locs = delete_ith_a(pole_locs,i)
                    else:
                        pole_locs = delete_ith_a(pole_locs,i-1)
                    pole_cnt = pole_cnt - 1
                else:
                    i = i + 1

        if direction == 1:
            self.p_loc = pole_locs
            self.p_cnt = pole_cnt
        else:
            self.v_loc = pole_locs
            self.v_cnt = pole_cnt
        return self

    def merge_close_peaks_valleys(self,a):
        if self.p_cnt > 1:
            self.merge_close_pole(a,1)
        if self.v_cnt > 1:
            self.merge_close_pole(a,-1)
        if self.p_cnt > 0 and ((self.v_cnt > 0 and self.p_loc[0] < self.v_loc[0]) or \
                               (self.v_cnt == 0)):
            self.p_loc = delete_ith_a(self.p_loc,0)
            self.p_cnt = self.p_cnt - 1
        return self

    def remove_false_peak_valley(self,a):
        a_mean = int(np.mean(a))
        i = 0
        while i < self.p_cnt:
            if a[self.p_loc[i]] < a_mean:
                self.p_loc = delete_ith_a(self.p_loc,i)
                self.p_cnt = self.p_cnt - 1
            else:
                i = i + 1
        i = 0
        while i < self.v_cnt:
            if a[self.v_loc[i]] > a_mean:
                self.v_loc = delete_ith_a(self.v_loc,i)
                self.v_cnt = self.v_cnt - 1
            else:
                i = i + 1

        return self

    def remove_asymmetric_peaks(self,a):
        last_valley_loc = int(0)
        TIME_THRESHOLD1 = 4
        TIME_THRESHOLD2 = 40
        if self.v_cnt <= 1:
            return self
        else:
            i = 0
            while i < self.p_cnt:
                r1 = 0
                r2 = 0
                j = 1
                while j < self.v_cnt:
                    if self.v_loc[j-1] < self.p_loc[i] and self.v_loc[j] > self.p_loc[i]:
                        r1 = 1
                        h1 = np.abs(a[self.p_loc[i]] - a[self.v_loc[j-1]])
                        h2 = np.abs(a[self.p_loc[i]] - a[self.v_loc[j]])
                        t1 = np.abs(self.p_loc[i] - self.v_loc[j-1])
                        t2 = np.abs(self.p_loc[i] - self.v_loc[j])
                        if h1 > PEAK_VALLEY_DIFFERENCE and h2 > PEAK_VALLEY_DIFFERENCE and \
                            h1 > h2/2 and h1 < h2*2 and \
                                t1 >= TIME_THRESHOLD1 and t1 <= TIME_THRESHOLD2 and t2 >= TIME_THRESHOLD1 and t2 <= TIME_THRESHOLD2:
                            last_valley_loc = self.v_loc[j]
                        else:
                            r2 = 1
                            self.p_loc = delete_ith_a(self.p_loc,i)
                            self.p_cnt = self.p_cnt - 1
                            if last_valley_loc != self.v_loc[j-1]:
                                self.v_loc = delete_ith_a(self.v_loc,j-1)
                                self.v_cnt = self.v_cnt - 1
                        break
                    j = j + 1
                if r1 == 0:
                    self.p_loc = delete_ith_a(self.p_loc,i)
                    self.p_cnt = self.p_cnt - 1
                elif r2 == 0:
                    i = i + 1

        return self

if __name__ == "__main__":
    file = "../../AccData/walk/IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt"

    data_len = 100000
    data = np.zeros(data_len)
    cnt = 0
    with open(file, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[5:]:
        data[cnt] = line
        cnt = cnt + 1
    data = data[0:cnt]
    data_len = cnt
    step_total = 0

    acc_len = int(cnt/7)
    # only use accelerometer data
    acc_data = np.zeros((acc_len,4))
    acc_data[:,0] = data[3:cnt:7]
    acc_data[:,1] = data[4:cnt:7]
    acc_data[:,2] = data[5:cnt:7]
    acc_data[:,3] = (acc_data[:,0]**2 + acc_data[:,1]**2 + acc_data[:,2]**2)**0.5

    m1 = 15
    m2 = 7
    acc_xyz_mean = np.zeros((acc_len,4))
    for i in range(3):
        tmp_data = acc_data[:,i]
        tmp_data = func_calculation(tmp_data,m1,np.mean)
        tmp_data = tmp_data.astype(int)
        tmp_data = func_calculation(tmp_data,m2,np.mean)
        tmp_data = tmp_data.astype(int)
        acc_xyz_mean[:,i] = tmp_data
    acc_xyz_mean[:,3] = (acc_xyz_mean[:,0]**2 + acc_xyz_mean[:,1]**2 + acc_xyz_mean[:,2]**2)**0.5

    plt.figure(11)
    plt.plot(acc_xyz_mean[:,0],'r')
    plt.plot(acc_xyz_mean[:,1],'g')
    plt.plot(acc_xyz_mean[:,2],'b')
    plt.grid(True)

    fs = 50
    win_sec = 5
    buf_sec = 3
    win_len = win_sec*fs
    buf_len = buf_sec*fs
    xyz_win = np.zeros((win_len,3))
    win_cnt = int(0)
    xyz_buf = np.zeros((buf_len,3))
    buf_cnt = np.zeros(3,dtype=int)
    ACC_SENSOR_GRAVITY = 4096
    STEP_ACC_DIFF_THRESHOLD = 4096 // 10
    PEAK_VALLEY_DIFFERENCE = 4096 // 14
    LEFT_DATA_NUM = int(2)

    axis_label = ['x','y','z']
    delay_point = m1//2 + m2//2
    for i in range(fs+fs-delay_point,acc_len-delay_point+1,fs):
        xyz_sec = acc_xyz_mean[(i-fs):i,0:3]
        if win_cnt + fs <= win_len:
            xyz_win[win_cnt:(win_cnt+fs),:] = xyz_sec
            win_cnt = win_cnt + fs
        else:
            xyz_win = np.concatenate((xyz_win[fs:win_len,:],xyz_sec))

        action_num = np.zeros(3,dtype=int)
        if win_cnt >= win_len:
            for j in range(3):
                if buf_cnt[j] > 0:
                    axis_win = np.concatenate((xyz_buf[0:buf_cnt[j],j],xyz_win[:,j]))
                    axis_len = win_cnt + buf_cnt[j]
                else:
                    axis_win = xyz_win[:,j]
                    axis_len = win_cnt
                xyz_buf[:,j] = np.zeros(buf_len)
                buf_cnt[j] = int(0)

                tmp_max = np.max(axis_win,0)
                tmp_min = np.min(axis_win,0)
                tmp_diff = np.max(tmp_max - tmp_min)
                axis_process = action_processor(250,250)
                if tmp_diff > STEP_ACC_DIFF_THRESHOLD:
                    axis_process = axis_process.find_possible_peak_valley(axis_win)

                    plt.figure(j)
                    plt.clf()
                    plt.subplot(2,2,1)
                    plt.plot(axis_win,'r')
                    plt.plot(axis_process.p_loc,axis_win[axis_process.p_loc],'go')
                    plt.plot(axis_process.v_loc,axis_win[axis_process.v_loc],'bo')
                    plt.grid(True)
                    plt.title(str(i) + '    possible peaks valleys:    '+ axis_label[j] + ':    ' + str(axis_process.p_cnt))

                    axis_process = axis_process.remove_false_peak_valley(axis_win)

                    plt.subplot(2,2,2)
                    plt.plot(axis_win,'r')
                    if axis_process.p_cnt > 0:
                        plt.plot(axis_process.p_loc,axis_win[axis_process.p_loc],'go')
                    if axis_process.v_cnt > 0:
                        plt.plot(axis_process.v_loc,axis_win[axis_process.v_loc],'bo')
                    plt.grid(True)
                    plt.title(str(i) + '    remove false peaks valleys:    '+ axis_label[j] + ':    ' + str(axis_process.p_cnt))

                    axis_process = axis_process.merge_close_peaks_valleys(axis_win)

                    plt.subplot(2,2,3)
                    plt.plot(axis_win,'r')
                    if axis_process.p_cnt > 0:
                        plt.plot(axis_process.p_loc,axis_win[axis_process.p_loc],'go')
                    if axis_process.v_cnt > 0:
                        plt.plot(axis_process.v_loc,axis_win[axis_process.v_loc],'bo')
                    plt.grid(True)
                    plt.title(str(i) + '    merge close peaks valleys:    '+ axis_label[j] + ':    ' + str(axis_process.p_cnt))

                    axis_process = axis_process.remove_asymmetric_peaks(axis_win)

                    if axis_process.v_cnt >= 1:
                        last_v_loc = axis_process.v_loc[axis_process.v_cnt-1]
                        left_len = axis_len - int(last_v_loc) + LEFT_DATA_NUM
                        if left_len < buf_len:
                            xyz_buf[0:left_len,j] = axis_win[(axis_len-left_len):axis_len]
                            buf_cnt[j] = int(left_len)

                    plt.subplot(2,2,4)
                    plt.plot(axis_win,'r')
                    if axis_process.p_cnt > 0:
                        plt.plot(axis_process.p_loc,axis_win[axis_process.p_loc],'gs')
                    if axis_process.v_cnt > 0:
                        plt.plot(axis_process.v_loc,axis_win[axis_process.v_loc],'bs')
                    if axis_process.v_cnt >= 1:
                        left_locs = range((axis_len-left_len),axis_len)
                        plt.plot(left_locs,axis_win[(axis_len-left_len):axis_len])
                    last_left_len = axis_len - win_cnt
                    if last_left_len > 0:
                        plt.plot(range(0,last_left_len),axis_win[0:last_left_len],'c')
                    plt.grid(True)
                    plt.title(str(i) + '    remove asymmetric peaks: ' + axis_label[j] + ':    ' + str(axis_process.p_cnt))
                    # plt.tight_layout()
                    plt.show()

                    action_num[j] = axis_process.p_cnt
            xyz_win = np.zeros((win_len,3))
            win_cnt = int(0)
        step_total += int(np.median(action_num))
        print("time = %3d(s)   idx = %3d   x = %3d   y = %3d   z = %3d   step = %d   total = %d"%(
            i//fs,i,action_num[0]*2,action_num[1]*2,action_num[2]*2,int(np.median(action_num))*2,step_total*2))
