#!/usr/bin/python
"""
改进的计步算法 V3 - 保留原始峰值检测 + 6轴噪声过滤

策略：
1. 步数检测完全使用原始算法（经过验证的3轴峰值检测）
2. 增加基于6轴数据的活动分类器用于噪声过滤
3. 对噪声窗口输出0，对步行窗口输出原始算法结果
4. 同时优化对跑步的检测（跑步加速度特征与走路不同）
"""

import numpy as np
import os
import glob
import re
import sys


def func_calculation(data, func_size, func):
    data_len = len(data)
    half_len = int((func_size - 1) / 2)
    func_data = np.zeros(data_len)
    for i in range(data_len):
        if i <= half_len:
            func_data[i] = func(data[0:2 * i + 1])
        else:
            if (i + half_len < data_len):
                func_data[i] = func(data[(i - half_len):(i + half_len + 1)])
            else:
                func_data[i] = func(data[(2 * i + 1 - data_len):data_len])
    return func_data


def compute_autocorrelation(signal, max_lag=None):
    n = len(signal)
    if max_lag is None:
        max_lag = n // 2
    signal = signal - np.mean(signal)
    result = np.zeros(max_lag)
    for lag in range(max_lag):
        if n - lag > 0:
            result[lag] = np.sum(signal[:n - lag] * signal[lag:])
    if result[0] > 0:
        result = result / result[0]
    return result


def if_a_in_B(a, B):
    B_len = len(B)
    i = 0
    while i < B_len:
        if B[i] == a:
            return 1
        i = i + 1
    return 0


def delete_ith_a(a, i):
    a = np.array(a)
    a_len = len(a)
    if a_len <= 1 and i == 0:
        return np.zeros(0)
    elif i < 0 or i > a_len - 1:
        return a
    else:
        if i == 0:
            return a[1:a_len]
        elif i == a_len - 1:
            return a[0:a_len - 1]
        else:
            return np.concatenate((a[0:i], a[i + 1:a_len]))


class ActionProcessor:
    """原始算法的峰值检测处理器（保持不变）"""
    def __init__(self, peak_num=250, valley_num=250):
        self.p_loc = np.zeros(peak_num, dtype=int)
        self.p_cnt = 0
        self.v_loc = np.zeros(valley_num, dtype=int)
        self.v_cnt = 0

    def find_possible_peak_valley(self, a):
        a_len = len(a)
        for i in range(1, a_len - 1):
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                self.p_loc[self.p_cnt] = i
                self.p_cnt += 1
            if a[i] <= a[i - 1] and a[i] < a[i + 1]:
                self.v_loc[self.v_cnt] = i
                self.v_cnt += 1
        self.p_loc = self.p_loc[0:self.p_cnt]
        self.v_loc = self.v_loc[0:self.v_cnt]
        return self

    def merge_close_pole(self, a, direction):
        if direction == 1:
            pole_locs, pole_cnt = self.p_loc, self.p_cnt
            v_loc = self.v_loc
        else:
            pole_locs, pole_cnt = self.v_loc, self.v_cnt
            v_loc = self.p_loc
        if pole_cnt > 1:
            i = 1
            while i < pole_cnt:
                r = 0
                j = pole_locs[i - 1]
                while j < pole_locs[i]:
                    r = if_a_in_B(j, v_loc)
                    if r == 1:
                        break
                    j += 1
                if r == 0:
                    if a[pole_locs[i - 1]] * direction > a[pole_locs[i]] * direction:
                        pole_locs = delete_ith_a(pole_locs, i)
                    else:
                        pole_locs = delete_ith_a(pole_locs, i - 1)
                    pole_cnt -= 1
                else:
                    i += 1
        if direction == 1:
            self.p_loc, self.p_cnt = pole_locs, pole_cnt
        else:
            self.v_loc, self.v_cnt = pole_locs, pole_cnt
        return self

    def merge_close_peaks_valleys(self, a):
        if self.p_cnt > 1:
            self.merge_close_pole(a, 1)
        if self.v_cnt > 1:
            self.merge_close_pole(a, -1)
        if self.p_cnt > 0 and ((self.v_cnt > 0 and self.p_loc[0] < self.v_loc[0]) or
                               (self.v_cnt == 0)):
            self.p_loc = delete_ith_a(self.p_loc, 0)
            self.p_cnt -= 1
        return self

    def remove_false_peak_valley(self, a):
        a_mean = int(np.mean(a))
        i = 0
        while i < self.p_cnt:
            if a[self.p_loc[i]] < a_mean:
                self.p_loc = delete_ith_a(self.p_loc, i)
                self.p_cnt -= 1
            else:
                i += 1
        i = 0
        while i < self.v_cnt:
            if a[self.v_loc[i]] > a_mean:
                self.v_loc = delete_ith_a(self.v_loc, i)
                self.v_cnt -= 1
            else:
                i += 1
        return self

    def remove_asymmetric_peaks(self, a):
        TIME_THRESHOLD1 = 4
        TIME_THRESHOLD2 = 40
        PEAK_VALLEY_DIFFERENCE = 4096 // 14
        if self.v_cnt <= 1:
            return self

        last_valley_loc = 0
        i = 0
        while i < self.p_cnt:
            r1 = 0
            r2 = 0
            j = 1
            while j < self.v_cnt:
                if self.v_loc[j - 1] < self.p_loc[i] and self.v_loc[j] > self.p_loc[i]:
                    r1 = 1
                    h1 = np.abs(a[self.p_loc[i]] - a[self.v_loc[j - 1]])
                    h2 = np.abs(a[self.p_loc[i]] - a[self.v_loc[j]])
                    t1 = np.abs(self.p_loc[i] - self.v_loc[j - 1])
                    t2 = np.abs(self.p_loc[i] - self.v_loc[j])
                    if (h1 > PEAK_VALLEY_DIFFERENCE and h2 > PEAK_VALLEY_DIFFERENCE and
                        h1 > h2 / 2 and h1 < h2 * 2 and
                        t1 >= TIME_THRESHOLD1 and t1 <= TIME_THRESHOLD2 and
                        t2 >= TIME_THRESHOLD1 and t2 <= TIME_THRESHOLD2):
                        last_valley_loc = self.v_loc[j]
                    else:
                        r2 = 1
                        self.p_loc = delete_ith_a(self.p_loc, i)
                        self.p_cnt -= 1
                        if last_valley_loc != self.v_loc[j - 1]:
                            self.v_loc = delete_ith_a(self.v_loc, j - 1)
                            self.v_cnt -= 1
                    break
                j += 1
            if r1 == 0:
                self.p_loc = delete_ith_a(self.p_loc, i)
                self.p_cnt -= 1
            elif r2 == 0:
                i += 1
        return self


class ImprovedStepCounterV3:
    """改进的计步器 V3: 原始峰值检测 + 6轴噪声分类"""

    def __init__(self):
        self.fs = 50
        self.m1 = 15
        self.m2 = 7
        self.win_sec = 5
        self.buf_sec = 3
        self.win_len = self.win_sec * self.fs
        self.buf_len = self.buf_sec * self.fs
        self.STEP_ACC_DIFF_THRESHOLD = 4096 // 10
        self.PEAK_VALLEY_DIFFERENCE = 4096 // 14
        self.LEFT_DATA_NUM = 2
        self.ACC_GRAVITY = 4096

    def load_imu_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = np.array([int(line.strip()) for line in lines[5:]], dtype=np.float64)
        cnt = len(data) // 7 * 7
        data = data[:cnt].reshape(-1, 7)
        return (data[:, 0], data[:, 1], data[:, 2],
                data[:, 3], data[:, 4], data[:, 5])

    def compute_features(self, acc_win_3axes, gyro_win_3axes):
        """
        计算窗口的6轴特征用于活动分类
        acc_win_3axes: (N, 3) 加速度 [x, y, z]
        gyro_win_3axes: (N, 3) 陀螺仪 [x, y, z]
        """
        features = {}

        # 加速度幅度特征
        acc_mag = np.sqrt(np.sum(acc_win_3axes**2, axis=1))
        acc_detrend = acc_mag - np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_detrend)
        features['acc_mag_range'] = np.max(acc_mag) - np.min(acc_mag)

        # 陀螺仪幅度特征
        gyro_mag = np.sqrt(np.sum(gyro_win_3axes**2, axis=1))
        features['gyro_mag_mean'] = np.mean(gyro_mag)
        features['gyro_mag_std'] = np.std(gyro_mag)
        features['gyro_energy'] = np.sum(gyro_mag**2) / len(gyro_mag)

        # 能量比值
        acc_energy = np.sum(acc_detrend**2)
        if acc_energy > 0:
            features['gyro_acc_ratio'] = np.sum(gyro_mag**2) / acc_energy
        else:
            features['gyro_acc_ratio'] = 0

        # 节律性特征（自相关）
        if len(acc_mag) >= 100:
            autocorr = compute_autocorrelation(acc_mag, min(len(acc_mag) // 2, 150))
            search_start = 10   # 0.2s (~5Hz max)
            search_end = 100    # 2s (~0.5Hz min)
            if search_end < len(autocorr):
                search_range = autocorr[search_start:search_end]
                features['autocorr_peak'] = np.max(search_range)
                features['autocorr_peak_idx'] = search_start + np.argmax(search_range)
            else:
                features['autocorr_peak'] = 0
                features['autocorr_peak_idx'] = 0

            # 自相关峰值宽度（尖锐度）
            if features['autocorr_peak'] > 0.1:
                peak_idx = features['autocorr_peak_idx']
                peak_val = features['autocorr_peak']
                # Find width at half max
                half_max = peak_val * 0.7
                left = peak_idx
                while left > 0 and autocorr[left] > half_max:
                    left -= 1
                right = peak_idx
                while right < len(autocorr) - 1 and autocorr[right] > half_max:
                    right += 1
                features['autocorr_width'] = right - left
            else:
                features['autocorr_width'] = 999

            # 谐波峰值（周期性的标志）
            if features['autocorr_peak_idx'] * 2 < len(autocorr):
                h_idx = features['autocorr_peak_idx'] * 2
                h_start = max(0, h_idx - 5)
                h_end = min(len(autocorr), h_idx + 6)
                features['harmonic_peak'] = np.max(autocorr[h_start:h_end])
            else:
                features['harmonic_peak'] = 0
        else:
            features['autocorr_peak'] = 0
            features['autocorr_peak_idx'] = 0
            features['autocorr_width'] = 999
            features['harmonic_peak'] = 0

        # 峰值一致性（检测步态规律性）
        if len(acc_detrend) > 20:
            # 寻找所有局部峰值
            peaks = []
            for i in range(1, len(acc_detrend) - 1):
                if acc_detrend[i] > acc_detrend[i - 1] and acc_detrend[i] > acc_detrend[i + 1]:
                    if acc_detrend[i] > np.std(acc_detrend) * 0.3:
                        peaks.append(acc_detrend[i])
            if len(peaks) >= 3:
                features['peak_height_cv'] = np.std(peaks) / (np.mean(peaks) + 1e-6)
                # 峰值间隔的变异系数
                if len(peaks) > 1:
                    peak_indices = []
                    for i in range(1, len(acc_detrend) - 1):
                        if acc_detrend[i] > acc_detrend[i - 1] and acc_detrend[i] > acc_detrend[i + 1]:
                            if acc_detrend[i] > np.std(acc_detrend) * 0.3:
                                peak_indices.append(i)
                    if len(peak_indices) >= 3:
                        intervals = np.diff(peak_indices)
                        features['peak_interval_cv'] = np.std(intervals) / (np.mean(intervals) + 1e-6)
                    else:
                        features['peak_interval_cv'] = 999
                else:
                    features['peak_interval_cv'] = 999
            else:
                features['peak_height_cv'] = 999
                features['peak_interval_cv'] = 999
        else:
            features['peak_height_cv'] = 999
            features['peak_interval_cv'] = 999

        return features

    def is_walking_or_running(self, features):
        """
        基于6轴特征判断是否为真实的步行/跑步
        返回: (is_valid, activity_type)
        """

        acc_std = features['acc_mag_std']
        acc_range = features['acc_mag_range']

        # 静止判定
        if acc_std < self.ACC_GRAVITY // 30:  # <0.03G
            return False, 'stationary'

        # 运动量太小
        if acc_range < self.ACC_GRAVITY // 6:  # <0.17G
            return False, 'stationary'

        gyro_energy = features['gyro_energy']
        gyro_std = features['gyro_mag_std']
        autocorr_peak = features['autocorr_peak']
        harmonic_peak = features['harmonic_peak']
        autocorr_width = features['autocorr_width']
        peak_height_cv = features.get('peak_height_cv', 999)
        peak_interval_cv = features.get('peak_interval_cv', 999)
        gyro_acc_ratio = features['gyro_acc_ratio']

        # === 跑步检测 ===
        # 跑步：高加速度变化 + 高节律性
        is_high_acc = acc_std > self.ACC_GRAVITY * 1.0  # >1G
        is_very_high_acc = acc_std > self.ACC_GRAVITY * 2.0  # >2G

        if is_high_acc and autocorr_peak > 0.1:
            # 高加速度 + 节律 = 跑步
            if acc_range > self.ACC_GRAVITY * 2:
                return True, 'run'
            if harmonic_peak > 0.05 and autocorr_width < 30:
                return True, 'run'

        # 极高加速度（无节律也可能是跑步，因为跑步窗口采样可能不完整）
        if is_very_high_acc and acc_range > self.ACC_GRAVITY * 3:
            return True, 'run'

        # === 步行检测 ===
        # 步行特征：中等加速度 + 明确节律 + 合理陀螺仪活动

        # 核心节律性检查
        has_rhythm = autocorr_peak > 0.12
        has_strong_rhythm = autocorr_peak > 0.2

        # 自相关峰宽度检查：步行有尖锐的峰值
        has_sharp_peak = autocorr_width < 40 and autocorr_width > 2

        # 峰值一致性检查：步行峰值高度和间隔都较规律
        has_consistent_peaks = (peak_height_cv < 1.5 and peak_interval_cv < 1.0)

        # 陀螺仪验证
        has_gyro_activity = gyro_energy > 30000
        has_strong_gyro = gyro_energy > 200000

        # 加速度变化足够
        sufficient_acc = acc_std > self.ACC_GRAVITY // 4  # >0.25G
        good_acc = acc_std > self.ACC_GRAVITY // 2.5  # >0.4G

        # === 综合判断 ===

        # 高置信度步行: 强节律 + 加速度足够 + 峰值规律
        if has_strong_rhythm and sufficient_acc and has_sharp_peak:
            if has_consistent_peaks:
                return True, 'walk'
            if has_gyro_activity:
                return True, 'walk'

        # 中等置信度步行: 有节律 + 加速度好 + 陀螺仪活动
        if has_rhythm and good_acc and has_gyro_activity:
            if harmonic_peak > 0.05:
                return True, 'walk'

        # 低加速度但有明显规律（可能是慢走或手未摆动）
        if has_strong_rhythm and acc_std > self.ACC_GRAVITY // 5:
            if has_consistent_peaks or harmonic_peak > 0.08:
                return True, 'walk'

        # 跑步备选: 加速度非常大 + 有陀螺仪活动
        if acc_std > self.ACC_GRAVITY * 1.5 and has_gyro_activity:
            if autocorr_peak > 0.08:
                return True, 'run'

        # === 噪声排除逻辑 ===
        # 以下情况很可能是噪声而非步行:

        # 有加速度但自相关峰值宽（非周期性运动）
        if sufficient_acc and autocorr_peak > 0.08 and autocorr_width > 50:
            if not has_gyro_activity:
                return False, 'noise'

        # 峰值高度变异太大（不规则运动）
        if sufficient_acc and peak_height_cv > 2.0 and peak_interval_cv > 1.5:
            if not has_strong_rhythm:
                return False, 'noise'

        # 极高陀螺仪能量但加速度低（只有手臂在动）
        if gyro_energy > 1000000 and acc_std < self.ACC_GRAVITY // 2:
            return False, 'noise'

        # 默认：如果加速度足够且有节律，倾向认为是步行
        if good_acc and has_rhythm:
            return True, 'walk'

        # 最后的后备判断
        if sufficient_acc and has_gyro_activity and autocorr_peak > 0.1:
            return True, 'walk'

        return False, 'noise'

    def process_window_original(self, axis_win, buf_data, buf_cnt):
        """
        使用原始算法处理单个轴的数据窗口
        返回: (step_count_for_this_axis, new_buf_data, new_buf_cnt)
        """
        axis_len = len(axis_win)

        if buf_cnt > 0:
            full_data = np.concatenate([buf_data[:buf_cnt], axis_win])
            full_len = axis_len + buf_cnt
        else:
            full_data = axis_win
            full_len = axis_len

        result_steps = 0
        new_buf = np.zeros(self.buf_len)
        new_buf_cnt = 0

        tmp_max = np.max(full_data)
        tmp_min = np.min(full_data)
        tmp_diff = tmp_max - tmp_min

        if tmp_diff > self.STEP_ACC_DIFF_THRESHOLD:
            processor = ActionProcessor(250, 250)
            processor.find_possible_peak_valley(full_data)
            processor.remove_false_peak_valley(full_data)
            processor.merge_close_peaks_valleys(full_data)
            processor.remove_asymmetric_peaks(full_data)

            result_steps = processor.p_cnt

            if processor.v_cnt >= 1:
                last_v_loc = processor.v_loc[processor.v_cnt - 1]
                left_len = full_len - int(last_v_loc) + self.LEFT_DATA_NUM
                if left_len < self.buf_len and left_len > 0:
                    new_buf[:left_len] = full_data[full_len - left_len:full_len]
                    new_buf_cnt = left_len

        return result_steps, new_buf, new_buf_cnt

    def process_file(self, filepath, verbose=False):
        """处理单个数据文件"""
        fname = os.path.basename(filepath)
        match = re.search(r'step(\d+)', fname)
        true_steps = int(match.group(1)) if match else 0

        # 加载6轴数据
        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = self.load_imu_data(filepath)
        total_samples = len(acc_x)

        if total_samples < self.fs:
            return 0, true_steps

        # 加速度预处理
        acc_data = np.zeros((total_samples, 4))
        acc_data[:, 0] = acc_x
        acc_data[:, 1] = acc_y
        acc_data[:, 2] = acc_z

        acc_xyz_mean = np.zeros((total_samples, 4))
        for j in range(3):
            tmp = func_calculation(acc_data[:, j], self.m1, np.mean)
            tmp = tmp.astype(int)
            tmp = func_calculation(tmp, self.m2, np.mean)
            tmp = tmp.astype(int)
            acc_xyz_mean[:, j] = tmp
        acc_xyz_mean[:, 3] = np.sqrt(acc_xyz_mean[:, 0]**2 +
                                      acc_xyz_mean[:, 1]**2 +
                                      acc_xyz_mean[:, 2]**2)

        # 陀螺仪预处理
        gyro_filt = np.zeros((total_samples, 3))
        for j in range(3):
            gyro_raw = np.array([gyro_x, gyro_y, gyro_z])[j]
            gyro_filt[:, j] = func_calculation(gyro_raw, 7, np.mean)

        # 窗口变量
        xyz_win = np.zeros((self.win_len, 3))
        gyro_win = np.zeros((self.win_len, 3))
        win_cnt = 0

        xyz_buf = np.zeros((self.buf_len, 3))
        gyro_buf = np.zeros((self.buf_len, 3))
        buf_cnt = np.zeros(3, dtype=int)

        total_steps = 0
        delay_point = self.m1 // 2 + self.m2 // 2

        for i in range(self.fs + self.fs - delay_point,
                       total_samples - delay_point + 1, self.fs):
            seg_end = i
            seg_start = i - self.fs

            xyz_sec = acc_xyz_mean[seg_start:seg_end, 0:3]
            gyro_sec = gyro_filt[seg_start:seg_end, :]

            seg_len = len(xyz_sec)

            if win_cnt + seg_len <= self.win_len:
                xyz_win[win_cnt:win_cnt + seg_len, :] = xyz_sec
                gyro_win[win_cnt:win_cnt + seg_len, :] = gyro_sec
                win_cnt += seg_len
            else:
                xyz_win = np.concatenate((xyz_win[seg_len:self.win_len, :], xyz_sec))
                gyro_win = np.concatenate((gyro_win[seg_len:self.win_len, :], gyro_sec))
                win_cnt = self.win_len

            if win_cnt >= self.win_len:
                # 构建完整窗口（含缓冲区）
                full_xyz = xyz_win[:win_cnt].copy()
                full_gyro = gyro_win[:win_cnt].copy()

                # ---- 活动分类（使用6轴数据） ----
                features = self.compute_features(full_xyz, full_gyro)
                is_walking, activity_type = self.is_walking_or_running(features)

                action_num = np.zeros(3, dtype=int)

                if is_walking:
                    # 使用原始算法进行步数检测
                    for j in range(3):
                        steps_j, new_buf_j, new_cnt_j = self.process_window_original(
                            full_xyz[:, j],
                            xyz_buf[:int(buf_cnt[j]), j],
                            int(buf_cnt[j])
                        )
                        action_num[j] = steps_j
                        xyz_buf[:self.buf_len, j] = 0
                        xyz_buf[:new_cnt_j, j] = new_buf_j[:new_cnt_j]
                        buf_cnt[j] = new_cnt_j
                else:
                    # 噪声/静止 → 步数为0，清空缓冲区
                    for j in range(3):
                        xyz_buf[:, j] = 0
                        buf_cnt[j] = 0

                # 原始算法的步数 = 中位数 * 2
                window_steps = int(np.median(action_num)) * 2
                total_steps += window_steps

                if verbose:
                    acc_std = features['acc_mag_std']
                    gyro_e = features['gyro_energy']
                    ap = features['autocorr_peak']
                    print(f"  time={seg_end//self.fs:3d}s  {activity_type:10s}  "
                          f"steps={window_steps:3d}  acc_std={acc_std:5.0f}  "
                          f"gyro_e={gyro_e:8.0f}  acorr={ap:.3f}")

                xyz_win = np.zeros((self.win_len, 3))
                gyro_win = np.zeros((self.win_len, 3))
                win_cnt = 0

        if verbose:
            print(f"  文件: {fname}  预测: {total_steps}  真实: {true_steps}")

        return total_steps, true_steps

    def evaluate_all(self, data_dir, verbose=False):
        """评估所有数据文件"""
        results = {'walk': [], 'run': [], 'others': []}

        for category in ['walk', 'run', 'others']:
            cat_dir = os.path.join(data_dir, category)
            if not os.path.exists(cat_dir):
                continue
            files = sorted(glob.glob(os.path.join(cat_dir, '*.txt')))

            for idx, f in enumerate(files):
                predicted, true = self.process_file(f, verbose=verbose)
                results[category].append({
                    'file': os.path.basename(f),
                    'predicted': predicted,
                    'true': true,
                    'error': predicted - true,
                    'abs_error': abs(predicted - true)
                })
                if (idx + 1) % 10 == 0:
                    print(f"  {category}: {idx+1}/{len(files)}")

            print(f"  已完成 {category}: {len(files)} 个文件")

        return results

    def compute_metrics(self, results):
        """计算并打印评估指标"""
        print("\n" + "=" * 70)
        print("改进的计步算法 V3 - 评估结果汇总")
        print("=" * 70)

        walk_run = results['walk'] + results['run']

        all_data = [
            ('步行 (walk+run)', walk_run),
            ('走路 (walk)', results['walk']),
            ('跑步 (run)', results['run']),
            ('噪声 (others)', results['others']),
        ]

        for name, data in all_data:
            if len(data) == 0:
                continue

            true_steps = [d['true'] for d in data]
            pred_steps = [d['predicted'] for d in data]
            abs_errors = [d['abs_error'] for d in data]

            mae = np.mean(abs_errors)
            non_zero = [(abs(p - t) / t * 100) for p, t in zip(pred_steps, true_steps) if t > 0]
            mape = np.mean(non_zero) if non_zero else float('nan')

            if '噪声' in name:
                fp_rate = sum(1 for p in pred_steps if p > 0) / len(pred_steps) * 100
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE: {mae:.1f} 步")
                print(f"  误检率: {fp_rate:.1f}%")
            else:
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE: {mae:.1f} 步")
                print(f"  MAPE: {mape:.1f}%")
                print(f"  准确率: {max(0, 100-mape):.1f}%")

        print("\n" + "-" * 70)
        print("详细结果 (步行+跑步):")
        for d in walk_run:
            ape = abs(d['error']) / max(1, d['true']) * 100
            print(f"  {d['file']:55s} pred={d['predicted']:4d}  true={d['true']:4d}  "
                  f"err={d['error']:+4d}  APE={ape:5.1f}%")

        noise_errors = [d for d in results['others'] if d['predicted'] > 0]
        if noise_errors:
            print(f"\n噪声误检 ({len(noise_errors)}/{len(results['others'])}):")
            for d in noise_errors:
                print(f"  {d['file']:55s} pred={d['predicted']:4d}")

        return mae, mape


def main():
    data_dir = 'VeriHealthi_Algorithm_Homework_Code_Data/AccData'

    print("改进的计步算法 V3 - 原始峰值检测 + 6轴噪声过滤")
    print("=" * 70)

    counter = ImprovedStepCounterV3()

    # 演示
    demo_file = os.path.join(data_dir, 'walk',
                             'IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt')
    print(f"\n演示: {os.path.basename(demo_file)}")
    print("-" * 40)
    counter.process_file(demo_file, verbose=True)

    # 批量评估
    print("\n\n批量评估...")
    results = counter.evaluate_all(data_dir)
    counter.compute_metrics(results)

    return results


if __name__ == '__main__':
    main()
