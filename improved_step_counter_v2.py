#!/usr/bin/python
"""
改进的计步算法 V2 - 基于原始算法的峰值检测 + 6轴数据噪声过滤

策略：
1. 保留原始算法的3轴峰值检测（经过验证的步数检测能力）
2. 增加基于6轴数据的活动分类器（区分 walk/run/noise）
3. 对噪声窗口输出0步，对步行窗口输出原始算法的步数
4. 同时增加基于加速度幅度的峰值检测（方向无关），取两者较优结果
"""

import numpy as np
import os
import glob
import re
import sys


def func_calculation(data, func_size, func):
    """滑动窗口计算（与原版相同）"""
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
    """计算信号的自相关函数（归一化）"""
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


def compute_rhythm_features(acc_mag_segment):
    """
    计算节律性特征
    返回: (rhythm_score, dominant_period)
    """
    n = len(acc_mag_segment)
    if n < 20:
        return 0.0, 0

    acc_detrend = acc_mag_segment - np.mean(acc_mag_segment)
    max_lag = min(n // 2, 150)  # 最多3秒滞后
    autocorr = compute_autocorrelation(acc_detrend, max_lag)

    # 在步频范围 0.5-5Hz (10-100 采样点@50Hz) 内搜索峰值
    min_lag = 10
    max_search = min(100, len(autocorr) - 1)

    if max_search <= min_lag:
        return 0.0, 0

    search_range = autocorr[min_lag:max_search + 1]
    peak_idx = np.argmax(search_range)
    peak_val = search_range[peak_idx]
    peak_lag = min_lag + peak_idx

    # 检查谐波：如果存在 2*peak_lag 处的峰值，则是强节律信号
    harmonic_score = 0.0
    if peak_lag * 2 < len(autocorr):
        h_start = max(0, peak_lag * 2 - 5)
        h_end = min(len(autocorr), peak_lag * 2 + 5)
        harmonic_peak = np.max(autocorr[h_start:h_end])
        if harmonic_peak > 0.1:
            harmonic_score = harmonic_peak * 0.5

    rhythm_score = peak_val + harmonic_score
    return rhythm_score, peak_lag


class ImprovedStepCounterV2:
    """改进的计步器 V2: 原始峰值检测 + 6轴噪声过滤"""

    def __init__(self):
        self.fs = 50
        self.m1 = 15
        self.m2 = 7

        self.win_sec = 5
        self.buf_sec = 3
        self.win_len = self.win_sec * self.fs
        self.buf_len = self.buf_sec * self.fs

        self.ACC_GRAVITY = 4096

        # 峰值检测参数（继承自原始算法）
        self.STEP_ACC_DIFF_THRESHOLD = self.ACC_GRAVITY // 10
        self.PEAK_VALLEY_DIFFERENCE = self.ACC_GRAVITY // 14
        self.TIME_THRESHOLD1 = 4   # 最小半周期
        self.TIME_THRESHOLD2 = 40  # 最大半周期
        self.LEFT_DATA_NUM = 2

        # 活动分类阈值
        self.STATIONARY_ACC_STD = self.ACC_GRAVITY // 30  # 静止
        self.LOW_ACC_STD = self.ACC_GRAVITY // 6          # 低运动量

        # 陀螺仪阈值
        self.GYRO_SCALE = 16.4
        # 走路时典型的陀螺仪能量范围
        self.MIN_GYRO_ENERGY_WALK = 50000     # 有摆臂
        self.MIN_GYRO_ENERGY_STRONG = 500000  # 强陀螺仪活动

    def load_imu_data(self, filepath):
        """加载6轴IMU数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = np.array([int(line.strip()) for line in lines[5:]], dtype=np.float64)
        cnt = len(data) // 7 * 7
        data = data[:cnt].reshape(-1, 7)
        return (data[:, 0], data[:, 1], data[:, 2],
                data[:, 3], data[:, 4], data[:, 5])

    def preprocess(self, acc_x, acc_y, acc_z):
        """预处理加速度数据"""
        acc_x_f = func_calculation(acc_x, self.m1, np.mean).astype(np.float64)
        acc_x_f = func_calculation(acc_x_f, self.m2, np.mean)
        acc_y_f = func_calculation(acc_y, self.m1, np.mean).astype(np.float64)
        acc_y_f = func_calculation(acc_y_f, self.m2, np.mean)
        acc_z_f = func_calculation(acc_z, self.m1, np.mean).astype(np.float64)
        acc_z_f = func_calculation(acc_z_f, self.m2, np.mean)

        acc_mag = np.sqrt(acc_x_f**2 + acc_y_f**2 + acc_z_f**2)
        return acc_x_f, acc_y_f, acc_z_f, acc_mag

    def preprocess_gyro(self, gyro_x, gyro_y, gyro_z):
        """预处理陀螺仪数据"""
        gx_f = func_calculation(gyro_x, 7, np.mean)
        gy_f = func_calculation(gyro_y, 7, np.mean)
        gz_f = func_calculation(gyro_z, 7, np.mean)
        gyro_mag = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)
        return gyro_mag

    def find_peaks_valleys(self, a):
        """寻找波峰和波谷"""
        p_loc, v_loc = [], []
        for i in range(1, len(a) - 1):
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                p_loc.append(i)
            if a[i] <= a[i - 1] and a[i] < a[i + 1]:
                v_loc.append(i)
        return np.array(p_loc, dtype=int), np.array(v_loc, dtype=int)

    def merge_close_poles(self, a, p_loc, v_loc, direction):
        """合并之间没有反向极值点的同向极值点"""
        if len(p_loc) <= 1:
            return p_loc

        result = p_loc.copy()
        i = 1
        while i < len(result):
            has_opposite = False
            for v in v_loc:
                if result[i - 1] < v < result[i]:
                    has_opposite = True
                    break
            if not has_opposite:
                # 保留值更大的
                if a[result[i - 1]] * direction > a[result[i]] * direction:
                    result = np.delete(result, i)
                else:
                    result = np.delete(result, i - 1)
            else:
                i += 1
        return result

    def count_steps_original_method(self, axis_data):
        """
        使用原始算法的峰值检测方法统计单个轴的步数
        返回: 该轴检测到的步数（= 峰值数 * 2）
        """
        # 步骤1: 找可能的波峰波谷
        p_loc, v_loc = self.find_peaks_valleys(axis_data)

        if len(p_loc) == 0:
            return 0

        # 步骤2: 去除伪峰/伪谷（基于均值）
        a_mean = np.mean(axis_data)
        valid_p = []
        for p in p_loc:
            if axis_data[p] >= a_mean:
                valid_p.append(p)
        p_loc = np.array(valid_p, dtype=int)

        valid_v = []
        for v in v_loc:
            if axis_data[v] <= a_mean:
                valid_v.append(v)
        v_loc = np.array(valid_v, dtype=int)

        # 步骤3: 合并相邻同向极值点
        p_loc = self.merge_close_poles(axis_data, p_loc, v_loc, 1)
        v_loc = self.merge_close_poles(axis_data, v_loc, p_loc, -1)

        # 确保第一个是波谷
        if len(p_loc) > 0:
            if len(v_loc) == 0 or (len(v_loc) > 0 and p_loc[0] < v_loc[0]):
                if len(p_loc) > 1:
                    p_loc = p_loc[1:]
                else:
                    p_loc = np.array([], dtype=int)

        # 步骤4: 去除非对称峰值
        if len(v_loc) >= 2:
            valid_peaks = []
            for p_idx in p_loc:
                # 寻找左右最近的波谷
                left_v = v_loc[v_loc < p_idx]
                right_v = v_loc[v_loc > p_idx]

                if len(left_v) == 0 or len(right_v) == 0:
                    continue

                lv = left_v[-1]
                rv = right_v[0]

                # 高度和时间检查
                lh = abs(axis_data[p_idx] - axis_data[lv])
                rh = abs(axis_data[p_idx] - axis_data[rv])
                lt = abs(p_idx - lv)
                rt = abs(p_idx - rv)

                if (lh > self.PEAK_VALLEY_DIFFERENCE and
                    rh > self.PEAK_VALLEY_DIFFERENCE and
                    lh > rh / 2 and lh < rh * 2 and
                    lt >= self.TIME_THRESHOLD1 and lt <= self.TIME_THRESHOLD2 and
                    rt >= self.TIME_THRESHOLD1 and rt <= self.TIME_THRESHOLD2):
                    valid_peaks.append(p_idx)

            return len(valid_peaks) * 2  # 每个峰值 = 2步

        return 0

    def classify_activity(self, acc_mag_window, gyro_mag_window, acc_x_win, acc_y_win, acc_z_win):
        """
        使用6轴数据分类活动类型
        返回: 'stationary', 'walk', 'run', 'noise'
        """
        acc_detrend = acc_mag_window - np.mean(acc_mag_window)
        acc_std = np.std(acc_detrend)
        acc_range = np.max(acc_mag_window) - np.min(acc_mag_window)

        # 静止检测
        if acc_std < self.STATIONARY_ACC_STD:
            return 'stationary'

        # 极低运动量 - 不太可能有有效步数
        if acc_range < self.ACC_GRAVITY // 5:  # <0.2G
            return 'stationary'

        # 陀螺仪分析
        gyro_energy = np.sum(gyro_mag_window**2) / len(gyro_mag_window)
        gyro_std = np.std(gyro_mag_window)

        # 加速度变化分析
        acc_motion_level = acc_std / self.ACC_GRAVITY  # 归一化到G单位

        # 节律性分析
        rhythm_score, dominant_period = compute_rhythm_features(acc_mag_window)

        # === 决策逻辑 ===

        # 跑步判定: 高加速度变化 + 较高频率节律
        if acc_std > self.ACC_GRAVITY * 1.2:  # >1.2G std
            if rhythm_score > 0.12 or gyro_energy > 100000:
                return 'run'

        # 非常高的加速度变化但无节律 → 可能是剧烈噪声
        if acc_std > self.ACC_GRAVITY * 2.0 and rhythm_score < 0.08:
            return 'noise'

        # 步行判定基础条件
        is_significant = acc_range > self.ACC_GRAVITY // 3  # >0.33G

        if not is_significant:
            # 小幅运动但有节律 → 也可能是步行（如手插口袋走路）
            if rhythm_score > 0.25:
                return 'walk'
            return 'noise'

        # 有节律的运动 → 步行
        if rhythm_score > 0.15:
            return 'walk'

        # 有一定陀螺仪能量 + 足够的加速度变化 → 可能是步行
        if gyro_energy > 100000 and acc_std > self.ACC_GRAVITY // 4:
            return 'walk'

        # 加速度变化大但无节律且陀螺仪能量低 → 噪声
        # (如切菜：有加速度变化但无规律摆臂)
        if gyro_energy < 50000 and rhythm_score < 0.12:
            if acc_std < self.ACC_GRAVITY * 1.5:
                return 'noise'

        # 默认: 有一定运动量，可能是步行
        if acc_std > self.ACC_GRAVITY // 2.5:
            return 'walk'

        return 'noise'

    def count_steps_in_window(self, acc_x_win, acc_y_win, acc_z_win,
                               acc_mag_win, activity_type):
        """
        在窗口内统计步数
        使用原始3轴峰值检测 + acc_mag峰值检测的融合策略
        """
        if activity_type in ['stationary', 'noise']:
            return 0

        # 检查信号变化是否足够
        acc_range = np.max(acc_mag_win) - np.min(acc_mag_win)
        if acc_range < self.STEP_ACC_DIFF_THRESHOLD:
            return 0

        # 方法1: 3轴独立检测取中位数（原始方法）
        axis_steps = []
        for axis_data in [acc_x_win, acc_y_win, acc_z_win]:
            steps = self.count_steps_original_method(axis_data)
            axis_steps.append(steps)

        # 取中位数（最可靠的轴）
        axis_steps_sorted = sorted(axis_steps)
        steps_from_axes = axis_steps_sorted[1]  # 中位数

        # 方法2: acc_mag直接检测
        mag_detrend = acc_mag_win - np.mean(acc_mag_win)
        steps_from_mag = self.count_steps_original_method(mag_detrend)

        # 融合策略:
        # - 跑步时优先使用mag（因为跑步各轴幅度都大）
        # - 走路时优先使用3轴中位数（更敏感）
        if activity_type == 'run':
            # 跑步各轴都有显著信号，取较大值
            steps = max(steps_from_axes, steps_from_mag)
        else:
            # 走路时3轴方法更可靠
            if steps_from_axes > 0:
                steps = steps_from_axes
            else:
                steps = steps_from_mag

        return steps

    def process_file(self, filepath, verbose=False):
        """处理单个数据文件"""
        fname = os.path.basename(filepath)
        match = re.search(r'step(\d+)', fname)
        true_steps = int(match.group(1)) if match else 0

        # 加载数据
        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = self.load_imu_data(filepath)
        total_samples = len(acc_x)

        if total_samples < self.fs:
            return 0, true_steps

        # 预处理
        acc_x_f, acc_y_f, acc_z_f, acc_mag = self.preprocess(acc_x, acc_y, acc_z)
        gyro_mag = self.preprocess_gyro(gyro_x, gyro_y, gyro_z)

        delay_point = self.m1 // 2 + self.m2 // 2

        total_steps = 0

        # 滑动窗口
        win_x = np.zeros(self.win_len)
        win_y = np.zeros(self.win_len)
        win_z = np.zeros(self.win_len)
        win_mag = np.zeros(self.win_len)
        win_gyro = np.zeros(self.win_len)
        win_cnt = 0

        buf_x = np.zeros(self.buf_len)
        buf_y = np.zeros(self.buf_len)
        buf_z = np.zeros(self.buf_len)
        buf_mag = np.zeros(self.buf_len)
        buf_gyro = np.zeros(self.buf_len)
        buf_cnt = 0

        for i in range(self.fs + self.fs - delay_point,
                       len(acc_mag) - delay_point + 1, self.fs):
            new_data = self.fs
            seg_end = min(i, len(acc_mag) - delay_point)
            seg_start = max(0, seg_end - self.fs)

            seg_x = acc_x_f[seg_start:seg_end]
            seg_y = acc_y_f[seg_start:seg_end]
            seg_z = acc_z_f[seg_start:seg_end]
            seg_mag = acc_mag[seg_start:seg_end]
            seg_gyro = gyro_mag[seg_start:seg_end]

            seg_len = len(seg_x)

            if win_cnt + seg_len <= self.win_len:
                win_x[win_cnt:win_cnt + seg_len] = seg_x
                win_y[win_cnt:win_cnt + seg_len] = seg_y
                win_z[win_cnt:win_cnt + seg_len] = seg_z
                win_mag[win_cnt:win_cnt + seg_len] = seg_mag
                win_gyro[win_cnt:win_cnt + seg_len] = seg_gyro
                win_cnt += seg_len
            else:
                shift = seg_len
                win_x[:self.win_len - shift] = win_x[shift:]
                win_y[:self.win_len - shift] = win_y[shift:]
                win_z[:self.win_len - shift] = win_z[shift:]
                win_mag[:self.win_len - shift] = win_mag[shift:]
                win_gyro[:self.win_len - shift] = win_gyro[shift:]
                win_x[self.win_len - shift:] = seg_x
                win_y[self.win_len - shift:] = seg_y
                win_z[self.win_len - shift:] = seg_z
                win_mag[self.win_len - shift:] = seg_mag
                win_gyro[self.win_len - shift:] = seg_gyro
                win_cnt = self.win_len

            if win_cnt >= self.win_len:
                # 合并缓冲区和窗口数据
                if buf_cnt > 0:
                    full_x = np.concatenate([buf_x[:buf_cnt], win_x[:win_cnt]])
                    full_y = np.concatenate([buf_y[:buf_cnt], win_y[:win_cnt]])
                    full_z = np.concatenate([buf_z[:buf_cnt], win_z[:win_cnt]])
                    full_mag = np.concatenate([buf_mag[:buf_cnt], win_mag[:win_cnt]])
                    full_gyro = np.concatenate([buf_gyro[:buf_cnt], win_gyro[:win_cnt]])
                else:
                    full_x = win_x[:win_cnt].copy()
                    full_y = win_y[:win_cnt].copy()
                    full_z = win_z[:win_cnt].copy()
                    full_mag = win_mag[:win_cnt].copy()
                    full_gyro = win_gyro[:win_cnt].copy()

                # 活动分类
                activity_type = self.classify_activity(
                    full_mag, full_gyro, full_x, full_y, full_z)

                # 步数统计
                steps = self.count_steps_in_window(
                    full_x, full_y, full_z, full_mag, activity_type)

                if verbose:
                    print(f"  time={seg_end//self.fs:3d}s  activity={activity_type:10s}  "
                          f"steps={steps:3d}  acc_std={np.std(full_mag-np.mean(full_mag)):.0f}")

                total_steps += steps

                # 保存缓冲区
                buf_cnt = min(self.buf_len, len(full_mag) - self.win_len)
                if buf_cnt > 0:
                    buf_x[:buf_cnt] = full_x[-buf_cnt:]
                    buf_y[:buf_cnt] = full_y[-buf_cnt:]
                    buf_z[:buf_cnt] = full_z[-buf_cnt:]
                    buf_mag[:buf_cnt] = full_mag[-buf_cnt:]
                    buf_gyro[:buf_cnt] = full_gyro[-buf_cnt:]

                win_cnt = 0

        if verbose:
            print(f"  文件: {fname}")
            print(f"  预测步数: {total_steps}, 真实步数: {true_steps}")

        return total_steps, true_steps

    def evaluate_all(self, data_dir, verbose=False):
        """评估所有数据文件"""
        results = {'walk': [], 'run': [], 'others': []}

        for category in ['walk', 'run', 'others']:
            cat_dir = os.path.join(data_dir, category)
            if not os.path.exists(cat_dir):
                continue
            files = sorted(glob.glob(os.path.join(cat_dir, '*.txt')))

            for f in files:
                predicted, true = self.process_file(f, verbose=verbose)
                results[category].append({
                    'file': os.path.basename(f),
                    'predicted': predicted,
                    'true': true,
                    'error': predicted - true,
                    'abs_error': abs(predicted - true)
                })

            print(f"  已完成 {category}: {len(files)} 个文件")

        return results

    def compute_metrics(self, results):
        """计算并打印评估指标"""
        print("\n" + "=" * 70)
        print("评估结果汇总")
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

            # MAPE: only for true > 0
            non_zero = [(abs(p - t) / t * 100) for p, t in zip(pred_steps, true_steps) if t > 0]
            mape = np.mean(non_zero) if non_zero else float('nan')
            accuracy = max(0, 100 - mape)

            if '噪声' in name:
                fp_rate = sum(1 for p in pred_steps if p > 0) / len(pred_steps) * 100
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE: {mae:.1f} 步")
                print(f"  误检率 (预测>0): {fp_rate:.1f}%")
            else:
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE: {mae:.1f} 步")
                print(f"  MAPE: {mape:.1f}%")
                print(f"  准确率: {accuracy:.1f}%")

        # 详细输出
        print("\n" + "-" * 70)
        print("详细结果 (步行+跑步):")
        for d in walk_run:
            ape = abs(d['error']) / max(1, d['true']) * 100
            print(f"  {d['file']:55s}  预测={d['predicted']:4d}  真实={d['true']:4d}  "
                  f"误差={d['error']:+4d}  APE={ape:5.1f}%")

        # 噪声误检详情
        noise_errors = [d for d in results['others'] if d['predicted'] > 0]
        if noise_errors:
            print(f"\n噪声误检文件 ({len(noise_errors)}/{len(results['others'])}):")
            for d in noise_errors:
                print(f"  {d['file']:55s}  预测={d['predicted']:4d}")

        return mae, mape


def main():
    data_dir = 'VeriHealthi_Algorithm_Homework_Code_Data/AccData'

    print("改进的计步算法 V2 - 原始峰值检测 + 6轴噪声过滤")
    print("=" * 70)

    counter = ImprovedStepCounterV2()

    # 演示文件
    demo_file = os.path.join(data_dir, 'walk',
                             'IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt')
    print(f"\n演示文件: {os.path.basename(demo_file)}")
    print("-" * 40)
    predicted, true = counter.process_file(demo_file, verbose=True)

    # 批量评估
    print("\n\n批量评估所有数据...")
    results = counter.evaluate_all(data_dir)
    mae, mape = counter.compute_metrics(results)

    return results


if __name__ == '__main__':
    main()
