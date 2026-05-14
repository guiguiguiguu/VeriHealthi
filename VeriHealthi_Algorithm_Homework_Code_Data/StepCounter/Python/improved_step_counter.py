#!/usr/bin/python
"""
改进的计步算法 V4 - 原始算法 + 6轴置信度评分

核心策略:
1. 步数检测: 完全使用原始算法（3轴峰值检测）
2. 噪声过滤: 计算6轴特征 → 生成置信度系数(0~1)
3. 最终步数 = 原始步数 × 置信度
4. 走路/跑步: 置信度应接近1.0
5. 噪声: 置信度应接近0.0

这种软决策比硬决策更稳健，边际情况有部分分数。
"""

import numpy as np
import os
import glob
import re


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
    for b in B:
        if b == a:
            return 1
    return 0


def delete_ith_a(a, i):
    a = np.array(a)
    if len(a) <= 1 and i == 0:
        return np.zeros(0)
    elif i < 0 or i > len(a) - 1:
        return a
    else:
        if i == 0:
            return a[1:]
        elif i == len(a) - 1:
            return a[:-1]
        else:
            return np.concatenate((a[:i], a[i + 1:]))


class ActionProcessor:
    """原始算法的峰值检测器（完全不变）"""
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
        self.p_loc = self.p_loc[:self.p_cnt]
        self.v_loc = self.v_loc[:self.v_cnt]
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
                r, j = 0, pole_locs[i - 1]
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
        T1, T2 = 4, 40
        PVD = 4096 // 14
        if self.v_cnt <= 1:
            return self
        last_valley_loc, i = 0, 0
        while i < self.p_cnt:
            r1, r2, j = 0, 0, 1
            while j < self.v_cnt:
                if self.v_loc[j - 1] < self.p_loc[i] and self.v_loc[j] > self.p_loc[i]:
                    r1 = 1
                    h1 = abs(a[self.p_loc[i]] - a[self.v_loc[j - 1]])
                    h2 = abs(a[self.p_loc[i]] - a[self.v_loc[j]])
                    t1 = abs(self.p_loc[i] - self.v_loc[j - 1])
                    t2 = abs(self.p_loc[i] - self.v_loc[j])
                    if (h1 > PVD and h2 > PVD and h1 > h2 / 2 and h1 < h2 * 2 and
                        t1 >= T1 and t1 <= T2 and t2 >= T1 and t2 <= T2):
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


class ImprovedStepCounterV4:
    """改进的计步器 V4: 原始步数检测 + 6轴置信度"""

    def __init__(self):
        self.fs = 50
        self.m1, self.m2 = 15, 7
        self.win_sec, self.buf_sec = 5, 3
        self.win_len = self.win_sec * self.fs  # 250
        self.buf_len = self.buf_sec * self.fs  # 150
        self.STEP_ACC_DIFF_THRESHOLD = 4096 // 10
        self.PEAK_VALLEY_DIFFERENCE = 4096 // 14
        self.LEFT_DATA_NUM = 2
        self.G = 4096  # 1G

        # 用于计算全局特征的累积统计
        self.global_acc_mag_std = 0
        self.global_gyro_energy = 0
        self.window_count = 0

    def load_imu_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = np.array([int(line.strip()) for line in lines[5:]], dtype=np.float64)
        cnt = len(data) // 7 * 7
        data = data[:cnt].reshape(-1, 7)
        return (data[:, 0], data[:, 1], data[:, 2],
                data[:, 3], data[:, 4], data[:, 5])

    def compute_confidence(self, acc_mag_window, gyro_mag_window):
        """
        计算步行置信度 (0~1)
        基于6轴数据特征判断当前窗口是真实步行的概率

        使用三个核心特征:
        1. 节律性 (自相关峰值)
        2. 加速度变化量
        3. 陀螺仪活动验证
        """
        acc_detrend = acc_mag_window - np.mean(acc_mag_window)
        acc_std = np.std(acc_detrend)
        acc_range = np.max(acc_mag_window) - np.min(acc_mag_window)

        gyro_energy = np.sum(gyro_mag_window**2) / max(1, len(gyro_mag_window))

        # === 特征1: 节律性评分 (0~1) ===
        rhythm_score = 0.0
        if len(acc_mag_window) >= 50:
            autocorr = compute_autocorrelation(acc_mag_window,
                                               min(len(acc_mag_window) // 2, 150))
            # 搜索步频范围: 0.5-5Hz → 10-100采样点 @50Hz
            search_start = max(1, min(10, len(autocorr) - 1))
            search_end = min(100, len(autocorr) - 1)
            if search_end > search_start:
                search_range = autocorr[search_start:search_end]
                ac_peak = np.max(search_range)
                # 将自相关峰值映射到0~1
                # 经验上: <0.05 = 噪声, 0.05-0.15 = 可疑, >0.15 = 步行
                rhythm_score = min(1.0, max(0.0, (ac_peak - 0.05) / 0.30))

        # === 特征2: 运动量评分 (0~1) ===
        # 步行滤波后 acc_std 通常在 200-800 范围（窗口级）
        motion_score = 0.0
        if acc_std < 30:        # 完全静止
            motion_score = 0.0
        elif acc_std < 80:      # 微动: 极低概率是步行
            motion_score = 0.05 + (acc_std - 30) / 50 * 0.10
        elif acc_std < 200:     # 低运动: 有一定概率
            motion_score = 0.15 + (acc_std - 80) / 120 * 0.40
        elif acc_std < 600:     # 典型步行范围: 高概率
            motion_score = 0.55 + (acc_std - 200) / 400 * 0.35
        else:                   # 高运动量 (跑步或剧烈运动)
            motion_score = 0.90 + min(0.10, (acc_std - 600) / 3000 * 0.10)

        # === 特征3: 陀螺仪验证评分 (0~1) ===
        # 真正的步行通常伴随摆臂动作（陀螺仪活动）
        # 步行: gyro_energy 通常在 500K-8M
        gyro_score = 0.0
        if gyro_energy < 2000:      # 几乎没有陀螺仪活动
            gyro_score = 0.0
        elif gyro_energy < 20000:   # 极轻微
            gyro_score = gyro_energy / 20000 * 0.15
        elif gyro_energy < 100000:  # 轻微
            gyro_score = 0.15 + (gyro_energy - 20000) / 80000 * 0.30
        elif gyro_energy < 1000000: # 中等偏低
            gyro_score = 0.45 + (gyro_energy - 100000) / 900000 * 0.30
        elif gyro_energy < 10000000: # 典型步行范围
            gyro_score = 0.75 + (gyro_energy - 1000000) / 9000000 * 0.20
        else:                       # 极高陀螺仪活动
            gyro_score = 0.95

        # === 特征4: 陀螺仪加速度一致性 ===
        # 真正的步行: 加速度和陀螺仪活动都在典型范围内
        # 噪声特征1: 陀螺仪极高但加速度正常 (只有手臂在动)
        # 噪声特征2: 加速度正常但陀螺仪极低 (身体动但手臂不动 — 也可能是真步行)
        consistency_score = 1.0

        # 归一化陀螺仪能量 (映射到合理范围)
        gyro_energy_norm = min(1.0, gyro_energy / 10000000)

        # 检测异常模式: 陀螺仪远大于加速度暗示的运动量
        # 使用运动评分作为代理
        if motion_score < 0.3 and gyro_score > 0.7:
            # 加速度很小但陀螺仪很大 → 只有手臂在动
            consistency_score = 0.1
        elif motion_score < 0.2 and gyro_score > 0.5:
            consistency_score = 0.3
        elif gyro_score > 0.8 and motion_score > 0.4:
            # 两者都高 → 步行/跑步的典型特征
            consistency_score = 1.0
        elif gyro_score < 0.2 and motion_score > 0.4:
            # 加速度足够但陀螺仪低 → 可能是未摆臂的步行
            consistency_score = 0.6

        # === 综合置信度 ===
        # 加权组合: 节律性为主要权重
        confidence = (rhythm_score * 0.50 +
                      motion_score * 0.30 +
                      gyro_score * 0.10 +
                      consistency_score * 0.10)

        # 安全检查: 如果运动量太低,直接归零
        if acc_range < self.G // 10:  # <0.1G
            confidence = 0.0

        # 对明确的步行模式提高基准置信度
        # 避免过度惩罚真实步行
        if rhythm_score > 0.6 and motion_score > 0.5:
            confidence = min(1.0, confidence + 0.10)
        if rhythm_score > 0.8 and motion_score > 0.55:
            confidence = min(1.0, confidence + 0.05)

        return confidence, {
            'rhythm': rhythm_score,
            'motion': motion_score,
            'gyro': gyro_score,
            'ratio': consistency_score,
            'acc_std': acc_std,
            'gyro_energy': gyro_energy,
        }

    def process_axis_original(self, axis_data, buf_data, buf_cnt):
        """使用原始算法处理单个轴"""
        if buf_cnt > 0:
            full_data = np.concatenate([buf_data[:buf_cnt], axis_data])
        else:
            full_data = axis_data

        result_steps = 0
        new_buf = np.zeros(self.buf_len)
        new_buf_cnt = 0

        tmp_diff = np.max(full_data) - np.min(full_data)

        if tmp_diff > self.STEP_ACC_DIFF_THRESHOLD:
            processor = ActionProcessor(250, 250)
            processor.find_possible_peak_valley(full_data)
            processor.remove_false_peak_valley(full_data)
            processor.merge_close_peaks_valleys(full_data)
            processor.remove_asymmetric_peaks(full_data)

            result_steps = processor.p_cnt

            if processor.v_cnt >= 1:
                last_v = processor.v_loc[processor.v_cnt - 1]
                left_len = len(full_data) - int(last_v) + self.LEFT_DATA_NUM
                if 0 < left_len < self.buf_len:
                    new_buf[:left_len] = full_data[-left_len:]
                    new_buf_cnt = left_len

        return result_steps, new_buf, new_buf_cnt

    def process_file(self, filepath, verbose=False):
        """处理单个文件"""
        fname = os.path.basename(filepath)
        match = re.search(r'step(\d+)', fname)
        true_steps = int(match.group(1)) if match else 0

        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = self.load_imu_data(filepath)
        total_samples = len(acc_x)

        if total_samples < self.fs:
            return 0, true_steps

        # 加速度预处理
        acc_data = np.zeros((total_samples, 4))
        acc_data[:, 0], acc_data[:, 1], acc_data[:, 2] = acc_x, acc_y, acc_z

        acc_xyz_mean = np.zeros((total_samples, 4))
        for j in range(3):
            tmp = func_calculation(acc_data[:, j], self.m1, np.mean).astype(int)
            tmp = func_calculation(tmp, self.m2, np.mean).astype(int)
            acc_xyz_mean[:, j] = tmp
        acc_xyz_mean[:, 3] = np.sqrt(
            acc_xyz_mean[:, 0]**2 + acc_xyz_mean[:, 1]**2 + acc_xyz_mean[:, 2]**2)

        # 陀螺仪预处理
        gyro_filt = np.zeros((total_samples, 3))
        for j, g in enumerate([gyro_x, gyro_y, gyro_z]):
            gyro_filt[:, j] = func_calculation(g, 7, np.mean)

        # 窗口处理
        win_len = self.win_len
        buf_len = self.buf_len
        xyz_win = np.zeros((win_len, 3))
        gyro_win = np.zeros((win_len, 3))
        win_cnt = 0
        xyz_buf = np.zeros((buf_len, 3))
        buf_cnt = np.zeros(3, dtype=int)
        total_steps = 0.0
        delay_point = self.m1 // 2 + self.m2 // 2

        for i in range(self.fs + self.fs - delay_point,
                       total_samples - delay_point + 1, self.fs):
            seg = acc_xyz_mean[i - self.fs:i, 0:3]
            gseg = gyro_filt[i - self.fs:i, :]

            if win_cnt + self.fs <= win_len:
                xyz_win[win_cnt:win_cnt + self.fs, :] = seg
                gyro_win[win_cnt:win_cnt + self.fs, :] = gseg
                win_cnt += self.fs
            else:
                xyz_win = np.concatenate((xyz_win[self.fs:win_len, :], seg))
                gyro_win = np.concatenate((gyro_win[self.fs:win_len, :], gseg))
                win_cnt = win_len

            if win_cnt >= win_len:
                # 完整窗口数据
                full_xyz = xyz_win[:win_cnt].copy()
                full_gyro = gyro_win[:win_cnt].copy()

                # 计算置信度
                acc_mag = np.sqrt(np.sum(full_xyz**2, axis=1))
                gyro_mag = np.sqrt(np.sum(full_gyro**2, axis=1))
                confidence, feat = self.compute_confidence(acc_mag, gyro_mag)

                # 原始算法步数检测
                action_num = np.zeros(3, dtype=int)
                for j in range(3):
                    s, nb, nc = self.process_axis_original(
                        full_xyz[:, j],
                        xyz_buf[:int(buf_cnt[j]), j],
                        int(buf_cnt[j]))
                    action_num[j] = s
                    xyz_buf[:buf_len, j] = 0
                    xyz_buf[:nc, j] = nb[:nc]
                    buf_cnt[j] = nc

                raw_steps = int(np.median(action_num)) * 2
                adjusted_steps = raw_steps * confidence

                total_steps += adjusted_steps

                if verbose:
                    print(f"  t={i//self.fs:3d}s  raw={raw_steps:3d}  conf={confidence:.2f}  "
                          f"adj={adjusted_steps:5.1f}  "
                          f"r={feat['rhythm']:.2f} m={feat['motion']:.2f} "
                          f"g={feat['gyro']:.2f} cs={feat['ratio']:.2f}")

                xyz_win = np.zeros((win_len, 3))
                gyro_win = np.zeros((win_len, 3))
                win_cnt = 0

        total_steps_int = int(round(total_steps))

        if verbose:
            print(f"  文件: {fname}  预测: {total_steps_int}  真实: {true_steps}")

        return total_steps_int, true_steps

    def evaluate_all(self, data_dir, verbose=False):
        results = {'walk': [], 'run': [], 'others': []}

        for category in ['walk', 'run', 'others']:
            cat_dir = os.path.join(data_dir, category)
            if not os.path.exists(cat_dir):
                continue
            files = sorted(glob.glob(os.path.join(cat_dir, '*.txt')))
            for idx, f in enumerate(files):
                p, t = self.process_file(f, verbose=verbose)
                results[category].append({
                    'file': os.path.basename(f),
                    'predicted': p, 'true': t,
                    'error': p - t, 'abs_error': abs(p - t)
                })
                if (idx + 1) % 20 == 0:
                    print(f"  {category}: {idx+1}/{len(files)}")
            print(f"  已完成 {category}: {len(files)} 文件")

        return results

    def compute_metrics(self, results):
        print("\n" + "=" * 70)
        print("V4 评估结果 - 原始算法 + 6轴置信度")
        print("=" * 70)

        walk_run = results['walk'] + results['run']

        for name, data in [('步行(walk+run)', walk_run), ('走路', results['walk']),
                           ('跑步', results['run']), ('噪声', results['others'])]:
            if not data:
                continue
            preds = [d['predicted'] for d in data]
            trues = [d['true'] for d in data]
            maes = [d['abs_error'] for d in data]
            mae = np.mean(maes)
            non_zero = [(abs(p - t) / t * 100) for p, t in zip(preds, trues) if t > 0]
            mape = np.mean(non_zero) if non_zero else float('nan')

            if '噪声' in name:
                fp = sum(1 for p in preds if p > 0) / len(preds) * 100
                print(f"\n{name}: MAE={mae:.1f}  误检率={fp:.1f}%")
            else:
                print(f"\n{name}: MAE={mae:.1f}  MAPE={mape:.1f}%  准确率={max(0,100-mape):.1f}%")

        print("\n" + "-" * 70)
        print("详细结果 (步行+跑步):")
        for d in walk_run:
            ape = abs(d['error']) / max(1, d['true']) * 100
            print(f"  {d['file']:55s} pred={d['predicted']:4d} true={d['true']:4d} "
                  f"err={d['error']:+4d} APE={ape:5.1f}%")

        noise_fp = [d for d in results['others'] if d['predicted'] > 0]
        print(f"\n噪声误检 ({len(noise_fp)}/{len(results['others'])}):")
        for d in noise_fp:
            print(f"  {d['file']:55s} pred={d['predicted']:4d}")

        return mae, mape


def main():
    data_dir = 'VeriHealthi_Algorithm_Homework_Code_Data/AccData'

    print("改进的计步算法 V4 - 原始算法 + 6轴置信度评分")
    print("=" * 70)

    counter = ImprovedStepCounterV4()

    demo = os.path.join(data_dir, 'walk',
                        'IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt')
    print(f"\n演示: {os.path.basename(demo)}")
    print("-" * 40)
    counter.process_file(demo, verbose=True)

    print("\n\n批量评估...")
    results = counter.evaluate_all(data_dir)
    counter.compute_metrics(results)


if __name__ == '__main__':
    main()
