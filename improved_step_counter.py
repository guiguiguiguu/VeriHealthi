#!/usr/bin/python
"""
改进的计步算法 - 基于6轴IMU数据（3轴加速度计 + 3轴陀螺仪）

核心改进:
1. 使用6轴数据（原始算法仅用3轴加速度）
2. 使用加速度幅度 (orientation-independent) 进行步数检测
3. 加入基于自相关的节律性检测，过滤非周期噪声
4. 加入陀螺仪能量验证，识别真正的步行摆臂动作
5. 动态阈值替代固定阈值
6. 区分走路和跑步的不同计步参数
"""

import numpy as np
import os
import glob
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


def compute_rhythm_score(acc_mag):
    """
    计算节律性评分
    基于自相关的峰值特征来判断运动是否有节律
    返回值: 0-1之间的分数，越高越有节律
    """
    max_lag = min(len(acc_mag) // 2, 250)  # 最多5秒的滞后
    if max_lag < 20:
        return 0.0
    autocorr = compute_autocorrelation(acc_mag, max_lag)

    # 在可能的步频范围内查找自相关峰值 (0.5-5 Hz, 即10-100采样点 @50Hz)
    min_lag = 10   # 200ms (5步/秒，对应跑步)
    max_lag_rhythm = 100  # 2000ms (0.5步/秒，对应慢走)

    if len(autocorr) <= min_lag:
        return 0.0

    search_range = autocorr[min_lag:min(max_lag_rhythm, len(autocorr))]
    if len(search_range) == 0:
        return 0.0

    peak_val = np.max(search_range)
    # 检查自相关函数是否存在明显的周期性峰值
    # 使用峰值和附近值的对比来评估
    if peak_val > 0.15:
        # 寻找峰值位置
        peak_lag = min_lag + np.argmax(search_range)
        # 检查是否存在多个峰值（真正的节律运动有多重自相关峰）
        if peak_lag * 2 < len(autocorr):
            second_peak = np.max(autocorr[peak_lag * 2 - 5:min(peak_lag * 2 + 5, len(autocorr))])
            if second_peak > peak_val * 0.4:
                return min(peak_val * 1.5, 1.0)

    return peak_val


class ImprovedStepCounter:
    """改进的计步器"""

    def __init__(self):
        # 采样率
        self.fs = 50  # Hz

        # 预处理滤波器参数
        self.m1 = 15  # 第一级均值滤波窗口
        self.m2 = 7   # 第二级均值滤波窗口

        # 窗口参数
        self.win_sec = 5    # 分析窗口（秒）
        self.buf_sec = 3    # 缓冲区（秒）
        self.win_len = self.win_sec * self.fs   # 250 样本
        self.buf_len = self.buf_sec * self.fs   # 150 样本

        # 加速度传感器参数
        self.ACC_GRAVITY = 4096  # 1G对应的ADC值

        # 动态阈值相关参数
        self.STATIONARY_ACC_STD = self.ACC_GRAVITY // 20  # ~0.05G, 静止判定
        self.MIN_WALK_ACC_STD = self.ACC_GRAVITY // 10    # ~0.1G, 最小步行加速度变化
        self.PEAK_HEIGHT_RATIO = 0.35   # 峰值至少为信号标准差的这个比例

        # 节律性检测
        self.MIN_RHYTHM_SCORE = 0.18    # 最小节律性评分

        # 陀螺仪验证参数
        self.GYRO_SCALE = 16.4  # 陀螺仪分辨率: 16.4/(1°/s)
        self.MIN_GYRO_ENERGY_THRESHOLD = 800   # 最小陀螺仪能量（用于步行验证）

        # 步数检测的时间约束
        self.MIN_STEP_INTERVAL_WALK = 15   # 300ms @50Hz (最多~3.3步/秒)
        self.MAX_STEP_INTERVAL_WALK = 120  # 2400ms @50Hz (最少~0.4步/秒)
        self.MIN_STEP_INTERVAL_RUN = 8     # 160ms @50Hz (最多~6步/秒)
        self.MAX_STEP_INTERVAL_RUN = 80    # 1600ms @50Hz (最少~0.6步/秒)

        # 左右缓冲区数据
        self.LEFT_DATA_NUM = 2

        # 峰值波谷差异阈值
        self.PEAK_VALLEY_MIN_DIFF = self.ACC_GRAVITY // 14  # 与原版一致

    def load_imu_data(self, filepath):
        """加载6轴IMU数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = np.array([int(line.strip()) for line in lines[5:]], dtype=np.float64)
        cnt = len(data) // 7 * 7
        data = data[:cnt].reshape(-1, 7)

        gyro_x = data[:, 0]
        gyro_y = data[:, 1]
        gyro_z = data[:, 2]
        acc_x = data[:, 3]
        acc_y = data[:, 4]
        acc_z = data[:, 5]

        return gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z

    def preprocess_acceleration(self, acc_x, acc_y, acc_z):
        """预处理加速度数据：两级均值滤波 + 计算幅度"""
        # 对每个轴进行两级均值滤波
        acc_x_filt = func_calculation(acc_x, self.m1, np.mean).astype(np.float64)
        acc_x_filt = func_calculation(acc_x_filt, self.m2, np.mean)

        acc_y_filt = func_calculation(acc_y, self.m1, np.mean).astype(np.float64)
        acc_y_filt = func_calculation(acc_y_filt, self.m2, np.mean)

        acc_z_filt = func_calculation(acc_z, self.m1, np.mean).astype(np.float64)
        acc_z_filt = func_calculation(acc_z_filt, self.m2, np.mean)

        # 计算加速度幅度（与方向无关）
        acc_mag = np.sqrt(acc_x_filt**2 + acc_y_filt**2 + acc_z_filt**2)

        return acc_x_filt, acc_y_filt, acc_z_filt, acc_mag

    def preprocess_gyroscope(self, gyro_x, gyro_y, gyro_z):
        """预处理陀螺仪数据：轻度滤波 + 计算幅度"""
        # 对陀螺仪数据轻度滤波（使用较小的均值窗口）
        gyro_x_filt = func_calculation(gyro_x, 7, np.mean)
        gyro_y_filt = func_calculation(gyro_y, 7, np.mean)
        gyro_z_filt = func_calculation(gyro_z, 7, np.mean)

        # 计算陀螺仪幅度
        gyro_mag = np.sqrt(gyro_x_filt**2 + gyro_y_filt**2 + gyro_z_filt**2)

        return gyro_mag

    def detect_activity_type(self, acc_mag_window, gyro_mag_window):
        """
        检测当前窗口的活动类型
        返回值: 'stationary', 'walk', 'run', 'noise'
        """
        if len(acc_mag_window) < 50:
            return 'stationary'

        acc_mag_detrend = acc_mag_window - np.mean(acc_mag_window)
        acc_std = np.std(acc_mag_detrend)
        acc_range = np.max(acc_mag_window) - np.min(acc_mag_window)

        # 静止检测
        if acc_std < self.STATIONARY_ACC_STD:
            return 'stationary'

        # 加速度变化太小，不太可能是有效的步行
        if acc_std < self.MIN_WALK_ACC_STD:
            return 'noise'

        # 计算陀螺仪能量（归一化）
        gyro_energy = np.sum(gyro_mag_window**2) / len(gyro_mag_window)

        # 走路/跑步的重要特征：加速度有足够变化
        is_significant_motion = (acc_range > self.ACC_GRAVITY // 3)  # >0.33G变化

        if not is_significant_motion:
            return 'noise'

        # 计算节律性评分
        rhythm_score = compute_rhythm_score(acc_mag_detrend)

        # 步行判定：需要一定节律性
        # 跑步的加速度变化更大，频率更高
        if rhythm_score > self.MIN_RHYTHM_SCORE or acc_std > self.ACC_GRAVITY // 2:
            if acc_std > self.ACC_GRAVITY * 1.5:  # 跑步特征：加速度变化很大
                return 'run'
            else:
                return 'walk'

        # 某些噪声虽然有加速度变化但缺乏节律性
        if gyro_energy > 100000 and rhythm_score > 0.1:
            return 'walk'  # 有一定陀螺仪能量和节律，可能是步行

        return 'noise'

    def find_peaks_valleys(self, signal):
        """寻找信号中的波峰和波谷"""
        p_loc = []
        v_loc = []
        for i in range(1, len(signal) - 1):
            if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]:
                p_loc.append(i)
            if signal[i] <= signal[i - 1] and signal[i] < signal[i + 1]:
                v_loc.append(i)
        return np.array(p_loc, dtype=int), np.array(v_loc, dtype=int)

    def merge_close_peaks(self, signal, p_loc, v_loc):
        """合并相邻且之间没有波谷的波峰（保留较高的）"""
        if len(p_loc) <= 1:
            return p_loc
        i = 1
        while i < len(p_loc):
            has_valley = False
            for v in v_loc:
                if p_loc[i - 1] < v < p_loc[i]:
                    has_valley = True
                    break
            if not has_valley:
                if signal[p_loc[i - 1]] > signal[p_loc[i]]:
                    p_loc = np.delete(p_loc, i)
                else:
                    p_loc = np.delete(p_loc, i - 1)
            else:
                i += 1
        return p_loc

    def remove_false_peaks(self, signal, p_loc, v_loc, activity_type):
        """去除不满足条件的伪峰值"""
        if len(p_loc) == 0 or len(v_loc) < 2:
            return np.array([], dtype=int)

        signal_mean = np.mean(signal)
        signal_std = np.std(signal)

        # 动态阈值：峰值必须足够高
        if activity_type == 'run':
            peak_height_threshold = signal_mean + signal_std * self.PEAK_HEIGHT_RATIO * 0.6
        else:
            peak_height_threshold = signal_mean + signal_std * self.PEAK_HEIGHT_RATIO

        valid_peaks = []
        last_valid_valley = 0

        for p_idx in p_loc:
            if signal[p_idx] < peak_height_threshold:
                continue

            # 寻找该峰值左右最近的波谷
            left_valleys = v_loc[v_loc < p_idx]
            right_valleys = v_loc[v_loc > p_idx]

            if len(left_valleys) == 0 or len(right_valleys) == 0:
                continue

            left_v = left_valleys[-1]
            right_v = right_valleys[0]

            # 波峰-波谷高度差检查
            left_height = signal[p_idx] - signal[left_v]
            right_height = signal[p_idx] - signal[right_v]

            if left_height < self.PEAK_VALLEY_MIN_DIFF or right_height < self.PEAK_VALLEY_MIN_DIFF:
                continue

            # 高度对称性检查
            if left_height > 0 and right_height > 0:
                ratio = max(left_height, right_height) / min(left_height, right_height)
                if ratio > 3.0:  # 太不对称，可能是噪声
                    continue

            # 时间间隔检查
            left_time = p_idx - left_v
            right_time = right_v - p_idx

            if activity_type == 'run':
                min_interval = self.MIN_STEP_INTERVAL_RUN
                max_interval = self.MAX_STEP_INTERVAL_RUN
            else:
                min_interval = self.MIN_STEP_INTERVAL_WALK
                max_interval = self.MAX_STEP_INTERVAL_WALK

            if left_time < min_interval or left_time > max_interval:
                continue
            if right_time < min_interval or right_time > max_interval:
                continue

            # 检查与前一个有效峰值的时间间隔
            if len(valid_peaks) > 0:
                time_since_last = p_idx - valid_peaks[-1]
                if time_since_last < min_interval // 2:
                    continue  # 太近，跳过

            valid_peaks.append(p_idx)
            last_valid_valley = right_v

        return np.array(valid_peaks, dtype=int)

    def validate_with_gyro(self, peak_indices, gyro_mag_window, acc_mag_window):
        """使用陀螺仪数据验证峰值是否对应真实的步行"""
        if len(peak_indices) == 0:
            return peak_indices

        gyro_energy = np.sum(gyro_mag_window**2) / len(gyro_mag_window)

        # 如果总体陀螺仪能量太低，可能是没有摆臂的假步行
        if gyro_energy < self.MIN_GYRO_ENERGY_THRESHOLD:
            # 但是跑步时加速度很大，即使陀螺仪能量低也可能是真跑步
            acc_std = np.std(acc_mag_window - np.mean(acc_mag_window))
            if acc_std < self.ACC_GRAVITY * 1.5:  # 不是跑步
                return np.array([], dtype=int)

        return peak_indices

    def count_steps_in_window(self, acc_mag_window, gyro_mag_window, activity_type):
        """在窗口内进行步数检测"""
        if activity_type in ['stationary', 'noise']:
            return 0, np.array([], dtype=int)

        # 去趋势
        acc_detrend = acc_mag_window - np.mean(acc_mag_window)

        # 寻找波峰波谷
        p_loc, v_loc = self.find_peaks_valleys(acc_detrend)

        if len(p_loc) == 0:
            return 0, np.array([], dtype=int)

        # 合并相邻波峰
        p_loc = self.merge_close_peaks(acc_detrend, p_loc, v_loc)

        # 去除伪峰
        valid_peaks = self.remove_false_peaks(acc_detrend, p_loc, v_loc, activity_type)

        # 陀螺仪验证
        valid_peaks = self.validate_with_gyro(valid_peaks, gyro_mag_window, acc_mag_window)

        return len(valid_peaks), valid_peaks

    def process_file(self, filepath, verbose=False):
        """处理单个数据文件，返回预测步数"""
        # 从文件名提取真实步数
        fname = os.path.basename(filepath)
        import re
        match = re.search(r'step(\d+)', fname)
        true_steps = int(match.group(1)) if match else 0

        # 加载数据
        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = self.load_imu_data(filepath)
        total_samples = len(acc_x)

        if total_samples < self.fs:
            return 0, true_steps

        # 预处理
        acc_x_filt, acc_y_filt, acc_z_filt, acc_mag = self.preprocess_acceleration(
            acc_x, acc_y, acc_z)
        gyro_mag = self.preprocess_gyroscope(gyro_x, gyro_y, gyro_z)

        # 滤波器延迟补偿
        delay_point = self.m1 // 2 + self.m2 // 2

        total_steps = 0

        # 滑动窗口处理
        acc_win = np.zeros(self.win_len)
        gyro_win = np.zeros(self.win_len)
        win_cnt = 0
        acc_buf = np.zeros(self.buf_len)
        gyro_buf = np.zeros(self.buf_len)
        buf_cnt = 0

        # 每1秒（50个样本）处理一次
        for i in range(self.fs + self.fs - delay_point,
                       len(acc_mag) - delay_point + 1, self.fs):
            # 获取当前秒的数据
            seg_acc = acc_mag[(i - self.fs):i]
            seg_gyro = gyro_mag[(i - self.fs):i]

            # 添加到窗口
            new_cnt = len(seg_acc)
            if win_cnt + new_cnt <= self.win_len:
                acc_win[win_cnt:win_cnt + new_cnt] = seg_acc
                gyro_win[win_cnt:win_cnt + new_cnt] = seg_gyro
                win_cnt += new_cnt
            else:
                acc_win = np.concatenate((acc_win[new_cnt:self.win_len], seg_acc))
                gyro_win = np.concatenate((gyro_win[new_cnt:self.win_len], seg_gyro))
                win_cnt = self.win_len

            if win_cnt >= self.win_len:
                # 合并缓冲区和窗口数据
                if buf_cnt > 0:
                    acc_full = np.concatenate((acc_buf[:buf_cnt], acc_win[:win_cnt]))
                    gyro_full = np.concatenate((gyro_buf[:buf_cnt], gyro_win[:win_cnt]))
                else:
                    acc_full = acc_win[:win_cnt].copy()
                    gyro_full = gyro_win[:win_cnt].copy()

                # 活动检测
                activity_type = self.detect_activity_type(acc_full, gyro_full)

                # 步数统计
                steps, _ = self.count_steps_in_window(acc_full, gyro_full, activity_type)

                if verbose:
                    print(f"  time={i//self.fs:3d}s  activity={activity_type:10s}  steps={steps}")

                total_steps += steps

                # 保存缓冲区数据用于下一个窗口
                buf_cnt = min(self.buf_len, len(acc_full) - self.win_len + self.LEFT_DATA_NUM)
                if buf_cnt > 0:
                    acc_buf[:buf_cnt] = acc_full[-buf_cnt:]
                    gyro_buf[:buf_cnt] = gyro_full[-buf_cnt:]

                win_cnt = 0
                acc_win[:] = 0
                gyro_win[:] = 0

        if verbose:
            print(f"  文件: {fname}")
            print(f"  预测步数: {total_steps}, 真实步数: {true_steps}")

        return total_steps, true_steps

    def evaluate_all(self, data_dir, verbose=False):
        """评估所有数据文件"""
        results = {'walk': [], 'run': [], 'others': []}

        for category in ['walk', 'run', 'others']:
            cat_dir = os.path.join(data_dir, category)
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

                if not verbose and len(results[category]) % 10 == 0:
                    print(f"  已处理 {category}/{len(results[category])}/{len(files)} 个文件")

        return results

    def compute_metrics(self, results):
        """计算MAE和MAPE指标"""
        print("\n" + "=" * 70)
        print("评估结果汇总")
        print("=" * 70)

        # Walk + Run 作为步行类（作业要求）
        walk_run = results['walk'] + results['run']

        all_data = {'步行(walk+run)': walk_run, '走路(walk)': results['walk'],
                    '跑步(run)': results['run'], '噪声(others)': results['others']}

        for name, data in all_data.items():
            true_steps = [d['true'] for d in data]
            pred_steps = [d['predicted'] for d in data]
            abs_errors = [d['abs_error'] for d in data]
            errors = [d['error'] for d in data]

            if len(true_steps) == 0:
                continue

            mae = np.mean(abs_errors)
            # MAPE: 对真实步数>0的计算百分比误差
            mape_data = [(abs(e) / t * 100) for e, t in zip(errors, true_steps) if t > 0]
            mape = np.mean(mape_data) if mape_data else float('nan')

            # 对others类特殊处理：期望输出0步
            if 'others' in name:
                false_positive_rate = sum(1 for p in pred_steps if p > 0) / len(pred_steps) * 100
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE (步数误差): {mae:.2f}")
                print(f"  误检率 (预测>0步的比例): {false_positive_rate:.1f}%")
                print(f"  预测步数范围: [{min(pred_steps)}, {max(pred_steps)}]")
            else:
                # 准确率（1 - MAPE/100）
                accuracy = max(0, 100 - mape)
                print(f"\n{name}:")
                print(f"  文件数: {len(data)}")
                print(f"  MAE (平均绝对误差): {mae:.2f} 步")
                print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
                print(f"  准确率: {accuracy:.2f}%")

                # 详细列表
                print(f"\n  详细结果:")
                for d in data:
                    print(f"    {d['file']}: 预测={d['predicted']:4d}, 真实={d['true']:4d}, "
                          f"误差={d['error']:+4d} ({abs(d['error'])/max(1,d['true'])*100:.1f}%)")

        return mae, mape


def main():
    data_dir = 'VeriHealthi_Algorithm_Homework_Code_Data/AccData'

    print("改进的计步算法 - 6轴IMU步数检测")
    print("=" * 70)

    counter = ImprovedStepCounter()

    # 单个文件演示
    demo_file = os.path.join(data_dir, 'walk',
                             'IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt')
    print(f"\n演示文件处理: {os.path.basename(demo_file)}")
    print("-" * 40)
    predicted, true = counter.process_file(demo_file, verbose=True)
    print(f"\n  结果: 预测={predicted}, 真实={true}, 误差={predicted-true:+d}")

    # 批量评估
    print("\n\n批量评估所有数据...")
    results = counter.evaluate_all(data_dir, verbose=False)
    mae, mape = counter.compute_metrics(results)

    return results, mae, mape


if __name__ == '__main__':
    main()
