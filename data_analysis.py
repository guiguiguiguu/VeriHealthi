#!/usr/bin/python
"""分析 walk/run/others 三类数据的特征差异"""
import numpy as np
import os
import glob

def load_imu_data(filepath, max_samples=5000):
    """加载6轴IMU数据，返回 (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z)"""
    with open(filepath, 'r') as f:
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

def compute_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """计算信号特征"""
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # 去除重力分量 (约4096)
    acc_mag_nog = acc_mag - np.mean(acc_mag)

    features = {
        'acc_mag_mean': np.mean(acc_mag),
        'acc_mag_std': np.std(acc_mag),
        'acc_mag_var': np.var(acc_mag_nog),
        'acc_mag_range': np.max(acc_mag) - np.min(acc_mag),
        'gyro_mag_mean': np.mean(gyro_mag),
        'gyro_mag_std': np.std(gyro_mag),
        'gyro_mag_energy': np.sum(gyro_mag**2) / len(gyro_mag),
        'gyro_energy_ratio': np.sum(gyro_mag**2) / np.sum(acc_mag_nog**2) if np.sum(acc_mag_nog**2) > 0 else 0,
    }

    # 频域特征 - 简化FFT
    if len(acc_mag_nog) >= 256:
        fft = np.abs(np.fft.rfft(acc_mag_nog[:256]))
        freqs = np.fft.rfftfreq(256, 1/50)
        # 主要频率 (1-4 Hz 对应步行/跑步)
        band = (freqs >= 1) & (freqs <= 4)
        if np.any(band):
            features['dominant_freq'] = freqs[band][np.argmax(fft[band])]
            features['freq_energy_1_4hz'] = np.sum(fft[band])
        else:
            features['dominant_freq'] = 0
            features['freq_energy_1_4hz'] = 0
        features['total_spectral_energy'] = np.sum(fft)

    return features

def main():
    base = 'VeriHealthi_Algorithm_Homework_Code_Data/AccData'

    for category in ['walk', 'run', 'others']:
        files = glob.glob(os.path.join(base, category, '*.txt'))
        print(f"\n{'='*60}")
        print(f"类别: {category} ({len(files)} 个文件)")
        print(f"{'='*60}")

        all_features = []
        for f in sorted(files):
            try:
                gx, gy, gz, ax, ay, az = load_imu_data(f)
                feats = compute_features(ax, ay, az, gx, gy, gz)
                all_features.append(feats)
                fname = os.path.basename(f)
                print(f"\n  文件: {fname}")
                print(f"  加速度幅度: mean={feats['acc_mag_mean']:.0f}, std={feats['acc_mag_std']:.0f}, range={feats['acc_mag_range']:.0f}")
                print(f"  陀螺仪幅度: mean={feats['gyro_mag_mean']:.1f}, std={feats['gyro_mag_std']:.1f}")
                print(f"  陀螺仪能量: {feats['gyro_mag_energy']:.1f}")
                if 'dominant_freq' in feats:
                    print(f"  主频: {feats['dominant_freq']:.2f} Hz, 1-4Hz能量占比: {feats['freq_energy_1_4hz']/feats['total_spectral_energy']*100:.1f}%")
            except Exception as e:
                print(f"  错误: {e}")

        # 汇总统计
        if all_features:
            print(f"\n  --- {category} 汇总 ---")
            for key in ['acc_mag_std', 'acc_mag_range', 'gyro_mag_std', 'gyro_mag_energy']:
                vals = [f[key] for f in all_features]
                print(f"  {key}: min={min(vals):.1f}, max={max(vals):.1f}, mean={np.mean(vals):.1f}")

if __name__ == '__main__':
    main()
