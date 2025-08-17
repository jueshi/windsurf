import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend instead of Tkinter

import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
fs = 100  # 采样频率
t = np.linspace(0, 1, fs, endpoint=False)  # 时间轴 (1秒内采样)
f = 1  # 信号频率 (1 Hz)

# 原始信号
s = np.sin(2 * np.pi * f * t)

# 噪声 (均值为0，高斯白噪声)
noise = np.random.normal(0, 0.5, t.shape)

# 观测信号 = 信号 + 噪声
x = s + noise

# 绘制原始信号和观测信号
plt.figure(figsize=(10, 4))
plt.plot(t, s, label="原始信号 s(t)")
plt.plot(t, x, label="观测信号 x(t)")
plt.legend()
plt.title("原始信号与观测信号")
plt.xlabel("时间 (s)")
plt.ylabel("幅值")
plt.show(block=True)

# 计算信号的功率谱密度 (简单假设)
signal_power = np.abs(np.fft.fft(s))**2
noise_power = np.abs(np.fft.fft(noise))**2

# 维纳滤波器的频率响应
H = signal_power / (signal_power + noise_power)

# 对观测信号进行傅里叶变换
X = np.fft.fft(x)

# 应用维纳滤波器
S_hat = H * X

# 反傅里叶变换得到滤波后的信号
s_hat = np.fft.ifft(S_hat).real

# 绘制滤波后的信号
plt.figure(figsize=(10, 4))
plt.plot(t, s, label="原始信号 s(t)")
plt.plot(t, x, label="观测信号 x(t)")
plt.plot(t, s_hat, label="滤波后信号 s_hat(t)", linestyle='dashed')
plt.legend()
plt.title("维纳滤波效果")
plt.xlabel("时间 (s)")
plt.ylabel("幅值")
plt.show(block=True)
