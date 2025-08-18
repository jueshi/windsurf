from pybert.pybert import PyBERT

# 初始化 PyBERT 对象
pybert = PyBERT()

# 设置参数（可以根据需求修改）
pybert.bit_rate = 10.0e9  # 比特率 (10 Gbps)
pybert.n_bits = 1000      # 仿真比特数
pybert.tx_tap_weights = [0.5, -0.25, 0.0]  # 发射端均衡器设置
pybert.rx_bw = 12.5e9     # 接收端带宽

# 运行仿真
pybert.run_simulation()

# 输出结果
print("眼图宽度:", pybert.eye_width)
print("眼图高度:", pybert.eye_height)
