import re

# 读取文件内容
file_path = "./grpo_output.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式匹配 loss 和 KL
loss_pattern = r"['\"]loss['\"]:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
kl_pattern = r"['\"]kl['\"]:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"

loss_values = list(map(float, re.findall(loss_pattern, content)))
kl_values = list(map(float, re.findall(kl_pattern, content)))

print("Loss values:", loss_values)
print("KL values:", kl_values)

import matplotlib.pyplot as plt

# 绘图设置
plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(loss_values, label="Loss", color="blue")
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# KL 散度曲线
plt.subplot(1, 2, 2)
plt.plot(kl_values, label="KL Divergence", color="green")
plt.title("KL Divergence")
plt.xlabel("Step")
plt.ylabel("KL")
plt.grid(True)
plt.legend()

plt.tight_layout()
>>>>>>> c7de23d (Initial commit)
plt.show()