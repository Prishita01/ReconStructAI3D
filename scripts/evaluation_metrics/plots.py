import pandas as pd
import matplotlib.pyplot as plt

# 1) Load your results
df = pd.read_csv("depth_metrics.csv")

# 2) Compute means
mean_psnr = df["PSNR"].mean()
mean_ssim = df["SSIM"].mean()

# 3) Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12,4))

# PSNR histogram
axes[0].hist(df["PSNR"], bins=10, edgecolor='black')
axes[0].axvline(mean_psnr, linestyle='--', linewidth=2,
                label=f"Mean = {mean_psnr:.2f} dB")
axes[0].set_title("PSNR Distribution")
axes[0].set_xlabel("PSNR (dB)")
axes[0].set_ylabel("Count")
axes[0].legend()

# SSIM histogram
axes[1].hist(df["SSIM"], bins=10, edgecolor='black')
axes[1].axvline(mean_ssim, linestyle='--', linewidth=2,
                label=f"Mean = {mean_ssim:.3f}")
axes[1].set_title("SSIM Distribution")
axes[1].set_xlabel("SSIM")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.show()