# main.py
# Sensor Data Drift Analyzer (ทำเองตั้งแต่ศูนย์)
# ใช้: python main.py --help  เพื่อดูพารามิเตอร์ทั้งหมด

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from pathlib import Path

# ---------- Utils ----------
def moving_average_1d(x: np.ndarray, window: int = 21) -> np.ndarray:
    """Moving average 1D (ค่าเดียวกันกับ 'same' length)"""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    y = fftconvolve(x, kernel, mode="same")
    return y

def fft_lowpass_1d(x: np.ndarray, keep_ratio: float = 0.08) -> np.ndarray:
    """Low-pass ด้วย FFT: เก็บสัดส่วนความถี่ต่ำ (0..1)"""
    n = len(x)
    X = np.fft.rfft(x)
    K = max(1, int(np.ceil(keep_ratio * len(X))))
    X_lp = np.zeros_like(X)
    X_lp[:K] = X[:K]
    y = np.fft.irfft(X_lp, n=n)
    return y

# ---------- Simulation ----------
def simulate_data(N=30, T=300, seed=42,
                  A=1.2, sigma=0.6,
                  linear_drift_rate=0.003,
                  slow_period=180,
                  sensor_bias_sigma=0.02,
                  noise_sigma=0.05):
    """
    สร้างข้อมูลเซนเซอร์ 2D (NxN) ต่อเนื่องเวลา T พร้อม drift + noise
    base: โค้ง Gaussian เหมือนผิวเลนส์
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    base = A * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    t = np.arange(T)
    slow_omega = 2 * np.pi / slow_period
    random_walk = np.cumsum(rng.normal(0, 0.0005, size=T))
    global_drift = (linear_drift_rate * t) + 0.05 * np.sin(slow_omega * t) + random_walk

    sensor_bias = rng.normal(0, sensor_bias_sigma, size=(N, N))
    noise = rng.normal(0, noise_sigma, size=(T, N, N))

    data = np.empty((T, N, N), dtype=np.float64)
    for i in range(T):
        data[i] = base + global_drift[i] + sensor_bias + noise[i]
    return data, base, global_drift

# ---------- Analysis + Plots ----------
def analyze_and_plot(data: np.ndarray, outdir: Path, ma_window=21, keep_ratio=0.08):
    T, N, _ = data.shape
    t = np.arange(T)

    means = data.reshape(T, -1).mean(axis=1)
    stds  = data.reshape(T, -1).std(axis=1)

    means_ma  = moving_average_1d(means, window=ma_window)
    means_fft = fft_lowpass_1d(means, keep_ratio=keep_ratio)

    # สรุปเวลาเป็นตาราง
    df = pd.DataFrame({
        "t": t,
        "mean_raw": means,
        "std_raw": stds,
        f"mean_ma_w{ma_window}": means_ma,
        "mean_fft_lowpass": means_fft
    })
    csv_path = outdir / "sensor_drift_summary.csv"
    df.to_csv(csv_path, index=False)

    # Heatmaps (ต้น กลาง ท้าย)
    for ts, name in [(0, "t000"), (T//2, f"t{T//2:03d}"), (T-1, f"t{T-1:03d}")]:
        plt.figure(figsize=(6,5))
        plt.title(f"Heatmap: Deformation at t={ts}")
        im = plt.imshow(data[ts], origin="lower", aspect="equal")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("X index"); plt.ylabel("Y index")
        plt.tight_layout()
        plt.savefig(outdir / f"heatmap_{name}.png", dpi=160)
        plt.close()

    # Time series: raw vs MA vs FFT
    plt.figure(figsize=(9,4))
    plt.plot(t, means, label="mean (raw)")
    plt.plot(t, means_ma, label=f"mean (MA, w={ma_window})")
    plt.plot(t, means_fft, label=f"mean (FFT lowpass, {keep_ratio:.0%})")
    plt.title("Global Drift / Noise — Denoising Comparison")
    plt.xlabel("Time step"); plt.ylabel("Mean deflection")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "denoise_comparison.png", dpi=160)
    plt.close()

    # Std over time
    plt.figure(figsize=(9,4))
    plt.plot(t, stds, label="std (raw)")
    plt.title("Spatial Standard Deviation Over Time")
    plt.xlabel("Time step"); plt.ylabel("Std of deflection")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "std_over_time.png", dpi=160)
    plt.close()

    return csv_path

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Sensor Data Drift Analyzer")
    p.add_argument("--N", type=int, default=30, help="ขนาดกริด N x N")
    p.add_argument("--T", type=int, default=300, help="จำนวน time steps")
    p.add_argument("--seed", type=int, default=42, help="seed สำหรับสุ่ม")
    p.add_argument("--noise", type=float, default=0.05, help="ค่า sigma ของ noise")
    p.add_argument("--drift", type=float, default=0.003, help="linear drift rate ต่อหนึ่ง step")
    p.add_argument("--slow_period", type=int, default=180, help="คาบของ drift แบบ sine (จำนวน step)")
    p.add_argument("--ma_window", type=int, default=21, help="ขนาดหน้าต่าง Moving Average")
    p.add_argument("--keep_ratio", type=float, default=0.08, help="สัดส่วนความถี่ต่ำที่เก็บไว้ใน FFT (0..1)")
    p.add_argument("--outdir", type=str, default="outputs", help="โฟลเดอร์บันทึกผลลัพธ์")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    data, base, global_drift = simulate_data(
        N=args.N, T=args.T, seed=args.seed,
        linear_drift_rate=args.drift,
        slow_period=args.slow_period,
        noise_sigma=args.noise
    )

    csv_path = analyze_and_plot(
        data=data, outdir=outdir,
        ma_window=args.ma_window, keep_ratio=args.keep_ratio
    )

    # บันทึกตัวอย่างเฟรม
    np.save(outdir / "frame_t000.npy", data[0])
    np.save(outdir / f"frame_t{args.T//2:03d}.npy", data[args.T//2])
    np.save(outdir / f"frame_t{args.T-1:03d}.npy", data[-1])

    print("✓ Done!")
    print(f"- Summary CSV: {csv_path}")
    print(f"- Images: {outdir / 'heatmap_t000.png'}, {outdir / f'heatmap_t{args.T//2:03d}.png'}, {outdir / f'heatmap_t{args.T-1:03d}.png'}")
    print(f"- Plots: {outdir / 'denoise_comparison.png'}, {outdir / 'std_over_time.png'}")

if __name__ == "__main__":
    main()
