import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Given .txt file from SPICE with a single transient analysis (not sweep or step),
# calculates frequency and stabilty

# === Load CSV file ===
data = pd.read_csv("multivibrator1.4.txt", delim_whitespace=True, comment='#')

# === Extract time and voltage ===
time = data.iloc[:, 0].values
voltage = data.iloc[:, 1].values

# === use find_peaks to find all candidate peaks ===
PEAK_THRESHOLD = 1.95
candidate_peaks, _ = find_peaks(voltage, height=PEAK_THRESHOLD)

# === filter peaks to keep only those preceded by dip below 0.2V ===
RESET_THRESHOLD = 0.2
filtered_peaks = []

last_accepted = 0  # index into voltage array
for peak in candidate_peaks:
    # Look between the last accepted peak and the current one
    if np.min(voltage[last_accepted:peak]) < RESET_THRESHOLD:
        filtered_peaks.append(peak)
        last_accepted = peak  # update for next iteration

peak_times = time[filtered_peaks]

# === Compute frequency stats ===
periods = np.diff(peak_times)
frequencies = 1 / periods
avg_freq = np.mean(frequencies)
std_freq = np.std(frequencies)
jitter = np.std(periods)
freq_stability = std_freq / avg_freq

# === Print stats ===
print(f"Average Frequency: {avg_freq:.3f} Hz")
print(f"Jitter (cycle-to-cycle std of period): {jitter:.6f} s")
print(f"Standard Deviation: {std_freq:.3f} Hz")
print(f"Frequency Stability (σ/μ): {freq_stability:.5f}")

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(time, voltage, label="Voltage")
plt.plot(peak_times, voltage[filtered_peaks], "rx", label="Filtered Peaks")
plt.title("Oscillator Output")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
