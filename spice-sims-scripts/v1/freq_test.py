import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#load csv file
data = pd.read_csv("multivibrator1.0.txt", delim_whitespace =True, comment = '#')

#extract time and voltage columns
time = data.iloc[:, 0].values
voltage = data.iloc[:, 1].values

#peak calculations
peaks, _ = find_peaks(voltage, height=1.95)  
peak_times = time[peaks] #array of times where we "peak"

# compute periods and frequency stats
periods = np.diff(peak_times)
jitter = np.std(periods)
frequencies = 1 / periods
avg_freq = np.mean(frequencies)
std_freq = np.std(frequencies)
freq_stability = std_freq / avg_freq  # coefficient of variation

# Print results
print(f"Average Frequency: {avg_freq:.3f} Hz")
print(f"Jitter (cycle-to-cycle std of period): {jitter:.6f} s")
print(f"Standard Deviation: {std_freq:.3f} Hz")
print(f"Frequency Stability (σ/μ): {freq_stability:.5f}")

plt.plot(time, voltage, label="Voltage")
plt.plot(peak_times, voltage[peaks], "rx", label="Peaks")
plt.title("Oscillator Output")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()
