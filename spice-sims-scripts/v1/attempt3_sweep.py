import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from matplotlib.ticker import FuncFormatter

# === Parameters ===
FILENAME = "multivibrator1.3.txt"
PEAK_HEIGHT = 1.8  # threshold for rising edge
V_ON_THRESHOLD = 1.8  # threshold for pulse width

# === Tick formatter for x-axis (e.g. show "100n", "1µ") ===
def format_tick(x, pos):
    if x >= 1e-6:
        return f"{x*1e6:.0f}µ"
    else:
        return f"{x*1e9:.0f}n"

# === Convert capacitance strings like '100n' or '1µ' to float ===
def parse_cap_value(cap_str):
    cap_str = cap_str.replace(' ', '').lower().replace('µ', 'u')
    if cap_str.endswith('p'):
        return float(cap_str[:-1]) * 1e-12
    elif cap_str.endswith('n'):
        return float(cap_str[:-1]) * 1e-9
    elif cap_str.endswith('u'):
        return float(cap_str[:-1]) * 1e-6
    elif cap_str.endswith('m'):
        return float(cap_str[:-1]) * 1e-3
    elif cap_str.endswith('f'):
        return float(cap_str[:-1])
    else:
        return float(cap_str)

# === Analysis Function ===
def analyze_sweep_file(filename):
    results = []
    with open(filename, 'r', encoding='latin1') as file:
        lines = file.readlines()

    current_cap = None
    time_vals = []
    volt_vals = []

    def process_block(time_vals, volt_vals, current_cap):
        result = {}
        time = np.array(time_vals)
        voltage = np.array(volt_vals)
        result['Capacitance'] = current_cap

        # Frequency-related metrics
        peaks, _ = find_peaks(voltage, height=PEAK_HEIGHT)
        if len(peaks) >= 2:
            peak_times = time[peaks]
            periods = np.diff(peak_times)
            frequencies = 1 / periods
            result['AvgFreq_Hz'] = np.mean(frequencies)
            result['Jitter_s'] = np.std(periods)
            result['Stability'] = np.std(frequencies) / np.mean(frequencies)
        else:
            result['AvgFreq_Hz'] = np.nan
            result['Jitter_s'] = np.nan
            result['Stability'] = np.nan

        # Pulse width ratio
        above = voltage > V_ON_THRESHOLD
        time_diff = np.diff(time, prepend=time[0])
        t_on = np.sum(time_diff[above])
        t_off = np.sum(time_diff[~above])
        result['PulseWidthRatio'] = t_on / t_off if t_off > 0 else np.nan

        return result

    for line in lines:
        line = line.strip()
        if line.startswith("Step Information:"):
            if time_vals and volt_vals:
                results.append(process_block(time_vals, volt_vals, current_cap))
                time_vals = []
                volt_vals = []

            match = re.search(r'C1=([\deE\+\-\.a-zA-Zµ]+)', line)
            if match:
                current_cap = match.group(1)

        elif line and not line.startswith("time") and '\t' in line:
            try:
                t, v = map(float, line.split('\t'))
                time_vals.append(t)
                volt_vals.append(v)
            except ValueError:
                continue

    # Final block
    if time_vals and volt_vals:
        results.append(process_block(time_vals, volt_vals, current_cap))

    # Create DataFrame
    df = pd.DataFrame(results)
    df['Capacitance_F'] = df['Capacitance'].apply(parse_cap_value)
    df = df.sort_values(by='Capacitance_F')

    return df

# === Run Analysis ===
df = analyze_sweep_file(FILENAME)

# === Print Table ===
print(df[['Capacitance', 'AvgFreq_Hz', 'Jitter_s', 'Stability', 'PulseWidthRatio']])

# === Plot Frequency vs Capacitance ===
plt.figure()
plt.plot(df['Capacitance_F'], df['AvgFreq_Hz'], 'o-')
plt.title("Frequency vs Capacitance")
plt.xlabel("Capacitance")
plt.ylabel("Average Frequency (Hz)")
plt.xscale("linear")
plt.xlim(1e-7, 1.2e-6)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_tick))
plt.tight_layout()

# === Plot Pulse Width Ratio vs Capacitance ===
plt.figure()
plt.plot(df['Capacitance_F'], df['PulseWidthRatio'], 'o-')
plt.title("Pulse Width Ratio vs Capacitance")
plt.xlabel("Capacitance")
plt.ylabel("T_on / T_off")
plt.xscale("linear")
plt.xlim(1e-7, 1.2e-6)
plt.grid(True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_tick))
plt.tight_layout()
plt.show()
