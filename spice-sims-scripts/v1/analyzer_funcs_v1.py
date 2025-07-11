from matplotlib.ticker import ScalarFormatter
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks

def plot_waveforms_from_step_file(filename, voltage_threshold=None):
    with open(filename, 'r', encoding='latin1') as file:
        lines = file.readlines()

    all_time = []
    all_voltage = []
    all_labels = []

    time_vals = []
    volt_vals = []
    current_cap = None

    def store_block():
        if time_vals and volt_vals:
            all_time.append(np.array(time_vals))
            all_voltage.append(np.array(volt_vals))
            all_labels.append(current_cap)

    for line in lines:
        line = line.strip()
        if line.startswith("Step Information:"):
            store_block()
            time_vals = []
            volt_vals = []
            match = re.search(r'C1=([\deE\+\-\.a-zA-Zµ]+)', line)
            current_cap = match.group(1) if match else "Unknown"
        elif line and not line.startswith("time") and '\t' in line:
            try:
                t, v = map(float, line.split('\t'))
                time_vals.append(t)
                volt_vals.append(v)
            except ValueError:
                continue

    # Don't forget the last block
    store_block()

    # === Create vertically stacked plots ===
    num_plots = len(all_time)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Make iterable if only one

    for i in range(num_plots):
        ax = axes[i]
        ax.plot(all_time[i], all_voltage[i])
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Cap = {all_labels[i]}")
        ax.grid(True)
        if voltage_threshold is not None:
            ax.axhline(voltage_threshold, color='red', linestyle='--')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


"""
Given a step file sweeping a certain value, produces the following:
- Pulse Width vs componenet value Graph
- Frequency vs component value graph

- Table of jitter and stability
- Calculation at which point the oscillator stops oscillating
- Table of for each resistor cap combination
    - Average Frequency
    - Pulse Width
    - Jitter and stability
    - If it stops oscillating, at what point

"""

# ── helper: unit-string → float ────────────────────────────────────
def parse_comp_value(s):
    if not s or not isinstance(s, str):
        return np.nan
    s = s.replace('µ', 'u').lower()
    mul = {'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'g':1e9}
    return float(s[:-1]) * mul.get(s[-1], 1) if s[-1].isalpha() else float(s)

# ── one-peak-per-cycle detector with dip filter ────────────────────
def filter_peaks(v, cand_peaks, reset_thr=0.2):
    keep, last = [], 0
    for p in cand_peaks:
        if np.min(v[last:p]) < reset_thr:
            keep.append(p); last = p
    return np.asarray(keep, dtype=int)         # ← ensure integer indices!

# ── main function ─────────────────────────────────────────────────
def analyze_sweep_file(filename,
                       param_name='C1',
                       PEAK_HEIGHT=1.8,
                       RESET_THRESHOLD=0.2,
                       V_ON_THRESHOLD=1.8):

    results = []
    wave_time, wave_volt, wave_label, wave_peaks = [], [], [], []

    with open(filename, 'r', encoding='latin1') as f:
        lines = f.readlines()

    cur_param, tbuf, vbuf = None, [], []

    def finish_block():
        if not tbuf: return
        t, v = np.asarray(tbuf), np.asarray(vbuf)

        cand, _ = find_peaks(v, height=PEAK_HEIGHT)
        peaks   = filter_peaks(v, cand, RESET_THRESHOLD)

        wave_time.append(t); wave_volt.append(v)
        wave_label.append(f"{param_name}={cur_param}")
        wave_peaks.append(peaks)

        d = {param_name: cur_param}
        if peaks.size >= 2:
            pt, per = t[peaks], np.diff(t[peaks])
            freq = 1/per
            d.update(AvgFreq_Hz=freq.mean(),
                     Jitter_s=per.std(),
                     Stability=freq.std()/freq.mean())
        else:
            d.update(AvgFreq_Hz=np.nan, Jitter_s=np.nan, Stability=np.nan)

        on = v > V_ON_THRESHOLD
        dt = np.diff(t, prepend=t[0])
        t_on, t_off = dt[on].sum(), dt[~on].sum()
        d['PulseWidthRatio'] = t_on/t_off if t_off else np.nan

        results.append(d)

    # ── parse sweep file ──────────────────────────────────────────
    for raw in lines:
        line = raw.strip()
        if line.startswith("Step Information:"):
            finish_block(); tbuf, vbuf = [], []
            m = re.search(rf'{param_name}=([^\s\]]+)', line)
            cur_param = m.group(1) if m else None
        elif '\t' in line and not line.startswith("time"):
            try:
                t, v = map(float, line.split('\t')[:2])
                tbuf.append(t); vbuf.append(v)
            except ValueError:
                pass
    finish_block()

    # ── DataFrame & summary plots ─────────────────────────────────
    df = pd.DataFrame(results)
    df[f'{param_name}_F'] = df[param_name].apply(parse_comp_value)
    df = df.sort_values(f'{param_name}_F')

    print(df[[param_name,'AvgFreq_Hz','Jitter_s','Stability','PulseWidthRatio']])

    for ycol, title, ylab in [
        ('AvgFreq_Hz', f'Frequency vs {param_name}', 'Average Frequency (Hz)'),
        ('PulseWidthRatio', f'Pulse-Width Ratio vs {param_name}', 'T_on / T_off')]:
        plt.figure(); plt.plot(df[f'{param_name}_F'], df[ycol], 'o-')
        plt.title(title); plt.xlabel(param_name); plt.ylabel(ylab)
        plt.xscale('log'); plt.grid(True)
        fmt = ScalarFormatter(useMathText=True); fmt.set_scientific(True)
        fmt.set_powerlimits((-3,3)); plt.gca().xaxis.set_major_formatter(fmt)
        plt.tight_layout()

    # ── stacked waveforms with single-peak markers ───────────────
    n = len(wave_time)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
    axs = [axs] if n == 1 else axs
    for ax, t, v, lab, pk in zip(axs, wave_time, wave_volt, wave_label, wave_peaks):
        ax.plot(t, v, label=lab)
        if pk.size:                                  # ← skip if no peaks
            ax.plot(t[pk], v[pk], 'rx', ms=6)
        ax.set_ylabel('V'); ax.set_title(lab); ax.grid(True); ax.legend()
    axs[-1].set_xlabel('Time (s)')
    fig.tight_layout(); plt.show()

    return df

