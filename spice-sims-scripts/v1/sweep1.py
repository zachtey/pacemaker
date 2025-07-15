"""
sweep_onoff_vs_pw.py   –   robust sweep + “Rejected” flag

Adds a Rejected column (True / False).  
A run is rejected – and NOT plotted – if **any** of these exceed 0.01:

    • Jitter_s
    • Stability               (freq σ / μ)
    • PW_Stability            (pulse-width σ / μ)
    • OnOff_Stability         (on/off-ratio σ / μ)

Accepted runs are shown exactly as before:
    1–10 Hz → viridis gradient,  >10 Hz → red.
Hover a dot to read its frequency (mplcursors).
"""

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks
try:
    import mplcursors
except ModuleNotFoundError:
    mplcursors = None


def sweep_onoff_vs_pw(filename,
                      hi_fracs=(0.6, 0.5, 0.4, 0.3),
                      lo_frac=0.2,
                      verbose=True,
                      interactive=True,
                      thresh=0.01):
    # ----- helpers -------------------------------------------------
    def one_peak(sig, hi, lo):
        cand, _ = find_peaks(sig, height=hi)
        keep, last = [], 0
        for p in cand:
            if sig[last:p].min() < lo:
                keep.append(p); last = p
        return np.asarray(keep, int)

    def stab(arr): return np.std(arr)/np.mean(arr) if len(arr)>1 and np.mean(arr) else np.nan

    # ----- analyse one block --------------------------------------
    def analyse_block(t, i, p_dict):
        for hf in hi_fracs:
            hi = i.min() + hf*(i.max()-i.min())
            lo = i.min() + lo_frac*(i.max()-i.min())
            pk = one_peak(i, hi, lo)
            if pk.size >= 2:
                break
        else:
            return None                                    # no oscillation

        per, freq = np.diff(t[pk]), 1/np.diff(t[pk])
        pw_l, io_l = [], []
        for a,b in zip(pk[:-1], pk[1:]):
            seg_t, seg_i = t[a:b], i[a:b]
            dt = np.diff(seg_t, prepend=seg_t[0])
            on = seg_i >= hi
            pw_l.append(dt[on].sum()/dt[~on].sum())
            ih = seg_i[on].mean()
            il = abs(seg_i[seg_i<=lo]).mean()
            io_l.append(ih/il)

        row = dict(**p_dict,
                   AvgFreq_Hz=freq.mean(),
                   Jitter_s=per.std(),
                   Stability=freq.std()/freq.mean(),
                   PulseWidthRatio=np.mean(pw_l),
                   PW_Stability=stab(pw_l),
                   OnOffRatio=np.mean(io_l),
                   OnOff_Stability=stab(io_l))
        # rejection check
        reject = any((row[k] > thresh) for k in
                     ('Jitter_s','Stability','PW_Stability','OnOff_Stability')
                     if not np.isnan(row[k]))
        row['Rejected'] = reject
        return row

    # ----- parse file ---------------------------------------------
    rows, tbuf, ibuf, cur = [], [], [], {}
    with open(filename,'r',encoding='latin1') as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Step Information:"):
                if tbuf:
                    r = analyse_block(np.asarray(tbuf), np.asarray(ibuf), cur)
                    if r: rows.append(r)
                tbuf, ibuf, cur = [], [], dict(re.findall(r'(\w+)=([\w\.]+)', line))
                continue
            if '\t' in line and not line.startswith('time'):
                try:
                    t_val, i_val = map(float, line.split('\t')[:2])
                    tbuf.append(t_val); ibuf.append(i_val)
                except ValueError:
                    pass
        if tbuf:
            r = analyse_block(np.asarray(tbuf), np.asarray(ibuf), cur)
            if r: rows.append(r)

    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False, float_format=lambda x:f"{x:.6g}"))

    # ------------- accepted points only ---------------------------
    acc = df[~df['Rejected']]
    freq_val = acc['AvgFreq_Hz'].to_numpy(float)

    # viridis 1–10 Hz
    colours = plt.cm.viridis((np.clip(freq_val,1,10)-1)/9)
    colours[freq_val > 10] = np.array([1,0,0,1])      # red

    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(acc['PulseWidthRatio'], acc['OnOffRatio'],
                    c=colours, s=45, alpha=.9)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis',
                                              norm=plt.Normalize(1,10)),
                        ax=ax, label='Avg Frequency (Hz, 1–10)')
    ax.set_xlabel('Pulse-Width Ratio  (T_on / T_off)')
    ax.set_ylabel('On-Off Current Ratio  (I_high / I_low)')
    ax.set_title('On-off vs Pulse-width  (red = >10 Hz, rejected omitted)')
    ax.grid(alpha=.3)

    if interactive and mplcursors:
        cursor = mplcursors.cursor(sc, hover=True)
        @cursor.connect("add")
        def _(sel):
            sel.annotation.set_text(f"{freq_val[sel.index]:.3f} Hz")

    plt.tight_layout(); plt.show()
    return df


# ---------------- run standalone ----------------------------------
if __name__ == "__main__":
    sweep_onoff_vs_pw("multivibrator1.8.1.txt",
                      hi_fracs=(0.6,0.5,0.4,0.3),
                      lo_frac=0.2,
                      verbose=True,
                      interactive=True,
                      thresh=0.05)