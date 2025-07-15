import re, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator
from scipy.signal import find_peaks

# ── helpers ────────────────────────────────────────────────────────
def parse_val(txt: str) -> float:
    if not isinstance(txt, str):
        return np.nan
    txt = txt.replace('µ', 'u').lower()
    table = {'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,
             'm':1e-3,'k':1e3,'meg':1e6,'g':1e9}
    for suf, fac in table.items():
        if txt.endswith(suf):
            return float(txt[:-len(suf)]) * fac
    return float(txt)

def one_peak(sig, hi, lo):
    cand, _ = find_peaks(sig, height=hi)
    keep, last = [], 0
    for p in cand:
        if sig[last:p].min() < lo:
            keep.append(p); last = p
    return np.asarray(keep, dtype=int)

# ── main function ─────────────────────────────────────────────────
def analyze_step_current(filename,
                         param_name='R2',
                         param_unit='Ω',
                         hi_frac=0.6,
                         lo_frac=0.2,
                         show_table=True):

    rows, w_t, w_i, w_pk, labels = [], [], [], [], []
    tbuf, ibuf, cur_val = [], [], None

    with open(filename, 'r', encoding='latin1') as f:
        lines = f.readlines()

    def finish():
        nonlocal tbuf, ibuf
        if not tbuf: return
        t = np.asarray(tbuf); i = np.asarray(ibuf)

        hi = i.min() + hi_frac * (i.max() - i.min())
        lo = i.min() + lo_frac * (i.max() - i.min())
        pk = one_peak(i, hi, lo)

        w_t.append(t); w_i.append(i); w_pk.append(pk)
        labels.append(f"{param_name}={cur_val}")

        d = {param_name: cur_val}
        if pk.size >= 2:
            per = np.diff(t[pk]); freq = 1 / per
            d['AvgFreq_Hz'] = freq.mean()
            d['Jitter_s']   = per.std()
            d['Stability']  = freq.std() / freq.mean()
        else:
            d.update(AvgFreq_Hz=np.nan, Jitter_s=np.nan, Stability=np.nan)

        dt = np.diff(t, prepend=t[0])
        on_mask = i >= hi
        t_on, t_off = dt[on_mask].sum(), dt[~on_mask].sum()
        d['PulseWidthRatio'] = t_on / t_off if t_off else np.nan

        I_high = i[on_mask].mean() if np.any(on_mask) else np.nan
        I_low  = np.abs(i[i <= lo]).mean() if np.any(i <= lo) else np.nan
        d['OnOffRatio'] = I_high / I_low if I_low else np.nan
        rows.append(d)
        tbuf, ibuf = [], []

    # ---- parse file ------------------------------------------------
    for raw in lines:
        line = raw.strip()
        if line.startswith("Step Information:"):
            finish()
            m = re.search(rf'{param_name}=([^\s\]]+)', line)
            cur_val = m.group(1) if m else None
        elif '\t' in line and not line.startswith('time'):
            try:
                t, amp = map(float, line.split('\t')[:2])
                tbuf.append(t); ibuf.append(amp)
            except ValueError:
                pass
    finish()

    # ---- DataFrame -------------------------------------------------
    df = pd.DataFrame(rows)
    df[f'{param_name}_num'] = df[param_name].apply(parse_val)
    df.sort_values(f'{param_name}_num', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if show_table:
        print(df[[param_name, 'AvgFreq_Hz', 'Jitter_s',
                  'PulseWidthRatio', 'OnOffRatio']])

    # ---- metric plots ---------------------------------------------
    eng = EngFormatter(unit=param_unit, places=0)

    def add_line(ax, ycol, title, ylab):
        ax.plot(df.index, df[ycol], 'o-')
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(True)
        ax.set_xticks(df.index)
        ax.set_xticklabels([eng(v) for v in df[f'{param_name}_num']],
                           rotation=30, ha='right', fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    fig, (axF, axPW, axIO) = plt.subplots(3, 1, figsize=(8, 8),
                                          sharex=True, constrained_layout=True)
    add_line(axF, 'AvgFreq_Hz',      f'Frequency vs {param_name}',        'Hz')
    add_line(axPW,'PulseWidthRatio', f'Pulse-Width Ratio vs {param_name}', 'T_on/T_off')
    add_line(axIO,'OnOffRatio',      f'On-Off Current Ratio vs {param_name}','I_high / I_low')
    axIO.set_xlabel(f'{param_name} [{param_unit}]')

    # ---- waveform panel (grid layout) -----------------------------
    n = len(w_t)
    # choose columns: 1, 2, or 3 for readability
    if   n <= 4:  cols = 1
    elif n <= 12: cols = 2
    else:         cols = 3
    rows = math.ceil(n / cols)

    fig_w, axs = plt.subplots(rows, cols, figsize=(10, 2.2*rows),
                              sharex=True)
    axs = np.atleast_1d(axs).ravel()  # flatten even if rows*cols==1

    for ax, t, i, pk, lab in zip(axs, w_t, w_i, w_pk, labels):
        ax.plot(t, i, label=lab, lw=1)
        if pk.size: ax.plot(t[pk], i[pk], 'rx', ms=6)
        ax.set_ylabel('A'); ax.set_title(lab, fontsize=9)
        ax.grid(True); ax.legend(fontsize=8)

    # turn off unused axes
    for k in range(len(w_t), len(axs)):
        axs[k].set_visible(False)

    axs[-1 if len(w_t) > 0 else 0].set_xlabel('Time (s)')
    fig_w.tight_layout()
    plt.show()

    return df


# df_cur = analyze_step_current(
#             "multivibrator1.7.txt",
#             param_name='Rcl1',  
#             hi_frac=0.6,         
#             lo_frac=0.2)        

df_cur = analyze_step_current(
            "multivibrator1.7.2.txt",
            param_name='Rcl1',  
            hi_frac=0.6,         
            lo_frac=0.2)  

# df_cur = analyze_step_current(
#             "multivibrator1.7.3.txt",
#             param_name='Rcl1',  
#             hi_frac=0.6,        
#             lo_frac=0.2)         