import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ── dip-rearm peak detector ───────────────────────────────────────
def _one_peak(sig, hi, lo):
    cand, _ = find_peaks(sig, height=hi)
    keep, last = [], 0
    for p in cand:
        if sig[last:p].min() < lo:
            keep.append(p); last = p
    return np.asarray(keep, dtype=int)


def analyze_single_waveform(
        filename: str,
        wave_col: int = 1,          # column index of voltage/current
        wave_name: str = "I",       # label for plot
        hi_frac: float = 0.6,       # 60 % of swing = high threshold
        lo_frac: float = 0.2,       # 20 % of swing = low threshold
        show_plot: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
    """
    Analyse ONE transient waveform (no .step, voltage or current).

    Returns one-row DataFrame with:
      AvgFreq_Hz · Jitter_s · Stability
      PulseWidthRatio · PW_Stability
      OnOffRatio     · OnOff_Stability
      PeakCurrent_A  · LowCurrent_A
    """

    # 1 ── load ----------------------------------------------------------------
    df   = pd.read_csv(filename, sep=r'\s+', comment='#', engine='python')
    time = df.iloc[:, 0].to_numpy()
    sig  = df.iloc[:, wave_col].to_numpy()

    # 2 ── adaptive thresholds -------------------------------------------------
    hi = sig.min() + hi_frac * (sig.max() - sig.min())
    lo = sig.min() + lo_frac * (sig.max() - sig.min())

    peaks = _one_peak(sig, hi, lo)
    if peaks.size < 2:
        raise RuntimeError("Fewer than two peaks detected – cannot compute metrics.")

    periods = np.diff(time[peaks])
    freq    = 1 / periods

    # 3 ── per-cycle PW ratio & on-off ratio -----------------------------------
    pw_list, io_list = [], []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        seg_t, seg_i = time[p0:p1], sig[p0:p1]
        dt = np.diff(seg_t, prepend=seg_t[0])

        on   = seg_i >= hi
        off  = ~on
        t_on = dt[on].sum()
        t_off= dt[off].sum()
        if t_off:
            pw_list.append(t_on/t_off)

        ih = seg_i[on].mean() if np.any(on) else np.nan
        il = abs(seg_i[seg_i <= lo]).mean() if np.any(seg_i <= lo) else np.nan
        if ih and il:
            io_list.append(ih/il)

    def stability(arr):
        return np.std(arr)/np.mean(arr) if len(arr)>1 and np.mean(arr)!=0 else np.nan

    # 4 ── summary row --------------------------------------------------------
    out = pd.DataFrame([dict(
        AvgFreq_Hz      = freq.mean(),
        Jitter_s        = periods.std(),
        Stability       = freq.std()/freq.mean(),
        PulseWidthRatio = np.mean(pw_list) if pw_list else np.nan,
        PW_Stability    = stability(pw_list),
        OnOffRatio      = np.mean(io_list) if io_list else np.nan,
        OnOff_Stability = stability(io_list),
        PeakCurrent_A   = sig.max(),
        LowCurrent_A    = abs(sig.min())
    )])

    if verbose:
        print("\n=== Metrics ===")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    # 5 ── annotated plot -----------------------------------------------------
    if show_plot:
        plt.figure(figsize=(9,4))
        plt.plot(time, sig, label=wave_name)
        plt.plot(time[peaks], sig[peaks], 'rx', ms=6, label='Peaks')
        plt.fill_between(time, sig.min(), sig,
                         where=sig>=hi, color='tab:orange', alpha=.15,
                         label='On region')
        plt.title(f'{wave_name} waveform')
        plt.xlabel('Time (s)'); plt.ylabel(wave_name)
        plt.grid(True, alpha=.3); plt.legend()
        plt.tight_layout(); plt.show()

    return out


metrics_i = analyze_single_waveform("multivibrator1.7.4.txt",
                                    wave_col=1,
                                    wave_name='I',
                                    hi_frac=0.6,
                                    lo_frac=0.2,
                                    show_plot=True,
                                    verbose=True)