import os, re, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks
import plotly.offline as pyo, plotly.graph_objs as go
try: import mplcursors
except ModuleNotFoundError: mplcursors = None
from pathlib import Path
from sklearn.cluster import KMeans


# ────────────────────────────────────────────────────────────────────
def sweep_onoff_vs_pw(
        filename: str,
        hi_fracs=(0.6, 0.5, 0.4, 0.3),
        lo_frac: float = 0.2,
        thresh: float =0.15,  # threshold for stability metrics
        show_rejected: bool = False,      
        interactive: bool = True):
    

    # ── helpers ──────────────────────────────────────────────
    def one_peak(sig, hi, lo):
        cand, _ = find_peaks(sig, height=hi)
        keep, last = [], 0
        for p in cand:
            if sig[last:p].min() < lo:
                keep.append(p); last = p
        return np.asarray(keep, int)

    stab = lambda a: np.std(a)/np.mean(a) if len(a)>1 and np.mean(a) else np.nan
    rows = []

    fp = Path(filename)
    if not fp.is_file():                       # not where we were called from …
        fp = fp.parent / "raw" / fp.name       # … so try sibling "raw/" dir
    filename = str(fp)                         # keep the rest of the code happy

    src_path   = Path(filename).expanduser().resolve()
    ver_match  = re.search(r'(\d+(?:\.\d+)+)$', src_path.stem)      # e.g. 1.8.1
    ver_folder = ver_match.group(1) if ver_match else src_path.stem
    # out_base   = src_path.parent / ver_folder                      # “…/1.8.1/”
    # wf_dir     = out_base / "waveforms"
    # out_base.mkdir(parents=True, exist_ok=True)
    # wf_dir.mkdir(exist_ok=True)
    base_dir = src_path.parent
    if base_dir.name == "raw":        # the .txt lives in raw/
        base_dir = base_dir.parent    # step one level up

    out_base = base_dir / ver_folder          # e.g. “…/1.8.1/”
    wf_dir   = out_base / "waveforms"
    out_base.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(exist_ok=True)


    def save_plotly_wave(t, i, run_id):
        fig = go.Figure(go.Scatter(x=t, y=i*1e6, mode='lines',
                                   line=dict(width=1), name='Current (µA)'))
        fig.update_layout(title=f'Run {run_id} – Current vs Time',
                          xaxis_title='Time (s)', yaxis_title='Current (µA)',
                          template="simple_white", height=450, width=850)
        path = wf_dir / f"run_{run_id}.html"
        pyo.plot(fig, filename=str(path), auto_open=False, include_plotlyjs='cdn')
        return f"waveforms/{path.name}"          # ← relative link for the table

    # ── analyse one step block ───────────────────────────────
    def analyse_block(rid, t, i, prm):
        rng = np.ptp(i)
        for hf in hi_fracs:
            hi, lo = i.min() + hf*rng, i.min() + lo_frac*rng
            pk = one_peak(i, hi, lo)
            if pk.size >= 2:
                break
        else:
            pk = np.array([])

        if pk.size < 2:
            duty = tone = onoff = np.nan
            f_mean = f_std = per_std = np.nan
            Ion_avg = charge = np.nan
        else:
            per, freq = np.diff(t[pk]), 1/np.diff(t[pk])
            f_mean, f_std, per_std = freq.mean(), freq.std(), per.std()
            duties, tones, ior = [], [], []
            for a, b in zip(pk[:-1], pk[1:]):
                seg_t, seg_i = t[a:b], i[a:b]
                dt = np.diff(seg_t, prepend=seg_t[0])
                on = seg_i >= hi; off = ~on
                if on.any() and off.any():
                    Ton, Toff = dt[on].sum(), dt[off].sum()
                    duties.append(Ton/(Ton+Toff)); tones.append(Ton)
                    ior.append(seg_i[on].mean()/abs(seg_i[seg_i<=lo]).mean())
            duty   = np.nanmean(duties)
            tone   = np.nanmean(tones)
            onoff  = np.nanmean(ior)
            Ion_avg = i[i >= hi].mean() if np.isfinite(tone) else np.nan
            charge  = tone * Ion_avg if np.isfinite(Ion_avg) else np.nan

        row = dict(RunID=rid, **prm,
               AvgFreq_Hz=f_mean,
               Jitter_s=per_std,
               Stability=f_std/f_mean if f_mean else np.nan,
               DutyCycle=duty,
               Duty_Stability = stab(duties) if pk.size else np.nan,
               PulseWidth_s   = tone,
               OnOffRatio     = onoff,
               OnOff_Stability= stab(ior)    if pk.size else np.nan,
               Peak_uA=i.max()*1e6,
               Low_uA =i.min()*1e6,
               PulseCharge_C  = charge)

        # ---- Updated rejection logic ----
        key_metrics = ['AvgFreq_Hz','DutyCycle','PulseWidth_s','OnOffRatio']
        nan_or_freq = (any(np.isnan(row.get(k, np.nan)) for k in key_metrics) or
                    (not np.isnan(row.get('AvgFreq_Hz', np.nan)) and row['AvgFreq_Hz'] > 10))

        thresh_reject = any(row[k] > thresh for k in
                            ('Jitter_s','Stability','Duty_Stability','OnOff_Stability')
                            if not np.isnan(row[k]))

        row['Rejected'] = nan_or_freq or thresh_reject
        # generate plot for every run
        row['Plot'] = f'<a href="{save_plotly_wave(t, i, rid)}" target="_blank">Plot</a>'
        rows.append(row)

    # ── parse LT-Spice export ─────────────────────────────────
    rid = 0; buf_t, buf_i, prm = [], [], {}
    with open(filename,'r',encoding='latin1') as fin:
        for raw in fin:
            line=raw.strip()
            if line.startswith("Step Information:"):
                if buf_t: analyse_block(rid, np.asarray(buf_t), np.asarray(buf_i), prm)
                rid += 1; buf_t, buf_i = [], []
                prm = dict(re.findall(r'(\w+)=([\w\.]+)', line)); continue
            if '\t' in line and not line.startswith('time'):
                try:
                    t_val, i_val = map(float, line.split('\t')[:2])
                    buf_t.append(t_val); buf_i.append(i_val)
                except ValueError: pass
    if buf_t: analyse_block(rid, np.asarray(buf_t), np.asarray(buf_i), prm)

    df = pd.DataFrame(rows)
    df['Charge_uC'] = df.PulseCharge_C * 1e6

    # ── write scrollable DataTables HTML ───────────────────────
    # html_path = os.path.splitext(filename)[0] + "_table.html"
    html_path = out_base / f"{ver_folder}_table.html"
    with open(html_path,'w') as f:
        f.write(f"""<!DOCTYPE html><html><head>
<link rel="stylesheet"
 href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<style>table.dataTable thead th{{position:sticky;top:0;background:#eee}}</style>
</head><body>
<table id="tbl" class="display nowrap" style="width:100%">
{df.to_html(index=False, escape=False)}
</table>
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>
$(document).ready(function(){{ $('#tbl').DataTable({{scrollY:'70vh',
 scrollX:true, paging:false}}); }});
</script></body></html>""")
    import webbrowser, pathlib; webbrowser.open(pathlib.Path(html_path).resolve().as_uri())
    print("🟢  Table with interactive links →", html_path)


    # ─── adaptive colour helper (min–max of data) ───────────────────
    def _freq_colors(freq_arr):
        finite = freq_arr[np.isfinite(freq_arr)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = finite.min(), finite.max()
            if vmin == vmax:                       # flat → widen slightly
                vmax = vmin + 1e-3
        norm = plt.Normalize(vmin, vmax)
        col  = plt.cm.viridis(norm(freq_arr))
        return col, norm

    REJ_CLR   = 'grey'
    min_on_uA = 100  # Minimum peak current to consider a run qualified

    # ── PLOT 1 : Duty‑cycle vs On/Off ratio ────────────────────────
    acc1 = df[~df.Rejected].reset_index(drop=True)
    rej1 = df[df.Rejected ].reset_index(drop=True)

    col1, norm1 = _freq_colors(acc1.AvgFreq_Hz.values)

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sc_acc1 = ax1.scatter(acc1.DutyCycle, acc1.OnOffRatio,
                        c=col1, s=80, alpha=.95,
                        edgecolor='k', linewidth=.4, label='Accepted')
    scatter_list1 = [sc_acc1]

    if show_rejected and not rej1.empty:
        sc_rej1 = ax1.scatter(rej1.DutyCycle, rej1.OnOffRatio,
                              marker='x', color=REJ_CLR, s=50, alpha=.25,
                              label='Rejected', zorder=1)
        scatter_list1.append(sc_rej1)

    # Always draw colored points on top
    sc_acc1.set_zorder(2)
    ax1.legend(frameon=False)

    ax1.set(xlabel=r'Duty Cycle $T_{\mathrm{on}}/T_{\mathrm{period}}$',
            ylabel=r'On‑Off Ratio $(I_{\mathrm{high}}/I_{\mathrm{low}})$',
            title='Duty vs On/Off ratio')
    ax1.grid(alpha=.3)

    fig1.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm1),
              ax=ax1,
              label=f'Avg Freq (Hz, {norm1.vmin:.2f}–{norm1.vmax:.2f})')

    if interactive and mplcursors:
        cur1 = mplcursors.cursor([sc_acc1, sc_rej1] if show_rejected else [sc_acc1], hover=True)
        @cur1.connect("add")
        def _(sel):
            src = acc1 if sel.artist is sc_acc1 else rej1
            r   = src.iloc[sel.index]
            reason = []
            if any(np.isnan(r[k]) for k in ['AvgFreq_Hz','DutyCycle','PulseWidth_s','OnOffRatio']):
                reason.append("NaN values")
            if not np.isnan(r.AvgFreq_Hz) and r.AvgFreq_Hz > 10:
                reason.append(">10 Hz")
            if r.Rejected and not reason:
                reason.append("Stability threshold")

            reason_txt = f"\nRejected: {', '.join(reason)}" if r.Rejected else ""
            sel.annotation.set_text(
                f"Run {int(r.RunID)}{reason_txt}\n"
                f"{r.AvgFreq_Hz:.2f} Hz\n"
                f"Duty {r.DutyCycle:.3f}\n"
                f"Ipk {r.Peak_uA:.1f} µA\n"
                f"Ilo {r.Low_uA:.1f} µA\n"
                f"Ion/off {r.OnOffRatio:.2f}")
            sel.annotation.get_bbox_patch().set(fc="#ffffcc", alpha=.9)
            if sel.annotation.arrow_patch:
                sel.annotation.arrow_patch.set_visible(False)

    # ── PLOT 2 : Pulse‑width vs Off‑current ────────────────────────
    df['Off_uA'] = df.Low_uA.abs()
    qual   = df[(~df.Rejected) & (df.Peak_uA >= min_on_uA)].reset_index(drop=True)
    nonqual= df[(df.Peak_uA  < min_on_uA) | df.Rejected    ].reset_index(drop=True)

    col2, norm2 = _freq_colors(qual.AvgFreq_Hz.values)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    # faint non‑qualified × first
    ax2.scatter(nonqual.PulseWidth_s, nonqual.Off_uA,
                marker='x', color=REJ_CLR, s=50, alpha=.25,
                label='Not qualified', zorder=1)
    # coloured qualified circles on top
    sc_q = ax2.scatter(qual.PulseWidth_s, qual.Off_uA,
                    c=col2, s=90, alpha=.95,
                    edgecolor='k', linewidth=.4,
                    label=f'Ipk ≥ {min_on_uA} µA', zorder=2)

    ax2.set(xlabel=r'Pulse‑width $T_{\mathrm{on}}$ (s)',
            ylabel=r'Off‑current $|I_{\mathrm{low}}|$ (µA)',
            title='Pulse‑width vs Off‑current')
    ax2.set_yscale('log')
    ax2.grid(alpha=.3, which='both')
    ax2.legend(frameon=False)

    fig2.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm2),
              ax=ax2,
              label=f'Avg Freq (Hz, {norm2.vmin:.2f}–{norm2.vmax:.2f})')

    if interactive and mplcursors:
        cur2 = mplcursors.cursor([sc_q, ax2.collections[0]], hover=True)  # qualified + grey X's
        @cur2.connect("add")
        def _(sel):
            src = qual if sel.artist is sc_q else nonqual
            r   = src.iloc[sel.index]
            reason = []
            if any(np.isnan(r[k]) for k in ['AvgFreq_Hz','DutyCycle','PulseWidth_s','OnOffRatio']):
                reason.append("NaN values")
            if not np.isnan(r.AvgFreq_Hz) and r.AvgFreq_Hz > 10:
                reason.append(">10 Hz")
            if r.Peak_uA < min_on_uA:
                reason.append("Low peak current")
            if r.Rejected and not reason:
                reason.append("Stability threshold")

            reason_txt = f"\nRejected: {', '.join(reason)}" if (r.Rejected or reason) else ""
            sel.annotation.set_text(
                f"Run {int(r.RunID)}{reason_txt}\n"
                f"{r.AvgFreq_Hz:.2f} Hz\n"
                f"PW  {r.PulseWidth_s:.3e} s\n"
                f"Ilo {r.Off_uA:.1f} µA\n"
                f"Duty {r.DutyCycle:.3f}")
            sel.annotation.get_bbox_patch().set(fc="#ffffcc", alpha=.9)
            if sel.annotation.arrow_patch:
                sel.annotation.arrow_patch.set_visible(False)




    # ── PLOT 3 : Charge vs Off‑current ────────────────────────────
    acc3 = df[~df.Rejected].reset_index(drop=True)
    rej3 = df[df.Rejected ].reset_index(drop=True)

    col3, norm3 = _freq_colors(acc3.AvgFreq_Hz.values)

    fig3, ax3 = plt.subplots(figsize=(6, 5))

    # Draw rejected (grey x) points first, behind accepted
    if show_rejected and not rej3.empty:
        ax3.scatter(rej3.Charge_uC, rej3.Off_uA,
                    marker='x', color=REJ_CLR, s=50, alpha=.25,
                    label='Rejected', zorder=1)

    # Draw accepted (colored) points on top
    sc_acc3 = ax3.scatter(acc3.Charge_uC, acc3.Off_uA,
                          c=col3, s=80, alpha=.95,
                          edgecolor='k', linewidth=.4,
                          label='Accepted', zorder=2)

    ax3.set(xlabel=r'Charge per pulse $Q$ (µC)',
            ylabel=r'Off‑current $|I_{\mathrm{low}}|$ (µA)',
            title='Charge vs Off‑current')
    ax3.set_xscale('linear')
    ax3.set_yscale('log')
    ax3.grid(alpha=.3, which='both')
    ax3.legend(frameon=False)

    fig3.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm3),
                  ax=ax3,
                  label=f'Avg Freq (Hz, {norm3.vmin:.2f}–{norm3.vmax:.2f})')

    # optional hover labels
    if interactive and mplcursors:
        # Accepted (colored) + Rejected (grey) points
        sc_rej3 = None
        if show_rejected and not rej3.empty:
            sc_rej3 = ax3.collections[0]  # first scatter is the grey X's

        cur3 = mplcursors.cursor(
            [sc_acc3, sc_rej3] if sc_rej3 else [sc_acc3], hover=True
        )
        @cur3.connect("add")
        def _(sel):
            src = acc3 if sel.artist is sc_acc3 else rej3
            r   = src.iloc[sel.index]
            reason = []
            if any(np.isnan(r[k]) for k in ['AvgFreq_Hz','DutyCycle','PulseWidth_s','OnOffRatio']):
                reason.append("NaN values")
            if not np.isnan(r.AvgFreq_Hz) and r.AvgFreq_Hz > 10:
                reason.append(">10 Hz")
            if r.Rejected and not reason:
                reason.append("Stability threshold")

            reason_txt = f"\nRejected: {', '.join(reason)}" if r.Rejected else ""
            sel.annotation.set_text(
                f"Run {int(r.RunID)}{reason_txt}\n"
                f"{r.AvgFreq_Hz:.2f} Hz\n"
                f"Charge {r.Charge_uC:.3e} µC\n"
                f"Ilo {r.Off_uA:.1f} µA")
            sel.annotation.get_bbox_patch().set(fc="#ffffcc", alpha=.9)
            if sel.annotation.arrow_patch:
                sel.annotation.arrow_patch.set_visible(False)

    plt.tight_layout()

        # ── SAVE PLOTS AS SVG ─────────────────────────────────────
    fig1.savefig(out_base / "duty_vs_onoff.svg",
                 format="svg", bbox_inches='tight')
    fig2.savefig(out_base / "pw_vs_offcurrent.svg",
                 format="svg", bbox_inches='tight')
    fig3.savefig(out_base / "charge_vs_offcurrent.svg",
                 format="svg", bbox_inches='tight')  
    
        # ── Find and cluster runs near 6 µC ─────────────────────────────
    target_charge = 6.0   # µC target
    target_freq   = 1.5   # Hz target
    num_near      = 20    # Keep 20 closest runs for clustering
    n_clusters    = 3     # Number of clusters (can adjust)

    if {'Charge_uC','AvgFreq_Hz'}.issubset(df.columns) and df[['Charge_uC','AvgFreq_Hz']].notna().any().any():
        # Calculate distance to target for sorting
        df['Distance'] = np.sqrt((df['Charge_uC'] - target_charge)**2 +
                                 (df['AvgFreq_Hz'] - target_freq)**2)
        near_df = df.sort_values('Distance').head(num_near).copy()

        # Prepare features for clustering (charge and frequency)
        features = near_df[['Charge_uC','AvgFreq_Hz']].to_numpy()

        # Run KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        near_df['Cluster'] = kmeans.fit_predict(features)

        # Summarize each cluster
        cluster_summary = near_df.groupby('Cluster').apply(
            lambda g: pd.Series({
                'ClusterCenter_Charge_uC': g['Charge_uC'].mean(),
                'ClusterCenter_Freq_Hz':   g['AvgFreq_Hz'].mean(),
                'RunIDs': ', '.join(g['RunID'].astype(str)),
                'RejectedRuns': ', '.join(g.loc[g['Rejected'], 'RunID'].astype(str)),
                'AcceptedRuns': ', '.join(g.loc[~g['Rejected'], 'RunID'].astype(str))
            })
        ).reset_index()

        # Save outputs
        near_df_path = out_base / f"closest_to_{target_charge}uC_{target_freq}Hz_detailed.csv"
        summary_path = out_base / f"closest_to_{target_charge}uC_{target_freq}Hz_summary.csv"
        near_df.to_csv(near_df_path, index=False)
        cluster_summary.to_csv(summary_path, index=False)
        print(f"🟢  Saved closest runs to ({target_charge} µC, {target_freq} Hz):")
        print(f"     Detailed list → {near_df_path}")
        print(f"     Cluster summary → {summary_path}")


    
    plt.show()
    return df



if __name__ == "__main__":
    sweep_onoff_vs_pw("multivibrator1.9.5.txt",
                      show_rejected=True,      # grey × markers
                      interactive=True)