"""
sweep_onoff_vs_pw.py

Hover text shows:
    • Frequency  (Hz)
    • Duty
    • Peak (µA)
    • Low  (µA)
    • I_on/off  (ratio)
"""

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import find_peaks
try:
    import mplcursors                   # for hover
except ModuleNotFoundError:
    mplcursors = None


def sweep_onoff_vs_pw(filename,
                      hi_fracs=(0.6, 0.5, 0.4, 0.3),
                      lo_frac=0.2,
                      verbose=True,
                      interactive=True,
                      thresh=0.01):

    one_peak = lambda s,h,l: np.asarray(
        [p for p in (lambda c= find_peaks(s, height=h)[0]:
                     [pp for pp in c
                      if s[ (c[np.where(c==pp)[0][0]-1] if np.where(c==pp)[0][0]>0 else 0):pp ].min() < l ])()], int)

    stab = lambda a: np.std(a)/np.mean(a) if len(a)>1 and np.mean(a) else np.nan

    def analyse(t, i, prm):
        for hf in hi_fracs:
            hi = i.min()+hf*(i.max()-i.min())
            lo = i.min()+lo_frac*(i.max()-i.min())
            pk = one_peak(i, hi, lo)
            if pk.size>=2: break
        else: return None  # reject silent

        per,f = np.diff(t[pk]), 1/np.diff(t[pk])
        duties, ton_list, io_list = [], [], []
        for a,b in zip(pk[:-1],pk[1:]):
            seg_t, seg_i = t[a:b], i[a:b]
            dt = np.diff(seg_t, prepend=seg_t[0])
            on = seg_i>=hi
            ton, toff = dt[on].sum(), dt[~on].sum()
            duties.append( ton/(ton+toff) )
            ton_list.append(ton)
            io_list.append( seg_i[on].mean() /
                            abs(seg_i[seg_i<=lo]).mean() )

        row = dict(**prm,
            AvgFreq_Hz      = f.mean(),
            Jitter_s        = per.std(),
            Stability       = f.std()/f.mean(),
            DutyCycle       = np.mean(duties),
            Duty_Stability  = stab(duties),
            PulseWidth_s    = np.mean(ton_list),
            OnOffRatio      = np.mean(io_list),
            OnOff_Stability = stab(io_list),
            PeakCurrent_A   = i.max(),
            LowCurrent_A    = i.min() )

        row['Rejected'] = any( row[k]>thresh for k in
                               ('Jitter_s','Stability',
                                'Duty_Stability','OnOff_Stability')
                               if not np.isnan(row[k]) )
        return row

    # -------- parse file ------------------------------------------
    rows,tbuf,ibuf,cur = [],[],[],{}
    with open(filename,'r',encoding='latin1') as f:
        for raw in f:
            line=raw.strip()
            if line.startswith("Step Information:"):
                if tbuf:
                    r=analyse(np.asarray(tbuf),np.asarray(ibuf),cur)
                    if r: rows.append(r)
                tbuf,ibuf=[],[]
                cur=dict(re.findall(r'(\w+)=([\w\.]+)',line));continue
            if '\t' in line and not line.startswith('time'):
                try:
                    t,i=map(float,line.split('\t')[:2])
                    tbuf.append(t); ibuf.append(i)
                except ValueError: pass
        if tbuf:
            r=analyse(np.asarray(tbuf),np.asarray(ibuf),cur)
            if r: rows.append(r)

    df=pd.DataFrame(rows)

    # -------- pretty print ----------------------------------------
    if verbose:
        disp=df.copy()
        disp['Peak_uA']=disp['PeakCurrent_A']*1e6
        disp['Low_uA'] =disp['LowCurrent_A']*1e6
        print(disp.drop(columns=['PeakCurrent_A','LowCurrent_A'])
                  .to_string(index=False,
                             float_format=lambda x:f"{x:.6g}"))

    # -------- plotting (accepted only) ----------------------------
    acc=df[~df.Rejected]; f=acc['AvgFreq_Hz'].to_numpy(float)
    colours=plt.cm.viridis((np.clip(f,1,10)-1)/9); colours[f>10]=[1,0,0,1]

    fig,ax=plt.subplots(figsize=(6,5))
    sc=ax.scatter(acc['DutyCycle'],acc['OnOffRatio'],c=colours,s=50,alpha=.9)
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis',
                                       norm=plt.Normalize(1,10)),
                 ax=ax,label='Avg Frequency (Hz, 1–10)')
    ax.set_xlabel('Duty Cycle  (T_on / Period)')
    ax.set_ylabel('On-Off Current Ratio  (I_high / I_low)')
    ax.set_title('On-Off vs Duty  (red = >10 Hz, rejected omitted)')
    ax.grid(alpha=.3)

    # hover tooltip with µA
    if interactive and mplcursors:
        pk_uA=(acc['PeakCurrent_A']*1e6).to_numpy()
        lo_uA=(acc['LowCurrent_A']*1e6 ).to_numpy()
        cur=mplcursors.cursor(sc,hover=True)
        @cur.connect("add")
        def _(sel):
            idx=sel.index
            sel.annotation.set_text(
               f"{f[idx]:.3f} Hz\n"
               f"Duty {acc.iloc[idx]['DutyCycle']:.3f}\n"
               f"I_pk {pk_uA[idx]:.1f} µA\n"
               f"I_lo {lo_uA[idx]:.1f} µA\n"
               f"I_on/off {acc.iloc[idx]['OnOffRatio']:.2f}")

    plt.tight_layout(); plt.show()
    return df


# standalone test
if __name__=="__main__":
    sweep_onoff_vs_pw("multivibrator1.8.1.txt",
                      hi_fracs=(0.6,0.5,0.4,0.3),
                      lo_frac=0.2,
                      verbose=True,
                      interactive=True,
                      thresh=0.05)
