import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.signal import find_peaks

def _to_float(txt):
    if not txt or not isinstance(txt, str):
        return np.nan
    txt = txt.replace('µ', 'u').lower()
    mul = {'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,
           'k':1e3,'meg':1e6,'g':1e9}
    for k,v in mul.items():
        if txt.endswith(k):
            return float(txt[:-len(k)])*v
    return float(txt)

def _one_peak_cycle(v, hi, lo):
    """
    Return index of exactly one peak per cycle.
    Arms when v crosses above `hi`, records the highest point until
    v dips below `lo`, then re-arms. Works even if the pulse is 1 sample wide.
    """
    peaks = []
    armed = False
    max_idx = -1
    max_val = -np.inf

    for i, val in enumerate(v):
        if not armed and val >= hi:            # rising edge crosses HI
            armed = True
            max_idx = i
            max_val = val
        elif armed:
            # still inside the high part of the pulse
            if val > max_val:                  # track absolute maximum
                max_val = val
                max_idx = i
            if val <= lo:                      # pulse ended (fell below LO)
                peaks.append(max_idx)
                armed = False                  # re-arm for next cycle
    return np.asarray(peaks, dtype=int)

def analyze_double_sweep_file(fname,
                              p1='C1', p2='R2',
                              PEAK_HI=1.8, RESET_LO=0.2, V_ON=1.8):

    res=[]; wt,wv,wl,wp = [],[],[],[]
    tbuf,vbuf=[] ,[]
    cur1=cur2=None

    with open(fname,'r',encoding='latin1') as f:
        lines=f.readlines()

    def fin():
        if not tbuf: return
        t=np.asarray(tbuf); v=np.asarray(vbuf)
        pk=_one_peak_cycle(v,PEAK_HI,RESET_LO)
        wt.append(t); wv.append(v); wl.append(f"{p1}={cur1}, {p2}={cur2}"); wp.append(pk)
        d={p1:cur1,p2:cur2}
        if pk.size>=2:
            per=np.diff(t[pk]); f=1/per
            d.update(AvgFreq_Hz=f.mean(),Jitter_s=per.std(),Stability=f.std()/f.mean())
        else:
            d.update(AvgFreq_Hz=np.nan,Jitter_s=np.nan,Stability=np.nan)
        dt=np.diff(t,prepend=t[0]); on=v>V_ON
        d['PulseWidthRatio']=dt[on].sum()/dt[~on].sum() if dt[~on].sum() else np.nan
        res.append(d)

    for raw in lines:
        line=raw.strip()
        if line.startswith("Step Information:"):
            fin(); tbuf,vbuf=[],[]
            m1=re.search(rf'{p1}=([^\s\]]+)',line)
            m2=re.search(rf'{p2}=([^\s\]]+)',line)
            cur1=m1.group(1) if m1 else None
            cur2=m2.group(1) if m2 else None
        elif '\t' in line and not line.startswith('time'):
            try:
                t,v=map(float,line.split('\t')[:2]); tbuf.append(t); vbuf.append(v)
            except ValueError: pass
    fin()

    df=pd.DataFrame(res)
    df[f'{p1}_F']=df[p1].apply(_to_float)
    df[f'{p2}_F']=df[p2].apply(_to_float)
    print(df[[p1,p2,'AvgFreq_Hz','Jitter_s','Stability','PulseWidthRatio']])

    f_grid = df.pivot(index=f'{p1}_F', columns=f'{p2}_F', values='AvgFreq_Hz')
    pw_grid= df.pivot(index=f'{p1}_F', columns=f'{p2}_F', values='PulseWidthRatio')

    # engineering formatters
    fmt_x = EngFormatter(unit='Ω', places=0)  # adjust units if p2 not a resistor
    fmt_y = EngFormatter(unit='F', places=0)  # adjust units if p1 not a capacitor

    def _heat(ax, grid, title, cbar_lab):
        im=ax.imshow(grid,origin='lower',aspect='auto',
                     extent=[grid.columns.min(),grid.columns.max(),
                             grid.index.min(),grid.index.max()])
        ax.set_xlabel(p2); ax.set_ylabel(p1); ax.set_title(title)
        ax.xaxis.set_major_formatter(fmt_x); ax.yaxis.set_major_formatter(fmt_y)
        plt.colorbar(im, ax=ax, label=cbar_lab)

    fig_h,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    _heat(ax1,f_grid,'Avg Frequency','Hz')
    _heat(ax2,pw_grid,'Pulse-Width Ratio','T_on/T_off')

    # stacked waveforms
    n=len(wt); fig_w,axs=plt.subplots(n,1,figsize=(10,2.2*n),sharex=True)
    axs=[axs] if n==1 else axs
    for ax,t,v,lab,pk in zip(axs,wt,wv,wl,wp):
        ax.plot(t,v,label=lab)
        if pk.size: ax.plot(t[pk],v[pk],'rx',ms=6)
        ax.set_ylabel('V'); ax.set_title(lab); ax.grid(True); ax.legend()
    axs[-1].set_xlabel('Time (s)')
    fig_w.tight_layout(); plt.show()
    return df

df = analyze_double_sweep_file("multivibrator1.6.t.txt",
                               p1='C1', p2='R2',
                               PEAK_HI=1.8, RESET_LO=0.2, V_ON=1.8)
