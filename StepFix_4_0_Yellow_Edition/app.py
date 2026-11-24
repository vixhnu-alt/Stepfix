import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
# Self-contained analysis functions are embedded below

import io
import numpy as np
import pandas as pd

SENSORS = [f'{foot}_sensor_{i}' for foot in ('left','right') for i in range(1,9)]
COLS = ['time_s'] + SENSORS + ['left_total','right_total']
FEATURES = ['duration_s','left_peak_N','right_peak_N','peak_asymmetry_pct','cadence_spm','left_stride_s','right_stride_s','left_stride_cv_pct','right_stride_cv_pct','left_stance_s','right_stance_s','left_swing_s','right_swing_s','left_swing_cv_pct','right_swing_cv_pct','left_mean_force_N','right_mean_force_N','mean_force_asymmetry_pct']

DISPLAY_FEATURES = [k for k in FEATURES if k not in ('left_stride_s', 'right_stride_s')]

def visible_metrics(metrics):
    return {k: v for k, v in metrics.items() if k not in ('left_stride_s', 'right_stride_s')}

def read_gait(data, filename='recording'):
    text = data.decode('utf-8-sig',errors='replace')
    lines=[x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith('#')]
    if not lines: raise ValueError('No data rows')
    sep=',' if ',' in lines[0] else r'\s+'
    df=pd.read_csv(io.StringIO('\n'.join(lines)),sep=sep,header=None,engine='python')
    names={19:COLS,3:['time_s','left_total','right_total'],7:['time_s','left_heel','left_toe','right_heel','right_toe','left_total','right_total']}
    if df.shape[1] not in names: raise ValueError(f'{df.shape[1]} columns found; expected 19, 3, or 7')
    if pd.to_numeric(df.iloc[0],errors='coerce').isna().any(): df=df.iloc[1:].reset_index(drop=True)
    df.columns=names[df.shape[1]]
    numeric=df.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    missing=numeric.isna().sum().to_dict()
    df=numeric.dropna(subset=['time_s','left_total','right_total']).sort_values('time_s').drop_duplicates('time_s').reset_index(drop=True)
    if len(df)<10: raise ValueError('At least 10 valid samples required')
    if (df[['left_total','right_total']]<0).any().any(): raise ValueError('Negative total force detected')
    dt=np.diff(df.time_s.to_numpy())
    if not len(dt) or np.median(dt)<=0: raise ValueError('Invalid timestamps')
    hz=float(1/np.median(dt))
    warnings=[]
    if not 95<=hz<=105: warnings.append(f'Observed sampling rate {hz:.2f} Hz differs from expected ~100 Hz')
    if len(dt)>1 and np.max(dt)>2*np.median(dt): warnings.append('Irregular time gaps detected')
    if df[SENSORS].isna().any().any() if all(s in df for s in SENSORS) else False: warnings.append('Some individual sensor samples are missing')
    return df, {'format':f'{len(df.columns)} columns','samples':len(df),'sampling_hz':hz,'missing_before_cleaning':missing,'warnings':warnings,'filename':filename}

def contact_events(force,time,threshold):
    smooth=pd.Series(force).rolling(5,center=True,min_periods=1).median().to_numpy()
    contact=np.isfinite(smooth)&(smooth>=threshold)
    edges=np.diff(contact.astype(int),prepend=0,append=0)
    starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
    dt=float(np.median(np.diff(time)))
    bouts=[(int(a),int(b)) for a,b in zip(starts,ends) if (b-a)*dt>=0.1]
    onsets=np.array([time[a] for a,b in bouts]); strides=np.diff(onsets)
    strides=strides[(strides>=0.35)&(strides<=3.0)]
    stances=np.array([(b-a)*dt for a,b in bouts]); stances=stances[(stances>=0.1)&(stances<=3.0)]
    swings=np.array([time[bouts[i+1][0]]-(time[min(b,len(time)-1)]+dt) for i,(a,b) in enumerate(bouts[:-1])]); swings=swings[(swings>=0.05)&(swings<=3.0)]
    return strides,stances,swings,len(bouts)

def avg(a): return float(np.mean(a)) if len(a) else np.nan
def cv(a): return float(100*np.std(a,ddof=1)/np.mean(a)) if len(a)>1 and np.mean(a)>0 else np.nan
def asym(a,b): return float(200*abs(a-b)/(a+b)) if np.isfinite(a+b) and a+b>0 else np.nan

def analyze(df,threshold=20):
    t=df.time_s.to_numpy(float); l=df.left_total.to_numpy(float); r=df.right_total.to_numpy(float)
    sl,tl,wl,nl=contact_events(l,t,threshold); sr,tr,wr,nr=contact_events(r,t,threshold)
    duration=float(t[-1]-t[0]); lp=float(np.max(l));rp=float(np.max(r));lm=float(np.mean(l));rm=float(np.mean(r))
    return dict(zip(FEATURES,[duration,lp,rp,asym(lp,rp),60*(nl+nr)/duration if duration else np.nan,avg(sl),avg(sr),cv(sl),cv(sr),avg(tl),avg(tr),avg(wl),avg(wr),cv(wl),cv(wr),lm,rm,asym(lm,rm)])) | {'left_contacts':nl,'right_contacts':nr}

def sensor_summary(df):
    if not all(c in df for c in SENSORS): return None
    return pd.DataFrame({'sensor':SENSORS,'mean_N':[df[c].mean() for c in SENSORS],'peak_N':[df[c].max() for c in SENSORS]})

def sensor_failure(df,sensor):
    if sensor not in SENSORS or sensor not in df: raise ValueError('Sensor not available')
    altered=df.copy(); altered[sensor]=0.0
    side=sensor.split('_')[0]; cols=[f'{side}_sensor_{i}' for i in range(1,9)]
    # Counterfactual reconstructed total; native total is kept unchanged in the original.
    altered[f'{side}_total']=altered[cols].sum(axis=1,min_count=8)
    return altered


st.set_page_config(page_title='Step Fix 4.0 | Biomedical Gait Analytics',page_icon='👣',layout='wide',initial_sidebar_state='expanded')

st.markdown("""<style>
:root{--sun:#ffcc24;--gold:#f6ae00;--ink:#12151b}
.stApp{background:radial-gradient(ellipse at 90% 0%,#242316 0%,#101217 42%,#0d0f14 100%);color:#f6f7fb}
[data-testid="stSidebar"]{background:#17191f;border-right:1px solid #393321}
[data-testid="stSidebar"] *{color:#f4f2e8}
.hero{padding:31px 34px 28px;background:linear-gradient(120deg,#26251b 0%,#171a20 72%);border:1px solid #665520;border-radius:22px;margin-bottom:22px;box-shadow:0 12px 40px #0005}
.hero .eyebrow{font-size:12px;letter-spacing:2.3px;font-weight:800;color:#e4b938}
.hero h1{font-size:clamp(38px,5vw,66px);font-weight:900;letter-spacing:-2px;color:#fff;margin:8px 0}
