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
.hero h1 span{color:#ffcc24}.hero h1 small{font-size:24px;letter-spacing:0;color:#c7c5ba}
.hero p{color:#d3cdb8;font-size:17px;margin:0}
h1,h2,h3{letter-spacing:-.5px}h2,h3{color:#ffdc65!important}
[data-testid="stMetric"]{background:#202128;border:1px solid #54471e;border-radius:17px;padding:18px;box-shadow:0 8px 20px #0003}
[data-testid="stMetricValue"]{color:#ffcf36}
[data-testid="stTabs"] button[aria-selected="true"]{color:#ffcf36!important;border-bottom:3px solid #ffcc24!important}
.stButton>button[kind="primary"],div[data-testid="stDownloadButton"] button{background:#ffcc24!important;color:#1b1a13!important;border:0!important;border-radius:12px!important;font-weight:750!important}
.stButton>button:hover,div[data-testid="stDownloadButton"] button:hover{background:#ffe27a!important}
[data-testid="stFileUploader"]{border:1px dashed #9b812f;border-radius:15px;padding:7px}
[data-testid="stAlert"]{border-radius:13px}
</style>""",unsafe_allow_html=True)

st.markdown('''<div class="hero"><div class="eyebrow">BIOMEDICAL RESEARCH PLATFORM • VERSION 4.0</div><h1>👣 STEP <span>FIX</span> <small>4.0</small></h1><p>Intelligent gait analytics · Sensor intelligence · Explainable AI</p></div>''',unsafe_allow_html=True)
st.caption('Yellow edition • Force-sensor biomechanics • Explainable machine learning • Longitudinal research monitoring')
st.info('Research use only · Not a medical diagnosis · Verify labels against original dataset metadata · Model scores are not clinical risk estimates.')

def fig_png(fig):
    b=io.BytesIO();fig.savefig(b,format='png',dpi=145,bbox_inches='tight');plt.close(fig);return b.getvalue()

def heatmap(df,kind='Mean'):
    fig,axs=plt.subplots(1,2,figsize=(8,5))
    v=np.array([[float(df[f'{side}_sensor_{i}'].mean() if kind=='Mean' else df[f'{side}_sensor_{i}'].max()) for i in range(1,9)] for side in ('left','right')]); vmax=max(np.nanmax(v),1)
    # Schematic positions, not measured sensor coordinates. No COP inferred.
    xy=[(-.38,.7),(.38,.7),(-.42,.3),(.42,.3),(-.4,-.1),(.4,-.1),(-.32,-.65),(.32,-.65)]
    for j,side in enumerate(('Left','Right')):
        ax=axs[j];ax.add_patch(Ellipse((0,0),1.35,2.25,facecolor='#eef3f9',edgecolor='#52617a',lw=2))
        for i,(x,y) in enumerate(xy):
            ax.scatter(x,y,s=850,c=[v[j,i]],vmin=0,vmax=vmax,cmap='inferno',edgecolors='white',linewidths=1.5)
            ax.text(x,y,f'{i+1}',color='white',ha='center',va='center',fontsize=10,weight='bold')
        ax.set_xlim(-.9,.9);ax.set_ylim(-1.3,1.3);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{side} foot')
    sm=plt.cm.ScalarMappable(cmap='inferno',norm=plt.Normalize(0,vmax));fig.colorbar(sm,ax=axs.tolist(),label=f'{kind} force (N)',shrink=.75)
    fig.suptitle('Schematic sensor layout — positions illustrative, NOT anatomical calibration',fontsize=10)
    return fig

def plot_metrics(metrics):
    keys=['left_peak_N','right_peak_N','left_mean_force_N','right_mean_force_N'];fig,ax=plt.subplots(figsize=(7,3));ax.bar(['L peak','R peak','L mean','R mean'],[metrics[k] for k in keys]);ax.set_ylabel('Force (N)');ax.set_title('Left/right vertical ground reaction force');return fig

def pdf_report(subject,metadata,metrics,quality,figures,prediction=None):
    buf=io.BytesIO();doc=SimpleDocTemplate(buf,pagesize=(595,842),leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=40)
    sty=getSampleStyleSheet();story=[Paragraph('STEP FIX 4.0 — Research Gait Analysis',sty['Title']),Spacer(1,10)]
    for s in [f"Subject: {subject}",f"Recording: {quality['filename']}",f"Format: {quality['format']} | Samples: {quality['samples']} | Sampling: {quality['sampling_hz']:.2f} Hz",f"Provenance: {metadata}","Research/educational output only. NOT a Parkinson's disease diagnosis."]:
        story.append(Paragraph(str(s).replace('&','&amp;').replace('<','&lt;'),sty['Normal']));story.append(Spacer(1,6))
    rows=[['Metric','Value']]+[[k,f'{v:.3f}' if np.isfinite(v) else 'Unavailable'] for k,v in metrics.items()]
    tab=Table(rows,colWidths=[295,130],repeatRows=1);tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8fa')),('GRID',(0,0),(-1,-1),.3,colors.grey),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(tab)
    if prediction:
        story.append(Spacer(1,12));story.append(Paragraph('Research model output',sty['Heading2']))
        story.append(Paragraph(prediction,sty['Normal']))
    for name,png in figures:
        story.append(Spacer(1,14));story.append(Paragraph(name,sty['Heading3']));story.append(Image(io.BytesIO(png),width=5.9*inch,height=2.6*inch,kind='proportional'))
    story.append(Spacer(1,12));story.append(Paragraph('Limitations: Contact-based gait events are estimates; schematic sensor positions are illustrative. Internal held-out model performance does not constitute clinical validation.',sty['Normal']))
    doc.build(story);return buf.getvalue()



def calibrated_cop(df, positions):
    """Compute force-weighted COP from user-supplied, calibrated sensor coordinates in mm."""
    outputs = {}
    for side in ('left', 'right'):
        cols = [f'{side}_sensor_{i}' for i in range(1, 9)]
        forces = df[cols].to_numpy(dtype=float)
        xy = positions.loc[cols, ['x_mm', 'y_mm']].to_numpy(dtype=float)
        valid = np.isfinite(forces).all(axis=1) & (forces >= 0).all(axis=1)
        denom = np.sum(forces, axis=1)
        valid &= denom > 1e-8
        x = np.full(len(df), np.nan)
        y = np.full(len(df), np.nan)
        x[valid] = (forces[valid] @ xy[:, 0]) / denom[valid]
        y[valid] = (forces[valid] @ xy[:, 1]) / denom[valid]
        outputs[side] = pd.DataFrame({'time_s': df.time_s, 'x_mm': x, 'y_mm': y})
    return outputs


def draw_cop(tracks):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (side, track) in zip(axs, tracks.items()):
        valid = track.dropna()
        if len(valid):
            sc = ax.scatter(valid.x_mm, valid.y_mm, c=valid.time_s, s=5, cmap='viridis')
            fig.colorbar(sc, ax=ax, label='Time (s)')
        ax.set_title(side.title() + ' foot COP trajectory')
        ax.set_xlabel('Calibrated x (mm)'); ax.set_ylabel('Calibrated y (mm)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def validate_positions(raw):
    pos = pd.read_csv(io.BytesIO(raw))
    if not {'sensor','x_mm','y_mm'}.issubset(pos.columns):
        raise ValueError('Calibration file needs sensor,x_mm,y_mm columns')
    if len(pos) != 16 or pos.sensor.duplicated().any() or set(pos.sensor) != set(SENSORS):
        raise ValueError('Provide exactly one calibrated coordinate for each of the 16 sensors')
    pos = pos.set_index('sensor').loc[SENSORS]
    pos[['x_mm','y_mm']] = pos[['x_mm','y_mm']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(pos[['x_mm','y_mm']].to_numpy()).all():
        raise ValueError('All coordinates must be finite')
    return pos

st.subheader('Upload your gait recording')
st.write('Upload your CSV/TXT here. The analysis tabs appear below after upload; no sidebar is needed to start.')
upload=st.file_uploader('Choose gait recording (19, 7, or 3 columns)',type=['txt','csv'],key='primary_recording')
with st.sidebar:
    st.header('Analysis settings')
    subject=st.text_input('Subject ID (optional)')
    label=st.selectbox('Verified subject group (descriptive only)',['Unknown / not verified','Parkinson’s disease (metadata-confirmed)','Healthy control (metadata-confirmed)'])
    condition=st.selectbox('Walking condition',['Usual walking','Dual task (serial 7 subtraction)','Unknown'])
    threshold=st.number_input('Foot-contact force threshold (N)',1.,500.,20.,5.)
    st.caption('5-sample median filter; minimum contact 0.1 s. Labels are not inferred from this dropdown.')
    calibration_upload=st.file_uploader('Optional calibrated sensor coordinates (CSV)',type=['csv'],help='Required for scientifically meaningful center-of-pressure tracking. CSV columns: sensor,x_mm,y_mm')
    model_upload=st.file_uploader('Optional trained model (.joblib)',type=['joblib'],help='Only load model files you created and trust. Joblib can execute code during loading.')

if upload is None:
    st.info('Waiting for a gait recording. Upload a CSV/TXT using the visible uploader above.')
    st.markdown('**Available after upload:** Gait dashboard · Foot-pressure heatmap · Center of pressure · Random Forest & evaluation · Sensor failure simulation · Longitudinal monitoring · Provenance & validation · PDF report')
    st.caption('Use the included synthetic demo CSV for interface testing only. Real model training requires verified participant metadata.')
else:
    try: df,quality=read_gait(upload.getvalue(),upload.name)
    except Exception as e: st.error(f'Cannot read recording: {e}');st.stop()
    st.success(f"Loaded {quality['samples']:,} samples | {quality['format']}")
    for w in quality['warnings']: st.warning(w)
    with st.expander('Select analysis time window'):
        a,b=float(df.time_s.iloc[0]),float(df.time_s.iloc[-1]);start,end=st.slider('Time (s)',min_value=a,max_value=b,value=(a,b))
    window=df[(df.time_s>=start)&(df.time_s<=end)].copy()
    if len(window)<10: st.error('Window too short');st.stop()
    metrics=visible_metrics(analyze(window,threshold))
    cols=st.columns(4)
    for c,(key,title,unit) in zip(cols,[('left_peak_N','Left peak',' N'),('right_peak_N','Right peak',' N'),('peak_asymmetry_pct','Peak asymmetry',' %'),('cadence_spm','Estimated cadence',' steps/min')]):
        v=metrics[key];c.metric(title,f'{v:.1f}{unit}' if np.isfinite(v) else 'Unavailable')
    summary,signal,events_tab,heat,cop,ml,robust,monitor,provenance,report=st.tabs(['Gait dashboard','Signal explorer','Gait event timeline','Foot-pressure heatmap','Center of pressure','Random Forest & evaluation','Sensor failure simulation','Longitudinal monitoring','Provenance & validation','Export report'])
    figures=[('Force summary',fig_png(plot_metrics(metrics)))]
    with summary:
        st.subheader('Calculated gait measures')
        st.dataframe(pd.DataFrame([{'Feature':k,'Value':round(v,3) if np.isfinite(v) else None} for k,v in metrics.items()]),hide_index=True,use_container_width=True)
        st.image(figures[0][1]);st.caption('Cadence and temporal features are force-threshold estimates, not annotated ground truth.')
        st.download_button('Download analysis CSV',pd.DataFrame([{**metrics,'subject':subject or upload.name,'label_claim':label,'condition':condition}]).to_csv(index=False),'stepfix_analysis.csv','text/csv')

    with signal:
        st.subheader('Interactive vertical ground reaction force explorer')
        st.caption('Explore measured left/right force traces; no values are synthesized.')
        stride=max(1,len(window)//4500)
        trace=window.iloc[::stride].set_index('time_s')
        st.line_chart(trace[['left_total','right_total']],height=350)
        if sensor_summary(window) is not None:
            side=st.radio('Foot to inspect',['left','right'],horizontal=True,key='trace_foot')
            sensor_names=[f'{side}_sensor_{i}' for i in range(1,9)]
            st.line_chart(trace[sensor_names],height=300)
        st.download_button('Export plotted force traces',trace.reset_index().to_csv(index=False),'stepfix_force_traces.csv','text/csv')
    with events_tab:
        st.subheader('Contact-event timeline and recording quality')
        time_array=window.time_s.to_numpy(float)
        event_fig,ax=plt.subplots(figsize=(12,3.6))
        for side,y,color in [('left',1,'#f4bb17'),('right',0,'#58c5e8')]:
            raw=window[f'{side}_total'].to_numpy(float)
            smoothed=pd.Series(raw).rolling(5,center=True,min_periods=1).median().to_numpy()
            active=smoothed>=threshold
            edges=np.diff(active.astype(int),prepend=0,append=0)
            starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            for a,b in zip(starts,ends):
                if b-a>=10:
                    ax.broken_barh([(time_array[a],time_array[min(b,len(time_array)-1)]-time_array[a])],(y-.27,.54),facecolors=color)
        ax.set_yticks([0,1],['Right','Left']);ax.set_xlabel('Time (seconds)');ax.set_title('Detected foot contacts (threshold-based estimates)')
        ax.set_facecolor('#181a20');event_fig.patch.set_facecolor('#181a20');ax.tick_params(colors='white');ax.xaxis.label.set_color('white');ax.title.set_color('white')
        st.image(fig_png(event_fig))
        dt=np.diff(time_array);jitter=100*np.std(dt)/np.mean(dt) if len(dt)>1 and np.mean(dt)>0 else np.nan
        qc1,qc2,qc3,qc4=st.columns(4)
        qc1.metric('Observed sampling rate',f"{quality['sampling_hz']:.1f} Hz")
        qc2.metric('Timestamp jitter (CV)',f'{jitter:.2f}%' if np.isfinite(jitter) else 'N/A')
        qc3.metric('Missing sensor samples',str(sum(window[c].isna().sum() for c in SENSORS if c in window)))
        qc4.metric('Contact threshold',f'{threshold:.0f} N')
        if quality['warnings']:
            for warning in quality['warnings']:st.warning(warning)
        else:st.success('No parser-level sampling or missing-sensor warnings detected. This does not establish clinical signal quality.')
        st.caption('Event timeline depends on threshold, smoothing and minimum-duration assumptions; no annotated ground-truth events are implied.')
    with heat:
        ss=sensor_summary(window)
        if ss is None: st.warning('Heatmap requires all 16 individual sensor channels (native 19-column recording).')
        else:
            kind=st.radio('Sensor statistic',['Mean','Peak','Instantaneous'],horizontal=True)
            if kind == 'Instantaneous':
                sample_idx=st.slider('Playback position (sample)',0,len(window)-1,0,step=max(1,len(window)//500))
                snapshot=window.iloc[[sample_idx]]
                st.caption(f"Recording time: {float(snapshot.time_s.iloc[0]):.2f} seconds")
                heat_png=fig_png(heatmap(snapshot,'Mean'))
            else:
                heat_png=fig_png(heatmap(window,kind))
            st.image(heat_png)
            st.caption('Sensor locations are illustrative; native PhysioNet channel numbering does not by itself establish exact physical positions. Do not interpret this as a calibrated anatomical pressure map.')
            st.dataframe(ss,hide_index=True,use_container_width=True)
            figures.append(('Sensor-force schematic',heat_png))
    with cop:
        st.subheader('Center-of-pressure (COP) tracking')
        st.caption('COP is calculated as the force-weighted mean of verified physical sensor coordinates. It is NOT calculated from the illustrative heatmap layout.')
        if sensor_summary(window) is None:
            st.warning('Requires a native 19-column recording with all 16 sensors.')
        elif calibration_upload is None:
            st.info('Upload a sensor-coordinate calibration CSV in the sidebar to enable COP. Columns: sensor,x_mm,y_mm. Coordinates must be measured or documented for the actual recording system.')
            st.download_button('Download coordinate template',pd.DataFrame({'sensor':SENSORS,'x_mm':['']*16,'y_mm':['']*16}).to_csv(index=False),'sensor_coordinates_TEMPLATE.csv','text/csv')
        else:
            try:
                positions=validate_positions(calibration_upload.getvalue())
                tracks=calibrated_cop(window,positions)
                cop_png=fig_png(draw_cop(tracks));st.image(cop_png)
                figures.append(('Calibrated center-of-pressure trajectories',cop_png))
                st.download_button('Download COP trajectories',pd.concat([track.assign(foot=side) for side,track in tracks.items()]).to_csv(index=False),'stepfix_cop.csv','text/csv')
                st.caption('Force-weighted COP assumes the channels and coordinates correspond to the same calibrated hardware and units.')
            except Exception as e: st.error(f'Invalid COP calibration: {e}')

    model=None;model_info=None
    if model_upload is not None:
        try:
            import joblib
            model_info=joblib.load(io.BytesIO(model_upload.getvalue()))
            if model_info.get('features')!=FEATURES: raise ValueError('Model feature schema mismatch')
            model=model_info['pipeline']
        except Exception as e: st.error(f'Model cannot be loaded: {e}')
    model_prediction=None
    with ml:
        st.subheader('Random Forest — research classifier')
        if model is None: st.warning('No trained model loaded. Train with verified metadata using train_model.py, then upload the model file in the sidebar.')
        else:
            trained_threshold=model_info['threshold_N']
            if abs(trained_threshold-threshold)>1e-6: st.warning(f'Model trained with threshold {trained_threshold} N; prediction features use this trained threshold rather than sidebar threshold.')
            model_metrics=analyze(window,trained_threshold)
            X=pd.DataFrame([{k:model_metrics[k] for k in FEATURES}]);p=float(model.predict_proba(X)[0,1]);classification='PD-pattern' if p>=.5 else 'Control-pattern'
            model_prediction=f'{classification}; model PD-class probability {p:.3f}. Not a clinical diagnosis.'
            st.metric('Model PD-class probability',f'{p:.1%}');st.write(f'**Predicted pattern:** {classification}')
            st.caption('Uncalibrated Random Forest probability, not an individual probability of disease.')
            evaluation=model_info['results'];st.subheader('Held-out subject evaluation')
            ec=st.columns(5)
            for col,k in zip(ec,['accuracy','precision','recall','f1','roc_auc']):col.metric(k.upper(),f"{evaluation[k]:.3f}")
            st.write(f"Train subjects: {evaluation['train_subjects']} | Test subjects: {evaluation['test_subjects']}")
            cm=np.asarray(evaluation['confusion_matrix']);st.dataframe(pd.DataFrame(cm,index=['True control','True PD'],columns=['Pred control','Pred PD']))
            imp=pd.Series(model.named_steps['classifier'].feature_importances_,index=FEATURES).reindex(DISPLAY_FEATURES).sort_values(ascending=False).head(12)
            st.subheader('Global feature importance (impurity-based)');st.bar_chart(imp)
            st.caption('These are global training-model feature importances, NOT an explanation of this individual prediction. Correlated features may bias impurity importance.')
            st.subheader('Local perturbation explanation (exploratory)')
            baseline=model.named_steps['imputer'].statistics_
            changes=[]
            for j,k in enumerate(FEATURES):
                if k not in DISPLAY_FEATURES: continue
                temp=X.copy();temp.loc[0,k]=baseline[j]
                new=float(model.predict_proba(temp)[0,1]);changes.append((k,p-new))
            changes=sorted(changes,key=lambda z:abs(z[1]),reverse=True)[:8]
            st.dataframe(pd.DataFrame(changes,columns=['Feature replaced by training median','Change in PD-class probability']),hide_index=True)
            st.caption('One-feature-at-a-time median replacement is a sensitivity analysis, not SHAP or a causal explanation. It can create implausible combinations of correlated gait features.')
    with robust:
        if sensor_summary(window) is None: st.warning('Sensor failure simulation requires a native 19-column file.')
        else:
            chosen=st.selectbox('Disable a sensor',SENSORS)
            altered=sensor_failure(window,chosen);after=analyze(altered,threshold)
            comp=pd.DataFrame([{'Feature':k,'Original':metrics[k],'Simulated failure':after[k],'Difference':after[k]-metrics[k]} for k in DISPLAY_FEATURES]);st.dataframe(comp,hide_index=True,use_container_width=True)
            st.caption('Counterfactual: the chosen sensor is set to zero and the affected foot total is reconstructed from the remaining sensor channels. Native total channels may not exactly equal sensor sums; compare with caution.')
            if model is not None:
                base=analyze(window,model_info['threshold_N']);changed=analyze(altered,model_info['threshold_N']);basep=float(model.predict_proba(pd.DataFrame([{k:base[k] for k in FEATURES}]))[0,1]);changedp=float(model.predict_proba(pd.DataFrame([{k:changed[k] for k in FEATURES}]))[0,1]);st.write(f'Model PD-class score: original **{basep:.3f}**, simulated **{changedp:.3f}**, difference **{changedp-basep:+.3f}**')
    with monitor:
        st.subheader('Longitudinal gait monitoring')
        st.write('Compare recordings of the same subject across sessions. This is a research trend analysis, not evidence of clinical progression or recovery.')
        extra_files=st.file_uploader('Upload additional session recordings (CSV/TXT)',type=['csv','txt'],accept_multiple_files=True,key='followups')
        st.caption('Enter actual session dates from your recording log. Files are not automatically matched to a patient identity.')
        session_rows=[{'session':'Baseline (current file)','date':None,**metrics}]
        for i,f in enumerate(extra_files or []):
            c1,c2=st.columns([2,1])
            with c1: st.write(f'**{f.name}**')
            with c2: date=st.date_input('Recording date',key=f'session_date_{i}')
            try:
                session_df,session_quality=read_gait(f.getvalue(),f.name)
                row=visible_metrics(analyze(session_df,threshold))
                session_rows.append({'session':f.name,'date':str(date),**row})
            except Exception as e: st.error(f'{f.name}: {e}')
        baseline_date=st.date_input('Baseline recording date',key='baseline_date')
        session_rows[0]['date']=str(baseline_date)
        if len(session_rows)>1:
            trends=pd.DataFrame(session_rows)
            if len(trends.date.unique())!=len(trends): st.warning('Some dates are identical. Confirm the actual session dates before interpreting time trends.')
            st.dataframe(trends,hide_index=True,use_container_width=True)
            chosen_metrics=st.multiselect('Compare features',DISPLAY_FEATURES,default=['cadence_spm','peak_asymmetry_pct'])
            for feature in chosen_metrics:
                st.line_chart(trends.set_index('session')[[feature]])
            st.download_button('Download longitudinal comparison CSV',trends.to_csv(index=False),'stepfix_longitudinal.csv','text/csv')
            st.caption('Session recordings should share sensor calibration, protocol, footwear, walking condition, and threshold for meaningful comparisons. No longitudinal reference thresholds are assumed.')
        else:
            st.info('Upload at least one additional recording to compare sessions.')

    with provenance:
        st.subheader('Data provenance & validation')
        st.write(f'**File:** {upload.name} | **Subject:** {subject or "Not provided"} | **Claimed label:** {label} | **Walking condition:** {condition}')
        st.json(quality)
        st.warning('Uploading a file or selecting a group does NOT independently verify its origin or subject label. Check the official PhysioNet metadata before training.')
        st.markdown('[PhysioNet Gait in Parkinson’s Disease database](https://physionet.org/content/gaitpdb/1.0.0/)')
        st.write('Dataset files must be mapped to official subject identifiers and verified group labels in the training manifest. Do not train from dropdown selections.')
    with report:
        st.subheader('Export research report')
        st.download_button('Download PDF report',pdf_report(subject or upload.name,f'{label}; {condition}',metrics,quality,figures,model_prediction),'stepfix_research_report.pdf','application/pdf')
        st.download_button('Download selected recording CSV',window.to_csv(index=False).encode(),'stepfix_recording.csv','text/csv')
