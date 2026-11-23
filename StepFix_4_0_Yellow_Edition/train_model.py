"""Train from a user-supplied manifest with VERIFIED labels. Never infer labels from filenames.
manifest.csv columns: filename,subject_id,label (PD or Control), optional condition.
Recording paths are relative to the manifest directory. Subject-wise holdout is mandatory.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from core import read_gait,analyze,FEATURES

def train(manifest_path,output_dir,threshold=20.0):
    manifest_path=Path(manifest_path).resolve(); out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(manifest_path)
    required={'filename','subject_id','label'}
    if not required.issubset(manifest): raise ValueError(f'Manifest requires {required}')
    if manifest[list(required)].isna().any().any(): raise ValueError('Manifest has missing required values')
    if not set(manifest.label.unique()).issubset({'PD','Control'}): raise ValueError('Labels must be PD or Control from verified metadata')
    if manifest.groupby('subject_id').label.nunique().max()>1: raise ValueError('Conflicting labels for same subject')
    records=[];errors=[]
    for row in manifest.itertuples(index=False):
        try:
            p=(manifest_path.parent/str(row.filename)).resolve()
            if not p.is_relative_to(manifest_path.parent): raise ValueError('File outside manifest directory')
            df,quality=read_gait(p.read_bytes(),p.name)
            f=analyze(df,threshold)
            records.append({'filename':row.filename,'subject_id':str(row.subject_id),'label':row.label,**{k:f[k] for k in FEATURES}})
        except Exception as exc: errors.append({'filename':row.filename,'error':str(exc)})
    pd.DataFrame(errors,columns=['filename','error']).to_csv(out/'rejected_files.csv',index=False)
    data=pd.DataFrame(records)
    if data.empty: raise ValueError('No valid recordings')
    counts=data.groupby('label').subject_id.nunique()
    if len(counts)!=2 or counts.min()<3: raise ValueError('Need at least 3 independent subjects per class')
    X=data[FEATURES];y=(data.label=='PD').astype(int);groups=data.subject_id
    split=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    tr,te=next(split.split(X,y,groups))
    if len(set(y.iloc[tr]))<2 or len(set(y.iloc[te]))<2: raise ValueError('Subject-wise split lacks both classes; add subjects and retry')
    model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('classifier',RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))])
    model.fit(X.iloc[tr],y.iloc[tr]); pred=model.predict(X.iloc[te]); prob=model.predict_proba(X.iloc[te])[:,1]
    results={'accuracy':float(accuracy_score(y.iloc[te],pred)),'precision':float(precision_score(y.iloc[te],pred,zero_division=0)),'recall':float(recall_score(y.iloc[te],pred,zero_division=0)),'f1':float(f1_score(y.iloc[te],pred,zero_division=0)),'roc_auc':float(roc_auc_score(y.iloc[te],prob)),'confusion_matrix':confusion_matrix(y.iloc[te],pred,labels=[0,1]).tolist(),'train_subjects':int(groups.iloc[tr].nunique()),'test_subjects':int(groups.iloc[te].nunique()),'train_recordings':len(tr),'test_recordings':len(te),'threshold_N':threshold,'label_source':'User-provided verified metadata manifest','classes':['Control','PD'],'features':FEATURES,'warning':'Research classifier; holdout scores are not external clinical validation'}
    joblib.dump({'pipeline':model,'features':FEATURES,'threshold_N':threshold,'results':results},out/'stepfix_model.joblib')
    (out/'evaluation.json').write_text(json.dumps(results,indent=2));data.to_csv(out/'extracted_features.csv',index=False)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',default='model_output');ap.add_argument('--threshold',type=float,default=20.0)
    args=ap.parse_args();train(args.manifest,args.output,args.threshold)
