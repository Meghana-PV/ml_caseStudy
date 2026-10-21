import sys,json,os,warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import traceback

def readcsv(root, rel): return pd.read_csv(os.path.join(root,rel), low_memory=False)
def config_df(root,dataset,cfg):
    df=readcsv(root,cfg["path"])
    if cfg["task"]=="regression" and cfg.get("weather"):
        w=readcsv(root,cfg["weather"])
        w["DATE_TIME"]=pd.to_datetime(w["DATE_TIME"],errors="coerce",dayfirst=True)
        g=df.copy();g["DATE_TIME"]=pd.to_datetime(g["DATE_TIME"],errors="coerce",dayfirst=True)
        weather_cols=[c for c in ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"] if c in w.columns]
        w=w.groupby("DATE_TIME",as_index=False)[weather_cols].mean()
        df=g.merge(w,on="DATE_TIME",how="left")
    return df
def clean_num(s):
    return pd.to_numeric(s,errors="coerce")
def inspect(root,dataset,cfg):
    df=config_df(root,dataset,cfg)
    columns=[]
    for c in df.columns:
        s=df[c]; n=pd.to_numeric(s,errors="coerce")
        numeric=bool(n.notna().sum()>0 and n.notna().sum()>=max(1,int(s.notna().sum()*.7)))
        columns.append({"name":str(c),"numeric":numeric,"missing":int(s.isna().sum()),"unique":int(s.nunique(dropna=True)),"min":float(n.min()) if numeric and n.notna().any() else None,"max":float(n.max()) if numeric and n.notna().any() else None,"mean":float(n.mean()) if numeric and n.notna().any() else None})
    preview=df.head(7).replace({np.nan:None}).to_dict("records")
    return {"totalRows":int(len(df)),"columns":columns,"preview":preview}
def json_clean(obj):
    if isinstance(obj,dict): return {str(k):json_clean(v) for k,v in obj.items()}
    if isinstance(obj,list): return [json_clean(x) for x in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if pd.isna(obj): return None
    return obj
def clustering(root,dataset,cfg,k,algorithm):
    df=readcsv(root,cfg["path"])
    df=df.drop(columns=[c for c in ["Unnamed: 0","id","satisfaction"] if c in df.columns],errors="ignore")
    df=df.drop_duplicates().copy()
    # Convert ?-style missing values and encode categorical variables as in notebook
    df=df.replace(["?","N/A","NA",""],np.nan)
    numeric=[c for c in df.columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>=max(1,int(df[c].notna().sum()*.7))]
    categorical=[c for c in df.columns if c not in numeric]
    # cap records for responsive web experiments, deterministic sample
    cap=3000 if algorithm=="hierarchical" else 10000
    if len(df)>cap: df=df.sample(n=cap,random_state=42)
    parts=[]
    if numeric:
        num=df[numeric].apply(pd.to_numeric,errors="coerce")
        num=num.fillna(num.median()).fillna(0)
        scaled=StandardScaler().fit_transform(num)
        parts.append(scaled)
    if categorical:
        cat=df[categorical].fillna("Unknown").astype(str)
        encoded=pd.get_dummies(cat,drop_first=True,dtype=float)
        if encoded.shape[1]:
            parts.append(StandardScaler().fit_transform(encoded))
    if not parts: raise ValueError("No usable features in this dataset.")
    X=np.concatenate(parts,axis=1)
    kk=max(2,min(10,int(k or 4),len(X)-1))
    if algorithm=="hierarchical":
        # Agglomerative clustering is quadratic; cap is deliberately lower.
        model=AgglomerativeClustering(n_clusters=kk,linkage="ward")
        labels=model.fit_predict(X)
        algorithm_name="Hierarchical (Agglomerative · Ward)"
        inertia=float(sum(np.sum((X[labels==i]-X[labels==i].mean(axis=0))**2) for i in range(kk) if np.any(labels==i)))
        dendro_n=min(60,len(X))
        Z=linkage(X[:dendro_n],method="ward")
        # Compact dendrogram data for a client-rendered tree-like plot
        dendro={"sampleRows":dendro_n,"merges":[{"left":int(a),"right":int(b),"distance":float(d),"count":int(c)} for a,b,d,c in Z[-min(35,len(Z)):]]}
    else:
        model=KMeans(n_clusters=kk,random_state=42,n_init=10)
        labels=model.fit_predict(X);inertia=float(model.inertia_);algorithm_name="K-Means"
        dendro=None
    sizes=np.bincount(labels,minlength=kk).tolist()
    sil=float(silhouette_score(X,labels,sample_size=min(1500,len(X)),random_state=42)) if len(set(labels))>1 and len(X)>2 else 0
    db=float(davies_bouldin_score(X,labels)) if len(set(labels))>1 else 0
    ch=float(calinski_harabasz_score(X,labels)) if len(set(labels))>1 else 0
    coords=PCA(n_components=2,random_state=42).fit_transform(X)
    points=[{"x":float(coords[i,0]),"y":float(coords[i,1]),"cluster":int(labels[i])} for i in range(min(600,len(X)))]
    return {"algorithm":algorithm_name,"algorithmKey":algorithm,"rows":int(len(X)),"features":numeric+categorical,"k":kk,"inertia":inertia,"metrics":{"silhouette":sil,"daviesBouldin":db,"calinskiHarabasz":ch},"clusterSizes":sizes,"scatter":points,"dendrogram":dendro,"note":("Hierarchical clustering uses up to 3,000 rows because agglomerative clustering is computationally expensive." if algorithm=="hierarchical" else "K-Means uses up to 10,000 rows. Categorical columns are one-hot encoded and features are standardized. The plot shows PCA coordinates.")}
def prep_reg(root,dataset,cfg):
    df=config_df(root,dataset,cfg).copy()
    df=df.drop_duplicates()
    df["DATE_TIME"]=pd.to_datetime(df["DATE_TIME"],errors="coerce",dayfirst=True)
    df["hour"]=df["DATE_TIME"].dt.hour;df["day"]=df["DATE_TIME"].dt.day;df["month"]=df["DATE_TIME"].dt.month;df["day_of_week"]=df["DATE_TIME"].dt.dayofweek
    features=["DC_POWER","DAILY_YIELD","TOTAL_YIELD","hour","day","month","day_of_week"]
    for c in ["AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]:
        if c in df.columns: features.append(c)
    for c in features+["AC_POWER"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["AC_POWER"])
    for c in features: df[c]=df[c].fillna(df[c].median()).fillna(0)
    return df,features
def regression(root,dataset,cfg,values=None):
    import joblib
    df,features=prep_reg(root,dataset,cfg);X=df[features];y=df["AC_POWER"]
    if len(df)<20: raise ValueError("Not enough records for regression.")
    cache_dir=os.path.join(root,".model_cache");os.makedirs(cache_dir,exist_ok=True)
    cache_path=os.path.join(cache_dir,dataset+"_regression.joblib")
    if os.path.exists(cache_path):
        bundle=joblib.load(cache_path);model=bundle["model"];metrics=bundle["metrics"];importances=bundle["importances"];chart=bundle["chart"]
    else:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
        model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1,min_samples_split=2)
        model.fit(X_train,y_train);pred=model.predict(X_test)
        metrics={"r2":float(r2_score(y_test,pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,pred))),"mae":float(mean_absolute_error(y_test,pred))}
        importances=sorted([{"feature":f,"importance":float(v)} for f,v in zip(features,model.feature_importances_)],key=lambda x:x["importance"],reverse=True)
        chart=[{"actual":float(a),"predicted":float(p)} for a,p in zip(y_test.iloc[:120],pred[:120])]
        model.fit(X,y)
        joblib.dump({"model":model,"metrics":metrics,"importances":importances,"chart":chart},cache_path)
    vals=values or {}
    sample={f:float(vals.get(f,df[f].median())) for f in features}
    prediction=float(model.predict(pd.DataFrame([sample],columns=features))[0])
    return {"algorithm":"Random Forest Regressor","rows":int(len(df)),"features":features,"target":"AC_POWER","metrics":metrics,"actualPredicted":chart,"importances":importances,"prediction":prediction,"inputValues":sample,"inputSchema":[{"name":f,"min":float(df[f].min()),"max":float(df[f].max()),"default":float(df[f].median()),"step":0.1 if df[f].dtype.kind in "fc" else 1} for f in features],"note":"Model selection follows your notebook's comparison: Random Forest had the lowest RMSE (0.8634) and MAE (0.1381) with R² ≈ 0.999995 on Plant 1's original split. Metrics here are recalculated for the selected plant using generation + timestamp-matched average weather features. Plant 2 is trained and evaluated separately."}
def classification(root,dataset,cfg,values=None):
    import joblib
    df=readcsv(root,cfg["path"]).copy()
    needed=["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machine failure"]
    for c in needed:
        if c not in df.columns: raise ValueError("Missing classification column: "+c)
    df["Temperature difference [K]"]=pd.to_numeric(df["Process temperature [K]"],errors="coerce")-pd.to_numeric(df["Air temperature [K]"],errors="coerce")
    numeric=["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Temperature difference [K]"]
    for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Machine failure"])
    for c in ["Rotational speed [rpm]","Torque [Nm]"]:
        q1,q3=df[c].quantile(.25),df[c].quantile(.75);iqr=q3-q1;df[c]=df[c].clip(q1-1.5*iqr,q3+1.5*iqr)
    X=df[["Type"]+numeric].copy()
    X["Type"]=X["Type"].fillna(X["Type"].mode().iloc[0] if not X["Type"].mode().empty else "M")
    for c in numeric: X[c]=X[c].fillna(X[c].median())
    X=pd.get_dummies(X,columns=["Type"],drop_first=True,dtype=int)
    y=pd.to_numeric(df["Machine failure"],errors="coerce").fillna(0).astype(int)
    cache_dir=os.path.join(root,".model_cache");os.makedirs(cache_dir,exist_ok=True)
    cache_path=os.path.join(cache_dir,"machine_classifier_gb.joblib")
    if os.path.exists(cache_path):
        bundle=joblib.load(cache_path);model=bundle["model"];metrics=bundle["metrics"];cm=bundle["confusion"];importance=bundle["importances"]
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        model=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=3,random_state=42)
        model.fit(Xtr,ytr);pred=model.predict(Xte);probs=model.predict_proba(Xte)[:,1]
        cm0=confusion_matrix(yte,pred,labels=[0,1])
        cm={"tn":int(cm0[0,0]),"fp":int(cm0[0,1]),"fn":int(cm0[1,0]),"tp":int(cm0[1,1])}
        metrics={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,average="weighted",zero_division=0)),"recall":float(recall_score(yte,pred,average="weighted",zero_division=0)),"f1":float(f1_score(yte,pred,average="weighted",zero_division=0)),"rocAuc":float(roc_auc_score(yte,probs))}
        importance=sorted([{"feature":f,"importance":float(v)} for f,v in zip(X.columns,model.feature_importances_)],key=lambda x:x["importance"],reverse=True)
        model.fit(X,y)
        joblib.dump({"model":model,"metrics":metrics,"confusion":cm,"importances":importance},cache_path)
    vals=values or {}
    raw={"Type":str(vals.get("Type","M"))}
    defaults={"Air temperature [K]":df["Air temperature [K]"].median(),"Process temperature [K]":df["Process temperature [K]"].median(),"Rotational speed [rpm]":df["Rotational speed [rpm]"].median(),"Torque [Nm]":df["Torque [Nm]"].median(),"Tool wear [min]":df["Tool wear [min]"].median()}
    for c in defaults: raw[c]=float(vals.get(c,defaults[c]))
    raw["Temperature difference [K]"]=raw["Process temperature [K]"]-raw["Air temperature [K]"]
    for c in ["Rotational speed [rpm]","Torque [Nm]"]:
        q1,q3=df[c].quantile(.25),df[c].quantile(.75);iqr=q3-q1;raw[c]=float(np.clip(raw[c],q1-1.5*iqr,q3+1.5*iqr))
    row=pd.DataFrame([raw]);row=pd.get_dummies(row,columns=["Type"],drop_first=True,dtype=int);row=row.reindex(columns=X.columns,fill_value=0)
    p=float(model.predict_proba(row)[0][1]);predicted=int(p>=.5)
    return {"algorithm":"Gradient Boosting Classifier (best weighted F1 in notebook comparison)","rows":int(len(df)),"features":list(X.columns),"target":"Machine failure","metrics":metrics,"confusion":cm,"importances":importance,"prediction":{"label":"Failure likely" if predicted else "No failure predicted","class":predicted,"probability":p},"inputValues":raw,"inputSchema":[{"name":"Type","kind":"select","options":["L","M","H"],"default":"M"},{"name":"Air temperature [K]","min":float(df["Air temperature [K]"].min()),"max":float(df["Air temperature [K]"].max()),"default":float(defaults["Air temperature [K]"]),"step":0.1},{"name":"Process temperature [K]","min":float(df["Process temperature [K]"].min()),"max":float(df["Process temperature [K]"].max()),"default":float(defaults["Process temperature [K]"]),"step":0.1},{"name":"Rotational speed [rpm]","min":float(df["Rotational speed [rpm]"].min()),"max":float(df["Rotational speed [rpm]"].max()),"default":float(defaults["Rotational speed [rpm]"]),"step":1},{"name":"Torque [Nm]","min":float(df["Torque [Nm]"].min()),"max":float(df["Torque [Nm]"].max()),"default":float(defaults["Torque [Nm]"]),"step":0.1},{"name":"Tool wear [min]","min":float(df["Tool wear [min]"].min()),"max":float(df["Tool wear [min]"].max()),"default":float(defaults["Tool wear [min]"]),"step":1}],"note":"Uses the tuned Gradient Boosting parameters from your notebook comparison (200 estimators, learning_rate=0.1, max_depth=3), which had the highest tuned weighted F1 in the final comparison table. Metrics use a stratified 20% holdout; the model is then refit on all rows for interactive predictions."}
def main():
    req=json.loads(sys.stdin.read());root=req["root"];dataset=req["dataset"];cfg=req["config"]
    if req["action"]=="inspect": result=inspect(root,dataset,cfg)
    elif req["task"]=="clustering": result=clustering(root,dataset,cfg,req.get("k",4),req.get("algorithm","kmeans"))
    elif req["task"]=="regression": result=regression(root,dataset,cfg,req.get("values"))
    elif req["task"]=="classification": result=classification(root,dataset,cfg,req.get("values"))
    else: raise ValueError("Unsupported action.")
    print(json.dumps(json_clean(result),allow_nan=False))
try: main()
except Exception as e:
    print(json.dumps({"error":str(e),"trace":traceback.format_exc()}))
    sys.exit(1)
