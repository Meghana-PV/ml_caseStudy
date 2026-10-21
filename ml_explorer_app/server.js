const express = require('express');
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');
const app = express();
const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
app.use(express.json({limit:'5mb'}));
app.use(express.static(path.join(ROOT,'public')));
const DATASETS = {
  airlineTrain: {label:'Airline clustering · train', path:'data/airline_clustering/train.csv', task:'clustering'},
  airlineTest: {label:'Airline clustering · test', path:'data/airline_clustering/test.csv', task:'clustering'},
  machine: {label:'Machine failure classification', path:'data/classification_dataset/machine_failure.csv', task:'classification'},
  plant1: {label:'Solar Plant 1 · generation', path:'data/solar/Plant_1_Generation_Data.csv', task:'regression'},
  plant1Weather: {label:'Solar Plant 1 · weather', path:'data/solar/Plant_1_Weather_Sensor_Data.csv', task:'regression'},
  plant2: {label:'Solar Plant 2 · generation', path:'data/solar/Plant_2_Generation_Data.csv', task:'regression'},
  plant2Weather: {label:'Solar Plant 2 · weather', path:'data/solar/Plant_2_Weather_Sensor_Data.csv', task:'regression'}
};
function readCsv(file) {
  const text = fs.readFileSync(path.join(ROOT,file),'utf8');
  return parse(text,{columns:true,skip_empty_lines:true,relax_column_count:true,trim:true});
}
function numeric(v){ if(v===null||v===undefined||String(v).trim()==='') return NaN; const n=Number(v); return Number.isFinite(n)?n:NaN; }
function summarize(rows) {
  const columns=Object.keys(rows[0]||{});
  return columns.map(name=>{
    const vals=rows.map(r=>r[name]).filter(v=>v!==''&&v!==null&&v!==undefined);
    const nums=vals.map(numeric).filter(Number.isFinite);
    const unique=new Set(vals.map(String)).size;
    return {name,nonEmpty:vals.length,missing:rows.length-vals.length,unique,numeric:nums.length>0&&nums.length>=vals.length*.7,
      min:nums.length?Math.min(...nums.slice(0,50000)):null,max:nums.length?Math.max(...nums.slice(0,50000)):null,
      mean:nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:null};
  });
}
function sampleRows(rows,limit=8){return rows.slice(0,limit);}
function numericMatrix(rows,cols){
  return rows.map(r=>cols.map(c=>numeric(r[c]))).filter(row=>row.every(Number.isFinite));
}
function standardize(X){
  const n=X.length,p=X[0]?.length||0, means=[],stds=[];
  for(let j=0;j<p;j++){let m=X.reduce((s,r)=>s+r[j],0)/n;let v=X.reduce((s,r)=>s+(r[j]-m)**2,0)/n;means.push(m);stds.push(Math.sqrt(v)||1);}
  return {X:X.map(r=>r.map((v,j)=>(v-means[j])/stds[j])),means,stds};
}
function kmeans(X,k,maxIter=40){
  if(X.length<k) throw Error('Not enough numeric rows for this number of clusters.');
  let centers=[];
  for(let i=0;i<k;i++) centers.push([...X[Math.floor(i*X.length/k)]]);
  let labels=new Array(X.length).fill(0);
  for(let iter=0;iter<maxIter;iter++){
    let changed=false;
    for(let i=0;i<X.length;i++){
      let best=0,bd=Infinity;
      for(let c=0;c<k;c++){let d=X[i].reduce((s,v,j)=>s+(v-centers[c][j])**2,0);if(d<bd){bd=d;best=c;}}
      if(labels[i]!==best){labels[i]=best;changed=true;}
    }
    const sums=Array.from({length:k},()=>Array(X[0].length).fill(0)),counts=Array(k).fill(0);
    X.forEach((r,i)=>{counts[labels[i]]++;r.forEach((v,j)=>sums[labels[i]][j]+=v);});
    for(let c=0;c<k;c++) if(counts[c]) centers[c]=sums[c].map(v=>v/counts[c]);
    if(!changed) break;
  }
  let inertia=0;X.forEach((r,i)=>{inertia+=r.reduce((s,v,j)=>s+(v-centers[labels[i]][j])**2,0);});
  return {labels,centers,inertia};
}
function solve(A,b){
  const n=b.length, M=A.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<n;i++){let max=i;for(let j=i+1;j<n;j++)if(Math.abs(M[j][i])>Math.abs(M[max][i]))max=j;[M[i],M[max]]=[M[max],M[i]];
    let pivot=M[i][i];if(Math.abs(pivot)<1e-10)continue;for(let j=i;j<=n;j++)M[i][j]/=pivot;
    for(let r=0;r<n;r++)if(r!==i){let f=M[r][i];for(let c=i;c<=n;c++)M[r][c]-=f*M[i][c];}
  }return M.map(r=>r[n]);
}
function linearRegression(rows,features,target){
  const usable=rows.map(r=>({x:features.map(c=>numeric(r[c])),y:numeric(r[target])})).filter(o=>o.x.every(Number.isFinite)&&Number.isFinite(o.y));
  const data=usable.slice(0,8000); if(data.length<5) throw Error('Not enough complete numeric rows for regression.');
  const X=data.map(o=>[1,...o.x]),y=data.map(o=>o.y),p=X[0].length;
  const A=Array.from({length:p},()=>Array(p).fill(0)),b=Array(p).fill(0);
  X.forEach((r,i)=>r.forEach((v,j)=>{b[j]+=v*y[i];r.forEach((w,k)=>A[j][k]+=v*w);}));
  for(let j=1;j<p;j++) A[j][j]+=1e-8;
  const beta=solve(A,b),pred=X.map(r=>r.reduce((s,v,j)=>s+v*beta[j],0)),mean=y.reduce((s,v)=>s+v,0)/y.length;
  const ssRes=y.reduce((s,v,i)=>s+(v-pred[i])**2,0),ssTot=y.reduce((s,v)=>s+(v-mean)**2,0);
  const mae=y.reduce((s,v,i)=>s+Math.abs(v-pred[i]),0)/y.length,rmse=Math.sqrt(ssRes/y.length);
  return {rows:data.length,metrics:{r2:ssTot?1-ssRes/ssTot:0,rmse,mae},coefficients:features.map((f,i)=>({feature:f,coefficient:beta[i+1]})),target,actualPredicted:y.slice(0,120).map((actual,i)=>({actual,predicted:pred[i]}))};
}
function logistic(rows,features,target){
  const data=rows.map(r=>({x:features.map(c=>numeric(r[c])),y:String(r[target])})).filter(o=>o.x.every(Number.isFinite)&&o.y!=='');
  const classes=[...new Set(data.map(o=>o.y))].slice(0,2);
  if(classes.length!==2) throw Error('Choose a target column with exactly two classes for this simple logistic classifier.');
  const use=data.slice(0,10000), raw=use.map(o=>o.x), norm=standardize(raw), X=norm.X, y=use.map(o=>o.y===classes[1]?1:0);
  let w=Array(features.length+1).fill(0),lr=.08;
  for(let epoch=0;epoch<220;epoch++){let g=Array(w.length).fill(0);X.forEach((r,i)=>{let z=w[0]+r.reduce((s,v,j)=>s+v*w[j+1],0);z=Math.max(-30,Math.min(30,z));let p=1/(1+Math.exp(-z)),e=p-y[i];g[0]+=e;r.forEach((v,j)=>g[j+1]+=e*v);});w=w.map((v,j)=>v-lr*g[j]/X.length);}
  let tp=0,tn=0,fp=0,fn=0,correct=0;
  X.forEach((r,i)=>{let z=w[0]+r.reduce((s,v,j)=>s+v*w[j+1],0),pred=1/(1+Math.exp(-z))>=.5?1:0;if(pred===y[i])correct++;if(pred&&y[i])tp++;else if(pred&&!y[i])fp++;else if(!pred&&y[i])fn++;else tn++;});
  return {rows:use.length,classes,metrics:{accuracy:correct/use.length,precision:tp+fp?tp/(tp+fp):0,recall:tp+fn?tp/(tp+fn):0,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):0},confusion:{tp,fp,tn,fn},coefficients:features.map((f,i)=>({feature:f,coefficient:w[i+1]}))};
}
app.get('/api/datasets',(req,res)=>res.json(Object.entries(DATASETS).map(([id,d])=>({id,label:d.label,task:d.task,exists:fs.existsSync(path.join(ROOT,d.path))}))));
app.get('/api/data/:id',(req,res)=>{
 try{const d=DATASETS[req.params.id];if(!d)return res.status(404).json({error:'Unknown dataset'});const rows=readCsv(d.path);res.json({id:req.params.id,label:d.label,task:d.task,totalRows:rows.length,columns:summarize(rows),preview:sampleRows(rows,10),rows:rows.length});}
 catch(e){res.status(500).json({error:e.message});}
});
app.post('/api/run',(req,res)=>{
 try{
  const {dataset,task,features,target,k}=req.body;const d=DATASETS[dataset];if(!d)return res.status(400).json({error:'Choose a dataset.'});
  const rows=readCsv(d.path);if(!Array.isArray(features)||!features.length)return res.status(400).json({error:'Select at least one feature.'});
  const missing=[...features,target].filter(c=>c&&!Object.keys(rows[0]||{}).includes(c));if(missing.length)return res.status(400).json({error:'Column not found: '+missing.join(', ')});
  let result;
  if(task==='clustering'){
    const cols=features.filter(c=>rows.some(r=>Number.isFinite(numeric(r[c]))));const all=numericMatrix(rows.slice(0,15000),cols);
    if(cols.length<2||all.length<10)throw Error('Select at least two numeric columns and ensure at least 10 usable rows.');
    const norm=standardize(all),kk=Math.max(2,Math.min(10,Number(k)||4)),model=kmeans(norm.X,kk);
    const sizes=Array(kk).fill(0);model.labels.forEach(v=>sizes[v]++);
    result={algorithm:'K-Means clustering',rows:all.length,features:cols,k:kk,inertia:model.inertia,clusterSizes:sizes,scatter: norm.X.slice(0,800).map((r,i)=>({x:r[0],y:r[1],cluster:model.labels[i]})),note:'Features standardized before clustering. For performance, clustering uses at most 15,000 rows.'};
  } else if(task==='regression'){
    if(!target)throw Error('Choose a numeric target column.');
    result={algorithm:'Multiple Linear Regression',...linearRegression(rows,features,target),note:'Transparent least-squares baseline; uses up to 8,000 complete rows. Compare with the notebook models for full experiment results.'};
  } else {
    if(!target)throw Error('Choose a target column.');
    result={algorithm:'Binary Logistic Regression',...logistic(rows,features,target),note:'Simple binary logistic regression baseline trained on up to 10,000 rows. For multiclass targets, select a binary target or use the notebook experiments.'};
  }
  res.json(result);
 }catch(e){res.status(400).json({error:e.message});}
});
app.listen(PORT,()=>console.log(`ML Lab Explorer running at http://localhost:${PORT}`));
