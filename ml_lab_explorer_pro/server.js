const express=require('express'),fs=require('fs'),path=require('path'),{spawn}=require('child_process');
const app=express(),PORT=process.env.PORT||3000,ROOT=__dirname;
app.use(express.json({limit:'2mb'}));app.use(express.static(path.join(ROOT,'public')));
const DATA={
 airlineTrain:{label:'Airline · train.csv',task:'clustering',path:'data/airline_clustering/train.csv'},
 airlineTest:{label:'Airline · test.csv',task:'clustering',path:'data/airline_clustering/test.csv'},
 plant1:{label:'Solar Plant 1 · generation + weather',task:'regression',plant:1,path:'data/solar/Plant_1_Generation_Data.csv',weather:'data/solar/Plant_1_Weather_Sensor_Data.csv'},
 plant2:{label:'Solar Plant 2 · generation + weather',task:'regression',plant:2,path:'data/solar/Plant_2_Generation_Data.csv',weather:'data/solar/Plant_2_Weather_Sensor_Data.csv'},
 machine:{label:'Machine failure classification',task:'classification',path:'data/classification_dataset/machine_failure.csv'}
};
function runPython(payload){
 return new Promise((resolve,reject)=>{
  const command=process.env.PYTHON||'python';
  const child=spawn(command,[path.join(ROOT,'ml_engine.py')],{cwd:ROOT,windowsHide:true});
  let out='',err='';child.stdout.on('data',d=>out+=d);child.stderr.on('data',d=>err+=d);
  child.on('error',e=>reject(new Error('Could not start Python. Install Python and requirements.txt. Details: '+e.message)));
  child.on('close',code=>{if(code!==0)return reject(new Error(err.trim()||'Python ML engine failed.'));try{resolve(JSON.parse(out));}catch(e){reject(new Error('Could not parse ML engine output. '+(err||out).slice(-700)));}});
  child.stdin.write(JSON.stringify(payload));child.stdin.end();
 });
}
app.get('/api/datasets',(req,res)=>res.json(Object.entries(DATA).map(([id,d])=>({id,label:d.label,task:d.task,exists:fs.existsSync(path.join(ROOT,d.path))}))));
app.get('/api/data/:id',async(req,res)=>{
 try{const d=DATA[req.params.id];if(!d)return res.status(404).json({error:'Unknown dataset'});const result=await runPython({action:'inspect',root:ROOT,dataset:req.params.id,config:d});res.json({...result,id:req.params.id,label:d.label,task:d.task});}
 catch(e){res.status(500).json({error:e.message});}
});
app.post('/api/run',async(req,res)=>{
 try{const {dataset,task,algorithm,k}=req.body;const d=DATA[dataset];if(!d)return res.status(400).json({error:'Choose a dataset.'});if(d.task!==task)return res.status(400).json({error:'Dataset does not match selected ML task.'});
 const result=await runPython({action:'run',root:ROOT,dataset,config:d,task,algorithm,k,features:req.body.features,target:req.body.target,values:req.body.values});
 res.json(result);
 }catch(e){res.status(400).json({error:e.message});}
});
app.listen(PORT,()=>console.log(`ML Lab Explorer running at http://localhost:${PORT}`));
