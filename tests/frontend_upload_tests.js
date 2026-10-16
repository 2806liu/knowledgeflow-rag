/* 上传客户端的受控故障单测：模拟XHR，不替代真实浏览器点击验收。 */
"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
function setup() {
  let sequence=0;
  const requests=[];
  class Form {constructor(){this.values={};}append(k,v){this.values[k]=v;}}
  class XHR {
    static queue=[];static sent=[];static last=null;
    constructor(){this.upload={};}
    open(){}
    send(form){
      XHR.sent.push(form);XHR.last=this;
      const outcome=XHR.queue.shift() || "success";
      if(outcome==="manual")return;
      queueMicrotask(()=>{
        if(outcome==="timeout")this.ontimeout();
        else if(outcome==="invalid"){this.status=415;this.responseText=JSON.stringify({detail:{code:"UNSUPPORTED_TYPE",message:"格式错误"}});this.onload();}
        else{this.status=202;this.responseText=JSON.stringify({id:"job",state:"queued"});this.onload();}
      });
    }
    abort(){this.onabort();}
  }
  const context={window:{},state:{session:"session",config:{max_upload_bytes:1000},conversation:{id:"session"}},
    crypto:{randomUUID:()=>String(++sequence)},XMLHttpRequest:XHR,FormData:Form,console,
    setTimeout:(fn)=>setTimeout(fn,0),clearTimeout,$:()=>({value:""}),refreshDocuments:async()=>{},toast:()=>{},
    readableError:(status,detail)=>detail?.message || String(status),
    ConversationUI:{apply(value){context.state.conversation=value;},load:async()=>{}},
    request:async(url,options)=>{
      requests.push([url,options]);
      if(url.includes("client_request_id="))return [];
      if(url==="/ingestions/job")return {id:"job",state:"ready",document_id:"document"};
      return {id:"session",pending_focus:false};
    }};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(process.env.KF_STATIC_DIR || path.resolve(__dirname,"../app/static"),"upload_manager.js"),"utf8"),context);
  const manager=context.window.UploadManager;manager.render=()=>{};
  const task={local_id:"task",session:"session",file:{size:20,name:"fixture.md"},filename:"fixture.md",request_id:"same-key",batch_id:"batch",batch_size:1,state:"queued"};
  manager.tasks.set("task",task);
  return {context,manager,task,XHR,requests};
}
async function settle(task){for(let i=0;i<30 && task.polling;i++)await new Promise(r=>setTimeout(r,0));}
const tests=[
 ["format error remains failed",async()=>{
   const {manager,task,XHR}=setup();XHR.queue=["invalid"];await manager.send(task);
   assert.equal(task.state,"failed");assert.equal(task.retryable,false);assert.equal(XHR.sent.length,1);
 }],
 ["network retries are bounded and keep idempotency",async()=>{
   const {manager,task,XHR}=setup();XHR.queue=["timeout","timeout","timeout"];await manager.send(task);
   assert.equal(task.state,"unknown");assert.equal(XHR.sent.length,3);
   assert.ok(XHR.sent.every(form=>form.values.client_request_id==="same-key"));assert.equal(task.xhr,null);
 }],
 ["lost response recovers without duplicate transmit",async()=>{
   const {context,manager,task,XHR}=setup();XHR.queue=["timeout"];
   const original=context.request;
   context.request=async(url,options)=>url.includes("client_request_id=")?[{id:"job",state:"ready"}]:original(url,options);
   await manager.send(task);await settle(task);
   assert.equal(XHR.sent.length,1);assert.equal(task.state,"ready");
 }],
 ["cancel in transfer gives immediate pending state",async()=>{
   const {manager,task,XHR}=setup();XHR.queue=["manual"];
   const sending=manager.send(task);assert.ok(task.xhr);await manager.cancel(task);
   assert.equal(task.state,"cancel_requested");await sending;assert.equal(task.state,"cancelled");
 }],
 ["accepted result unknown cannot claim cancelled",async()=>{
   const {manager,task,XHR}=setup();XHR.queue=["manual"];
   const sending=manager.send(task);task.transferred=true;await manager.cancel(task);await sending;
   assert.equal(task.state,"unknown");
 }],
 ["poll failure does not fabricate ready",async()=>{
   const {context,manager,task}=setup();task.job_id="job";task.state="parsing";
   context.request=async()=>{throw new Error("offline");};
   await manager.poll(task);assert.equal(task.state,"unknown");assert.equal(task.polling,false);
 }],
 ["completion wins cancel race",async()=>{
   const {context,manager,task}=setup();task.job_id="job";task.state="parsing";
   context.request=async(url)=>{if(url.endsWith("/cancel"))throw new Error("already completed");return {id:"job",state:"ready"};};
   await manager.cancel(task);assert.equal(task.state,"ready");
 }],
 ["restore abandoned transmission clears only empty batch",async()=>{
   const {context,manager,requests}=setup();
   context.state.conversation={id:"session",pending_focus:true,focus_batch:"batch"};
   context.request=async(url,options)=>{
     requests.push([url,options]);
     return url.endsWith("/finish")?{id:"session",pending_focus:false}:[];
   };
   await manager.restore();assert.equal(context.state.conversation.pending_focus,false);
   assert.ok(requests.some(([url])=>url.endsWith("/batches/batch/finish")));
 }]
];
(async()=>{
 const records=[];
 for(const [name,test] of tests){await test();records.push({name,passed:true});}
 console.log(JSON.stringify({passed:true,checks:records.length,records,browser_acceptance:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
