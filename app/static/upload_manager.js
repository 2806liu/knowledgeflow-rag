/* 每份文件独立传输与后台任务；客户端取消不冒充服务端取消。 */
window.UploadManager = {
  tasks: new Map(),
  labels: {transmitting:"正在传输",queued:"排队中",parsing:"解析 / OCR",chunking:"切块中",indexing:"索引中",
    ready:"已就绪",failed:"失败",cancel_requested:"正在取消",cancelled:"已取消",interrupted:"已中断",unknown:"状态待确认"},
  async add(files) {
    $("#file-input").value = "";
    if (!files.length) return;
    if (state.conversation?.history_meta?.archived_at) { toast("请先恢复已归档会话，再添加资料。",true); return; }
    if (files.length > 32) { toast("每批最多选择32份文件。",true); return; }
    const session = state.session, batch = crypto.randomUUID();
    const tasks = files.map(file => ({local_id:crypto.randomUUID(),session,file,filename:file.name,
      request_id:crypto.randomUUID(),batch_id:batch,batch_size:files.length,state:"queued",loaded:0,total:file.size}));
    // 用户主动添加文件时展开进度；任务结束后自动收起，不让状态行长期盖住聊天。
    const menu=$("#materials-menu");
    if (menu) { menu.open=true; menu.dataset.uploadOpened="1"; }
    tasks.forEach(task => this.tasks.set(task.local_id,task)); this.render();
    try {
      const value = await request("/conversations/"+session+"/batches",{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify({batch_id:batch,size:files.length})});
      if (session === state.session) ConversationUI.apply(value);
    } catch (error) {
      tasks.forEach(task => { task.state="failed"; task.error=error.message; }); this.render(); return;
    }
    // 顺序传输但不锁全局添加入口；后台由服务端单worker有界处理。
    for (const task of tasks) {
      if (task.state === "cancelled") continue;
      await this.send(task);
    }
    try {
      const value = await request("/conversations/"+session+"/batches/"+batch+"/finish",{method:"POST"});
      if (session === state.session) ConversationUI.apply(value);
    } catch (_) { /* 状态查询暂不可用时保留可恢复任务，不伪造完成。 */ }
  },
  transport(task) {
    return new Promise((resolve,reject) => {
      const xhr = new XMLHttpRequest(); task.xhr=xhr; task.state="transmitting"; task.error=null;
      xhr.open("POST","/ingestions"); xhr.timeout=120000;
      xhr.upload.onprogress = event => {
        task.loaded=event.loaded; task.total=event.total; task.transferred=event.lengthComputable && event.loaded>=event.total;
        this.render();
      };
      xhr.onload = () => {
        task.xhr=null; let value; try { value=JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status>=200 && xhr.status<300) resolve(value);
        else {
          const detail=value?.detail;
          reject(Object.assign(new Error(readableError(xhr.status,detail)),{status:xhr.status,code:detail?.code}));
        }
      };
      xhr.onerror = () => { task.xhr=null; reject(new Error("传输连接中断，可恢复状态后重试。")); };
      xhr.ontimeout = () => { task.xhr=null; reject(new Error("上传传输超时。")); };
      xhr.onabort = () => { task.xhr=null; reject(Object.assign(new Error("已停止传输，正在确认后台状态。"),{aborted:true})); };
      const form=new FormData();form.append("file",task.file);form.append("session_id",task.session);
      form.append("client_request_id",task.request_id);form.append("batch_id",task.batch_id);form.append("batch_size",task.batch_size);
      xhr.send(form); this.render();
    });
  },
  async lookup(task) {
    const rows=await request("/conversations/"+task.session+"/ingestions?client_request_id="+encodeURIComponent(task.request_id));
    return rows[0] || null;
  },
  async send(task) {
    if (!task.file) { task.state="failed";task.error="请重新选择原文件。";this.render();return; }
    if (!task.file.size || task.file.size>state.config.max_upload_bytes) {
      task.state="failed";task.error=task.file.size ? "文件超过单文件大小限制。" : "文件为空。";this.render();return;
    }
    for (let attempt=0;attempt<=2;attempt++) {
      try {
        // 同一请求标识先恢复服务端任务，响应丢失时不能重复创建资料。
        const previous=attempt ? await this.lookup(task) : null;
        const job=previous || await this.transport(task);
        task.job_id=job.id;this.accept(task,job);
        if (task.cancelledLocally && !["ready","cancelled"].includes(job.state)) await this.cancel(task);
        this.poll(task);return;
      } catch (error) {
        if (error.aborted || task.cancelledLocally) { await this.confirmCancelled(task);return; }
        task.error=error.message;this.render();
        if (error.status && error.status<500 && error.status!==429) {
          task.state="failed";task.retryable=false;this.render();return;
        }
        if (attempt<2) await new Promise(resolve=>setTimeout(resolve,500*2**attempt));
      }
    }
    task.state="unknown"; task.error ||= "未能确认上传结果，请查询状态。";this.render();
  },
  accept(task,job) {
    task.job_id=job.id;task.state=job.state;task.error=job.error?.message || null;task.retryable=job.error?.retryable;
    task.processed_pages=job.processed_pages;task.total_pages=job.total_pages;task.document_id=job.document_id;
    this.render();
  },
  async poll(task) {
    if (task.polling) return; task.polling=true;
    try {
      let failures=0;
      while (!["ready","failed","cancelled","interrupted"].includes(task.state)) {
        try {
          const job=await request("/ingestions/"+task.job_id);this.accept(task,job);failures=0;
        } catch (error) {
          task.error=error.message;this.render();
          if (++failures>2) { task.state="unknown";break; }
        }
        if (!["ready","failed","cancelled","interrupted"].includes(task.state))
          await new Promise(resolve=>setTimeout(resolve,800));
      }
      if (task.state==="ready") {
        await refreshDocuments();
        if (task.session===state.session) { await ConversationUI.load(); HistoryUI.refresh(); }
        toast(task.filename+" 已就绪，可用于问答。");
      } else if (task.session===state.session) {
        await ConversationUI.load();
      }
    } catch (error) { task.error=error.message; }
    finally { task.polling=false;this.render(); }
  },
  async confirmCancelled(task) {
    task.state="cancel_requested";this.render();
    try {
      let job=null;
      for (let i=0;i<3;i++) {
        job=await this.lookup(task);
        if (job) break;
        await new Promise(resolve=>setTimeout(resolve,400));
      }
      if (job) { task.job_id=job.id;await this.cancel(task);return; }
      // 字节已传完但受理结果未知，必须显示待确认，不能声称服务端已取消。
      task.state=task.transferred ? "unknown" : "cancelled";
      task.error=task.transferred ? "传输已停止，服务端受理结果待确认，请查询状态。" : null;
    } catch (error) { task.state="unknown";task.error=error.message; }
    finally { this.render(); }
  },
  async cancel(task) {
    task.cancelledLocally=true;
    if (task.state==="queued" && !task.job_id && !task.xhr) { task.state="cancelled";this.render();return; }
    if (task.xhr) { task.state="cancel_requested";this.render();task.xhr.abort();return; }
    if (!task.job_id) { await this.confirmCancelled(task);return; }
    task.state="cancel_requested";this.render();
    try {
      const job=await request("/ingestions/"+task.job_id+"/cancel",{method:"POST"});this.accept(task,job);this.poll(task);
    } catch (error) {
      // 发布先完成时以服务端ready为准，不删除已经就绪的资料。
      try { this.accept(task,await request("/ingestions/"+task.job_id)); } catch (_) {}
      task.error=error.message;
    } finally { this.render(); }
  },
  async retry(task) {
    task.cancelledLocally=false;task.error=null;
    try {
      if (task.state==="cancelled" && task.file) { await this.add([task.file]);return; }
      if (task.job_id && ["failed","interrupted"].includes(task.state)) {
        this.accept(task,await request("/ingestions/"+task.job_id+"/retry",{method:"POST"}));this.poll(task);
      } else {
        const job=await this.lookup(task);
        if (job) { this.accept(task,job);this.poll(task); }
        else await this.send(task);
      }
    } catch (error) { task.error=error.message; }
    finally { this.render(); }
  },
  async restore() {
    // 恢复上传列表时固定会话；迟到响应不能把旧会话任务挂到新会话名下。
    const session=state.session;
    const rows=await request("/conversations/"+session+"/ingestions");
    if (session!==state.session) return;
    for (const job of rows) {
      const task={local_id:job.id,session,request_id:job.request_id,batch_id:job.batch_id,
        batch_size:job.batch_size,filename:job.filename};
      this.tasks.set(task.local_id,task);this.accept(task,job);
      if (!["ready","failed","cancelled","interrupted"].includes(job.state)) this.poll(task);
    }
    if (state.conversation?.pending_focus && !rows.some(job=>!["ready","failed","cancelled","interrupted"].includes(job.state))) {
      try {
        const value=await request("/conversations/"+session+"/batches/"+state.conversation.focus_batch+"/finish",{method:"POST"});
        if (session===state.session) ConversationUI.apply(value);
      } catch (_) { /* 无法确认时继续显示待恢复，不假定后台已取消。 */ }
    }
    this.render();
  },
  render() {
    const box=$("#upload-tasks");
    if (!box) return;
    box.replaceChildren();
    // 已就绪文件由会话资料列表承接；这里只保留进行中和需要处理的任务。
    const tasks=[...this.tasks.values()].filter(t=>t.session===state.session && t.state!=="ready").slice(-12);
    const active=tasks.filter(t=>!["failed","interrupted","unknown","cancelled"].includes(t.state)).length;
    const badge=$("#upload-count");
    badge.textContent=active ? "处理中 "+active : "需处理 "+tasks.length;
    badge.hidden=!tasks.length;
    for (const task of tasks) {
      const row=node("div",{class:"upload-task","data-upload-state":task.state});
      const text=[this.labels[task.state] || task.state];
      if (task.state==="transmitting" && task.total) text.push("传输 "+Math.min(100,Math.round(task.loaded/task.total*100))+"%");
      if (task.total_pages) text.push(task.processed_pages+"/"+task.total_pages+"页");
      row.append(node("div",{},[node("strong",{},task.filename),node("small",{},text.join(" · ")),
        task.error ? node("small",{class:"upload-error"},task.error) : null]));
      if (!["ready","failed","cancelled","interrupted","unknown"].includes(task.state))
        row.append(action("取消",()=>this.cancel(task),"upload-action"));
      else if (task.state==="unknown" || task.state==="interrupted" || task.retryable || task.state==="cancelled" && task.file)
        row.append(action(task.state==="unknown" ? "查询 / 恢复" : "重试",()=>this.retry(task),"upload-action"));
      box.append(row);
    }
    $("#upload-section").hidden=!tasks.length;
    const menu=$("#materials-menu");
    if (!tasks.length && menu?.dataset.uploadOpened) { menu.open=false; delete menu.dataset.uploadOpened; }
  }
};
