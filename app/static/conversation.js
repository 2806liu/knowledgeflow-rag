/* 会话恢复与资料范围；服务端状态为权威，浏览器只保存恢复入口。 */
window.ConversationUI = {
  switchRevision: 0,
  turnCursor: null,
  async initialize() {
    let saved; try { saved = sessionStorage.getItem("knowledgeflow-session-v2"); } catch (_) {}
    if (saved) {
      try { await this.load(saved, true); return; } catch (_) {}
    }
    await this.create();
  },
  async create() {
    const value = await request("/conversations", {method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({answer_policy:"document_enhanced"})});
    this.switchRevision++;
    state.session = value.id; state.conversation = {...value, history_meta:{archived_at:null}};
    $("#answer-policy").value = "document_enhanced";
    try { sessionStorage.setItem("knowledgeflow-session-v2", value.id); } catch (_) {}
    state.selected = new Set(); this.turnCursor=null; $("#turns-more").hidden=true;
    this.setArchived(false); updateScope(); this.render();
  },
  async load(id = state.session, restore = false) {
    const revision = restore ? ++this.switchRevision : this.switchRevision;
    const value = await request("/conversations/" + encodeURIComponent(id) + "?turn_limit=50");
    if (restore && revision !== this.switchRevision) return null;
    if (!restore && id !== state.session) return;
    // 切换会话先清掉上一会话的引用和可见轮次，避免旧证据短暂显示在新主题旁边。
    if (restore) { renderSources([]); $("#sources-panel").classList.remove("is-open"); }
    state.session = id; this.apply(value);
    try { sessionStorage.setItem("knowledgeflow-session-v2", id); } catch (_) {}
    if (restore) {
      const box = $("#conversation"); $$(".message,.loading-message",box).forEach(item => item.remove());
      this.turnCursor=value.next_turn_cursor;
      $("#turns-more").hidden=!this.turnCursor;
      for (const turn of value.turns) {
        box.append(node("article",{class:"message user"},node("div",{class:"user-bubble"},turn.question)));
        showAnswer(turn.response,box,false);
      }
      renderSources(value.turns.at(-1)?.response?.citations || [],null,false);
      $("#sources-panel").classList.remove("is-open");
      $("#welcome").hidden = value.turns.length > 0; scrollConversation();
    }
    return value;
  },
  apply(value) {
    if (!value || value.id !== state.session) return;
    if (!value.history_meta && state.conversation?.id===value.id)
      value.history_meta=state.conversation.history_meta;
    state.conversation = value;
    $("#answer-policy").value = value.answer_policy || "document_enhanced";
    state.selected = new Set(value.scope_mode === "all" ? [] : value.document_ids);
    this.setArchived(Boolean(value.history_meta?.archived_at));
    updateScope(); this.render();
  },
  setArchived(archived) {
    // 已归档会话保持可阅读；恢复归档后才允许继续改范围和发送问题。
    $("#archive-notice").hidden=!archived;
    $("#question").disabled=archived;
    $("#composer-upload").disabled=archived;
    $("#scope-button").disabled=archived || state.asking;
    $("#answer-policy").disabled=archived || state.asking;
    $("#send-button").disabled=archived || state.asking || !$("#question").value.trim();
  },
  async loadOlder() {
    if (!this.turnCursor) return;
    const session=state.session, cursor=this.turnCursor;
    const box=$("#conversation"), previousHeight=box.scrollHeight;
    $("#turns-more").disabled=true;
    try {
      const page=await request("/conversations/"+encodeURIComponent(session)+"/turns?limit=50&before="+
                               encodeURIComponent(cursor));
      if (session!==state.session) return;
      const fragment=document.createDocumentFragment();
      for (const turn of page.items) {
        fragment.append(node("article",{class:"message user"},node("div",{class:"user-bubble"},turn.question)));
        showAnswer(turn.response,fragment,false);
      }
      $("#turns-more").after(fragment);
      this.turnCursor=page.next_cursor;
      $("#turns-more").hidden=!this.turnCursor;
      box.scrollTop+=box.scrollHeight-previousHeight;
    } catch (error) { toast(error.message,true); }
    finally { $("#turns-more").disabled=false; }
  },
  async setScope(mode, ids, active) {
    const conversation = state.conversation;
    // 已删除资料可留在历史审计中，但不能再次作为当前范围提交。
    ids = ids.filter(id=>state.docs.some(doc=>doc.document_id===id && doc.chunk_count>0));
    if (active && !state.docs.some(doc=>doc.document_id===active)) active=null;
    if (mode !== "all") {
      const missing = ids.filter(id => !(conversation.associated_document_ids || conversation.document_ids).includes(id));
      if (missing.length) {
        const attached = await request("/conversations/"+state.session+"/documents",{
          method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({document_ids:missing})});
        this.apply(attached);
      }
    }
    const value = await request("/conversations/"+state.session+"/scope",{
      method:"PATCH",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode,document_ids:ids,active_document_id:active || null,revision:state.conversation.revision})});
    this.apply(value);
    HistoryUI.refresh();
  },
  async detach(id) {
    try {
      this.apply(await request("/conversations/"+state.session+"/documents/"+id,{method:"DELETE"}));
      HistoryUI.refresh();
    }
    catch (error) { toast(error.message,true); }
  },
  render() {
    // 资料只渲染到顶部弹层；输入区和对话区不再因附件数量变化而被挤压。
    const bar = $("#session-attachments");
    if (!bar) return;
    bar.replaceChildren(); const value = state.conversation;
    if (!value) { $("#materials-count").textContent = "0"; return; }
    const docs = state.docs.filter(doc => (value.associated_document_ids || value.document_ids).includes(doc.document_id));
    $("#materials-count").textContent = String(docs.length);
    if (!docs.length) { bar.append(node("span",{class:"attachment-hint"},"本会话尚未关联资料，可添加附件或选择知识库资料。")); return; }
    for (const doc of docs) {
      const focused = value.active_document_id === doc.document_id;
      const chip = node("div",{class:"attachment-chip"+(focused?" current":"")});
      const select = action((focused ? "当前 · " : "") + doc.filename,async () => {
        try { await this.setScope(value.scope_mode,[...new Set([...value.document_ids,doc.document_id])],doc.document_id); }
        catch (error) { toast(error.message,true); }
      });
      select.title = "设为当前讨论文档"; select.disabled = state.asking || Boolean(value.history_meta?.archived_at);
      const remove = action("×",()=>this.detach(doc.document_id),"attachment-remove");
      remove.disabled = Boolean(value.history_meta?.archived_at);
      chip.append(select,remove);
      bar.append(chip);
    }
    bar.append(node("span",{class:"attachment-hint"},value.pending_focus ? "新资料处理中，尚未就绪。" :
      value.active_document_id ? "继续添加会保留已有附件。" : "多份资料：请点击指定当前文档，或要求分别概述。"));
  }
};
