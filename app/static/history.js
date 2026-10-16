/* 历史侧栏只读取轻量索引；切换后由会话模块分页恢复轮次与独立范围。 */
window.HistoryUI = {
  archived: false,
  cursor: null,
  loading: false,
  revision: 0,
  editId: null,

  initialize() {
    $("#history-toggle").addEventListener("click", () => this.drawer(true));
    $("#history-close").addEventListener("click", () => this.drawer(false));
    $("#history-backdrop").addEventListener("click", () => this.drawer(false));
    $("#sidebar-new-chat").addEventListener("click", newChat);
    $("#history-more").addEventListener("click", () => this.more());
    $("#history-search").addEventListener("input", () => {
      clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => this.refresh(), 180);
    });
    $$("[data-history-filter]").forEach(button => button.addEventListener("click", () => {
      this.archived = button.dataset.historyFilter === "archived";
      $$("[data-history-filter]").forEach(item => item.classList.toggle("selected", item === button));
      this.refresh();
    }));
    $("#title-form").addEventListener("submit", event => this.rename(event));
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") this.drawer(false);
    });
  },

  drawer(open) {
    $("#history-panel").classList.toggle("is-open", open);
    $("#history-backdrop").hidden = !open;
    $("#history-toggle").setAttribute("aria-expanded", String(open));
  },

  async refresh() {
    // 搜索、筛选或新轮次变化时丢弃旧分页响应，避免列表混入另一筛选状态。
    this.revision += 1;
    this.cursor = null;
    $("#history-list").replaceChildren();
    $("#history-more").hidden = true;
    await this.more();
  },

  async more() {
    const revision = this.revision;
    if (this.loading && this.loadingRevision === revision) return;
    this.loading = true;
    this.loadingRevision = revision;
    const params = new URLSearchParams({archived: String(this.archived), limit: "30"});
    const query = $("#history-search").value.trim();
    if (query) params.set("q", query);
    if (this.cursor) params.set("cursor", this.cursor);
    try {
      const page = await request("/conversations?" + params);
      if (revision !== this.revision) return;
      for (const item of page.items) $("#history-list").append(this.row(item));
      this.cursor = page.next_cursor;
      $("#history-more").hidden = !this.cursor;
      if (!$("#history-list").children.length)
        $("#history-list").append(node("p", {class:"history-empty"},
          this.archived ? "还没有已归档会话" : query ? "没有匹配的会话" : "新建会话后，历史会显示在这里"));
    } catch (error) {
      if (revision === this.revision) $("#history-list").append(node("p",{class:"history-empty"},error.message));
    } finally { if (this.loadingRevision === revision) this.loading = false; }
  },

  row(item) {
    const row = node("div", {class:"history-row" + (item.session_id === state.session ? " current" : "")});
    const label = item.scope_mode === "all" ? "全部资料" :
      "范围 " + item.selected_document_count + " · 关联 " + item.associated_document_count;
    const time = new Date(item.last_turn_at * 1000).toLocaleString("zh-CN", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
    const open = action(item.title, () => this.open(item.session_id), "history-open");
    open.title = item.title;
    if (item.session_id === state.session) open.setAttribute("aria-current", "true");
    const detail = node("small",{},label + " · " + time);
    const rename = action("改名", () => this.startRename(item), "history-action");
    rename.setAttribute("aria-label", "重命名会话：" + item.title);
    const archive = action(this.archived ? "恢复" : "归档", () => this.setArchive(item), "history-action");
    archive.setAttribute("aria-label", (this.archived ? "恢复会话：" : "归档会话：") + item.title);
    row.append(node("div",{class:"history-row-main"},[open,detail]),
               node("div",{class:"history-row-actions"},[rename,archive]));
    return row;
  },

  async open(id) {
    if (!ensureIdle()) return;
    try {
      if (id !== state.session) {
        const loaded = await ConversationUI.load(id, true);
        if (!loaded) return;
        $("#question").value = "";
        $("#send-button").disabled = true;
        UploadManager.render();
        await UploadManager.restore();
      }
      showPage("chat");
      this.drawer(false);
      await this.refresh();
    } catch (error) { toast(error.message,true); }
  },

  startRename(item) {
    this.editId = item.session_id;
    $("#title-input").value = item.title;
    $("#title-dialog").showModal();
    $("#title-input").focus();
  },

  async rename(event) {
    event.preventDefault();
    const title = $("#title-input").value.trim();
    if (!title || !this.editId) return;
    try {
      const result = await request("/conversations/"+encodeURIComponent(this.editId), {
        method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({title})});
      if (result.session_id === state.session && state.conversation?.history_meta)
        state.conversation.history_meta.title = result.title;
      $("#title-dialog").close();
      await this.refresh();
    } catch (error) { toast(error.message,true); }
  },

  async setArchive(item) {
    if (item.session_id === state.session && !ensureIdle()) return;
    try {
      const meta = await request("/conversations/"+encodeURIComponent(item.session_id), {
        method:"PATCH",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({archived: !this.archived})});
      if (item.session_id === state.session) {
        state.conversation.history_meta = meta;
        ConversationUI.setArchived(Boolean(meta.archived_at));
      }
      await this.refresh();
    } catch (error) { toast(error.message,true); }
  }
};
