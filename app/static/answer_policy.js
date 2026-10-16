/* 来源展示独立于答案状态；模型知识不能被显示成已有文档依据。 */
window.AnswerPolicyUI = {
  labels: {document_only:"文档依据",document_and_model:"文档＋模型知识",model_only:"模型通用知识"},
  sectionTitle(type) {
    return {document:"文档依据",model_knowledge:"通用知识补充 · 未由文档直接证明",limitation:"范围与缺失说明"}[type] || "回答内容";
  },
  basisLabel(data, fallback) {
    return this.labels[data.answer_basis] || fallback;
  },
  render(data, citations) {
    const container = node("div", {class:"answer-sections"});
    for (const section of data.answer_sections) {
      const block = node("section", {class:"answer-section source-"+section.source_type});
      block.append(node("h3", {class:"answer-source-title"}, this.sectionTitle(section.source_type)));
      // 只向文档区块传递引用；模型补充由安全文本渲染器展示，不附伪造文档来源。
      block.append(renderAnswer(section.content, section.source_type==="document" ? citations : []));
      container.append(block);
    }
    return container;
  }
};
