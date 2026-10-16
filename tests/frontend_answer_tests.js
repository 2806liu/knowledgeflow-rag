/* 客户端来源标记检查；使用最小DOM替身，不冒充浏览器点击验收。 */
const fs=require("fs"), vm=require("vm"), path=require("path"), assert=require("assert");
const captures=[];
const context={window:{},node:(tag,attrs={},text)=>({tag,attrs,text,children:[],append(x){this.children.push(x)}}),
  renderAnswer:(text,citations)=>{captures.push({text,citations});return {text,citations}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(process.env.KF_STATIC_DIR,"answer_policy.js"),"utf8"),context);
const ui=context.window.AnswerPolicyUI;
assert.equal(ui.basisLabel({answer_basis:"model_only"},"知识库回答"),"模型通用知识");
assert.equal(ui.basisLabel({answer_basis:"document_and_model"},"知识库回答"),"文档＋模型知识");
ui.render({answer_sections:[{source_type:"model_knowledge",content:"<script>unsafe</script>",citation_numbers:[]}]},[{citation_number:1}]);
assert.equal(captures[0].citations.length,0);
assert.equal(captures[0].text,"<script>unsafe</script>");
process.stdout.write(JSON.stringify({checks:4,browser_acceptance:false}));
