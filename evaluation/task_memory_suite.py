"""公开合成的冻结评测集；只包含虚构资料，不读取用户上传文档。"""

SUITE_VERSION = "task-memory-v1"

# 每个段落独立成为一个原文块，故意让时间、次数等方面分布在不同块中。
SOURCES = {
    "lab": ("Public_Lab_Rules.md", [
        "实验室考勤时间：周一至周五在08:00-09:00、14:00-15:00、18:30-19:30三个固定时段打卡。",
        "实验室打卡次数：每日采用4次采样法，三个固定时段各打卡一次，最后离开实验室时再打卡一次。",
        "实验室迟到认定：超过固定时段的结束时间打卡，认定为迟到；请假必须事先获得批准。",
        "课程报告文件命名规则：学号-姓名-报告。课程报告只接收PDF格式。",
        "课程报告截止时间：2026年6月30日18:00。个人学习计划不能修改课程报告截止时间。",
    ]),
    "library": ("Public_Library_Rules.md", [
        "图书馆借阅数量：每位读者最多同时借阅6本图书。",
        "图书馆借阅期限：每本书初始借期为30天。",
        "图书馆续借规定：没有其他读者预约的图书，可以续借1次，续借期限为15天。",
    ]),
    "alpha": ("Synthetic_Alpha_Kinematic_Control.md", [
        "Abstract. Alpha investigates task-priority kinematic control of redundant manipulators. Its focus is ordering multiple motion tasks by priority. 中文对照：Alpha研究任务优先级运动学控制，按照优先级组织多个运动任务。",
        "Conclusion. Alpha preserves the primary end-effector task and places secondary objectives in its null space. 中文对照：Alpha的结论是在保持末端主任务的同时，将次任务放入零空间。",
    ]),
    "beta": ("Synthetic_Beta_Redundancy_Resolution.md", [
        "Abstract. Beta investigates constrained redundancy resolution by quadratic programming. Its focus is handling joint limits and velocity constraints with a QP formulation. 中文对照：Beta研究二次规划冗余求解，重点处理关节限位和速度约束。",
        "Conclusion. Beta incorporates joint and velocity bounds into one constrained optimization problem. 中文对照：Beta的结论是将关节边界与速度边界纳入统一的约束优化问题。",
    ]),
}


def fact(document, *patterns):
    """同一事实的多个正则必须全部满足；不把关键词出现等同于语义正确。"""
    return {"document": document, "patterns": list(patterns)}


TIME = fact("lab", r"(?:08|8)[:：]00\s*[-—~至到–]\s*(?:09|9)[:：]00",
            r"14[:：]00\s*[-—~至到–]\s*15[:：]00", r"18[:：]30\s*[-—~至到–]\s*19[:：]30")
COUNT = fact("lab", r"(?:4|四)\s*次", r"(?:离开|离室)")
ALPHA = fact("alpha", r"任务优先级|task.priority|优先级.*任务")
BETA = fact("beta", r"二次规划|quadratic.programming|\bQP\b", r"关节", r"速度")
ALPHA_CONCLUSION = fact("alpha", r"主任务|主要任务|末端.*任务", r"零空间|null.space")
BETA_CONCLUSION = fact("beta", r"关节", r"速度", r"约束优化|约束.*优化|constrained.optimization")


def step(question, scope, facts=(), *, refusal=False, both_sources=False):
    """一次预先规定的提问；后续轮次不是失败后追加的人工提示。"""
    return dict(question=question, scope=list(scope), facts=list(facts), refusal=refusal,
                both_sources=both_sources)


# 前18个为可回答任务，后2个为缺证据负例；拒答不能提高正例任务完成率。
TASKS = [
    dict(id="T01", category="single_fact", steps=[step("实验室规定的打卡时间是什么？", ["lab"], [TIME])]),
    dict(id="T02", category="single_fact", steps=[step("实验室每天打卡几次，最后一次什么时候打？", ["lab"], [COUNT])]),
    dict(id="T03", category="multiple_aspects", steps=[step("实验室打卡时间和次数分别是什么？", ["lab"], [TIME, COUNT])]),
    dict(id="T04", category="single_fact", steps=[step("实验室制度如何认定迟到？", ["lab"], [fact("lab", r"(?:超过|超出|晚于|迟于)", r"(?:结束|时段|规定时间)", r"迟到")])]),
    dict(id="T05", category="single_fact", steps=[step("课程报告文件如何命名？", ["lab"], [fact("lab", r"学号\s*[-—]\s*姓名\s*[-—]\s*报告")])]),
    dict(id="T06", category="single_fact", steps=[step("课程报告规定提交什么文件格式？", ["lab"], [fact("lab", r"PDF")])]),
    dict(id="T07", category="single_fact", steps=[step("课程报告截止时间是什么？", ["lab"], [fact("lab", r"6\s*月\s*30\s*日|06[-/]30|6[-/]30", r"18[:：]00")])]),
    dict(id="T08", category="single_fact", steps=[step("图书馆每位读者最多同时借几本书？", ["library"], [fact("library", r"(?:6|六)\s*本")])]),
    dict(id="T09", category="single_fact", steps=[step("图书馆初始借阅期限是多少天？", ["library"], [fact("library", r"(?:30|三十)\s*天")])]),
    dict(id="T10", category="multiple_aspects", steps=[step("图书馆续借需要什么条件，可以续借几次、多久？", ["library"], [fact("library", r"(?:没有|无|未被|无人|未有).{0,15}预约", r"(?:1|一)\s*次", r"(?:15|十五)\s*天")])]),
    dict(id="T11", category="paper_fact", steps=[step("Synthetic Alpha论文的研究重点是什么？", ["alpha"], [ALPHA])]),
    dict(id="T12", category="paper_fact", steps=[step("Synthetic Beta论文的研究重点是什么？", ["beta"], [BETA])]),
    dict(id="T13", category="paper_fact", steps=[step("Synthetic Alpha论文的主要结论是什么？", ["alpha"], [ALPHA_CONCLUSION])]),
    dict(id="T14", category="paper_fact", steps=[step("Synthetic Beta论文的主要结论是什么？", ["beta"], [BETA_CONCLUSION])]),
    dict(id="T15", category="comparison", steps=[step("这两篇论文各自的研究重点和侧重点区别是什么？", ["alpha", "beta"], [ALPHA, BETA], both_sources=True)]),
    dict(id="T16", category="comparison", steps=[step("分别列出这两篇论文的主要结论。", ["alpha", "beta"], [ALPHA_CONCLUSION, BETA_CONCLUSION])]),
    dict(id="T17", category="followup", steps=[
        step("课程报告提交格式是什么？", ["lab"], [fact("lab", r"PDF")]),
        step("那截止时间呢？", ["lab"], [fact("lab", r"6\s*月\s*30\s*日|06[-/]30|6[-/]30", r"18[:：]00")])]),
    dict(id="T18", category="history_return", steps=[
        step("这两篇论文的研究重点区别是什么？", ["alpha", "beta"], [ALPHA, BETA], both_sources=True),
        step("图书馆初始借阅期限是多少天？", ["library"], [fact("library", r"(?:30|三十)\s*天")]),
        step("前面Synthetic Alpha和Synthetic Beta那两篇论文，分别列出主要结论。", ["library", "alpha", "beta"], [ALPHA_CONCLUSION, BETA_CONCLUSION])]),
    dict(id="N01", category="missing_evidence", steps=[step("仅根据实验室文档，指纹考勤机的设备品牌是什么？", ["lab"], refusal=True)]),
    dict(id="N02", category="missing_evidence", steps=[step("仅根据Synthetic Alpha论文，其实验样本数是多少？", ["alpha"], refusal=True)]),
]

PROBES = ("T01", "T03", "T05", "T06", "T07", "T08", "T11", "T13")

# 固定16轮完整历史；是正常长度的学习讨论，不用重复字符人为放大收益。
HISTORY = [
    ("我想做一份文献阅读计划，第一步怎么安排？", "先确认研究问题，再按主题整理资料。每篇记录问题、方法、证据和局限；阅读时保存页码与原文，不将自己的猜测当成作者结论。第一遍看摘要、引言、结论，第二遍再检查推导与实验，最后写一段能回到证据的总结。"),
    ("阅读笔记应该有哪些栏目？", "可包含资料名称、阅读日期、研究问题、核心方法、主要结论、实验条件、尚未理解的点和原文位置。把作者的论断、自己的解释与待核实问题分栏保存。记录原文定位可以帮助后续复查，避免只留下脱离上下文的一句话。"),
    ("如何比较两篇研究方法不同的文章？", "先对齐比较维度，包括目标、假设、输入输出、约束条件、计算成本和实验设置。分别陈述各篇有原文支持的内容，再归纳异同。不同数据集上的数值不宜直接判断优劣；无法对齐的条件应写进限制说明，而不是省略。"),
    ("我准备每晚读两小时，怎么分配？", "可以把前半小时用于回顾问题与旧笔记，中间一小时精读一段关键内容，最后半小时整理结论与疑问。若某个公式暂时无法推导，先标记依赖条件。安排应跟随理解进度调整，避免只用读过多少页评价学习效果。"),
    ("怎样处理不认识的专业术语？", "先从当前段落判断术语所属的学科与任务，再查看文章给出的定义、符号表和引用文献。记录这个术语在本篇中的含义，必要时补充一般定义。一般解释可以帮助理解，但不能反过来证明文章确实采用了某种具体方法。"),
    ("我想把笔记按章节编号。", "章节号可以帮助整理内容，但它与引用编号承担不同用途。章节号表示笔记结构，引用编号用于找到证据。若文章版本变化，应检查页码和段落是否移动；保留资料版本，避免把旧版本的定位用于新版文本。"),
    ("实验表格里应关注什么？", "除了结果数值，还要记录数据来源、样本数量、划分方式、评价指标和对照组。对比改善时保持条件一致，并说明变化来自哪一项调整。若表格没有报告运行时间，就把这一项标为未报告，不能根据方法名称猜测速度。"),
    ("什么样的疑问值得加入待办？", "优先记录影响主要结论的问题，例如关键假设是否成立、证据是否完整、比较是否公平以及结果能否复现。为每个待办写上相关页码和下一步动作；已经解决的问题保留处理记录，防止后面重新把同一疑问当成新结论。"),
    ("我希望每周复盘一次。", "每周可检查已读材料、已确认结论、仍有冲突的解释与下一周计划。复盘时回到原文抽查几条笔记，特别关注自己曾经修改的判断。把计划变更与文档事实分开记录，计划延期不会改变原文写明的时间要求。"),
    ("如何安排复现实验的步骤？", "先保存代码与数据版本，再检查环境和依赖，随后用一个小样本跑通流程。记录配置、随机种子和运行日志，最后按固定评测集复核指标。若只验证了流程没有检查结果，就应写成运行通过，不能直接等同于实验结论得到复现。"),
    ("读完以后要写多长的总结？", "总结的长度应服务于问题，而不是固定字数。可以先写一段研究问题与主要结论，再补方法、证据和适用边界。保留关键术语与原文定位；对于文章没有覆盖的内容，明确写出未知项，避免为了段落完整而补出未经证实的信息。"),
    ("我需要一个可复用的阅读检查表。", "检查表可包含目标是否明确、方法是否可解释、结论是否有证据、对照是否公平、局限是否说明、引用能否回溯。每次阅读只填写实际找到的内容。暂缺证据的条目保留为空并列入待办，不使用通用常识替代指定文章中的论断。"),
    ("更正一下，我的个人复盘日从周五改成周日。", "已记录个人复盘日改为周日。这个变更只涉及你的学习安排，不修改任何文档制度。后续谈个人安排时按周日理解；若询问文档中的截止日期，则需要重新读取对应原文，不能从个人计划推断文档规定。"),
    ("下次回答时请简洁些，但保留证据。", "可以先直接回答当前问题，再附必要的原文引用与边界说明。多个方面分别回答，避免重复背景。如果你问到具体文件，我会以该文件原文为依据；历史讨论仅帮助理解问题，不能独立充当文件内容的证明。"),
    ("还要记住我的复盘日已经改过。", "当前个人复盘日是周日，之前的周五安排已被更正。原始讨论仍可保留用于回溯，后续应采用最新明确纠正。这个偏好不影响资料库中课程或实验室的规则，讨论不同对象时需要保持来源与适用范围清楚。"),
    ("接下来我会针对资料继续提问。", "可以直接提出文档问题，也可以指定要比较的资料与维度。若问题承接历史，我会结合当前对象理解；若证据不足，应指出缺的是哪个方面。对于明确的文档事实，回答需要给出可回溯的原文依据，并说明未能确认的内容。"),
]
