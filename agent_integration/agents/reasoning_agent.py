# agents/reasoning_agent.py
import os, json, random, uuid, re
import dspy
from typing import List, Optional
# [MOD] 你原有的工具；确保存在
from utils.text_utils import trim_text_to_tokens


# [NEW] 轨迹日志（可选，但强烈推荐开启，后续 SFT/RL 必备）
from utils.trajectory_logger import TrajectoryLogger

# 本地 OpenAI 兼容端点 → 指向你上面开的 llama.cpp 服务器
# os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"
# os.environ["OPENAI_API_KEY"]  = "EMPTY"  # llama.cpp 不校验但需要占位

_OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
_OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY_REAL", "sk-fake"))
_REASON_LLM_MODEL = os.getenv("REASON_LLM_MODEL", "gpt-3.5-turbo")
_REASON_TIMEOUT   = float(os.getenv("REASON_TIMEOUT", "60"))
_REASON_MAXTOK    = int(os.getenv("REASON_MAX_TOKENS", "384"))

dspy.configure(
    lm=dspy.LM(
        model=_REASON_LLM_MODEL,
        api_base=_OPENAI_API_BASE,
        api_key=_OPENAI_API_KEY,
        max_tokens=_REASON_MAXTOK,
        top_p=1.0,
        temperature=0.0,
        timeout=_REASON_TIMEOUT,
    )
)
# ============================================================================

# ======================= [ADD] 严格双输出 Signature（方案 A） =======================
class StrictRRSignature(dspy.Signature):
    """
    先思考后作答：在内部逐步推理，但**只输出结构化字段**，用于稳定解析与评测。
    字段含义：
      - reasoning：用 1–2 句简要说明关键依据（不要展开长思维链）
      - response ：最终产出文本（此处用于“检索查询”或改写结果）
    注意：不要输出与字段无关的多余文字。
    """
    context  = dspy.InputField(desc="检索到的上下文文本，可为空")
    question = dspy.InputField(desc="用户问题或待改写查询")
    reasoning = dspy.OutputField(desc="简要理由（1–2 句）")
    response  = dspy.OutputField(desc="最终产出文本")
# ============================================================================


def load_dataset(json_path="data-hotpot/hotpot_mini_corpus.json",
                 train_ratio=0.7, val_ratio=0.2):
    """加载 Hotpot mini corpus (json) 并拆分 train/val/test"""
    with open(json_path, 'r') as f:
        corpus = json.load(f)

    if len(corpus) < 10:
        raise ValueError("⚠️ Data is too small, cannot train MIPROv2! Please increase data!")

    examples = []
    for item in corpus:
        context_text = item['context']
        question = item['question']
        answer = item['answer']
        examples.append(
            dspy.Example(context=context_text, question=question, response=answer).with_inputs("context", "question")
        )

    random.shuffle(examples)
    total = len(examples)
    tr = int(total * train_ratio)
    vr = int(total * val_ratio)
    trainset, valset, testset = examples[:tr], examples[tr:tr+vr], examples[tr+vr:]

    if not trainset or not valset:
        raise ValueError("⚠️ trainset or valset is empty, cannot run MIPROv2, please check data!")
    return trainset, valset, testset


class ReasoningAgent:
    """
    [MOD] 作用：面向 Retrieval 的“查询重写器”，不直接产最终答案。
    输出：refined_query（用于 dense retrieval），以及 fallback 标记。
    """

    def __init__(self,
                 logger: Optional[TrajectoryLogger] = None,
                 dataset_path: str = "data-hotpot/hotpot_mini_corpus.json",
                 compile_on_init: bool = False):
        import os, random
        self.logger = logger

        # --- 轻量化开关：环境变量控制 ---
        LIGHT_MODE = os.getenv("LIGHT_MODE", "0") == "1"

        # [MOD] DSPy 的推理链
        # 如果后续想强制 JSON 两字段，可改成：
        # self.chain = dspy.Predict("context, question -> reasoning, response")
        #self.chain = dspy.ChainOfThought("context, question -> response")
        # ======================= [CHANGE] 使用双输出 Predict(StrictRRSignature) =======================
        # 原：self.chain = dspy.ChainOfThought("context, question -> response")
        self.chain = dspy.Predict(StrictRRSignature)
        # =====================================================================

        # [MOD] MIPROv2 优化器（轻量化参数）
        if LIGHT_MODE:
            # 极简自举：只跑 1 轮、1 候选、少量样本，避免长时间重试
            # self.optimizer = dspy.MIPROv2(
            #     metric=dspy.evaluate.SemanticF1(),
            #     auto="off",                # 关闭自动搜索，进一步降耗（也可用 "medium"）
            #     num_rounds=1,
            #     num_candidates=1,
            #     bootstrap_samples=5,       # 关键：自举样本量小
            #     max_bootstrapped_traces=10,
            #     patience=0,                # 不做额外探索
            #     num_threads=int(os.getenv("DSPY_THREADS", "2")),
            # )
            self.optimizer = None  # 轻量：不做自举优化，直接用基础链路
        else:
            # 原配置（可按你原值放大）
            self.optimizer = dspy.MIPROv2(
                metric=dspy.evaluate.SemanticF1(),
                auto="medium",
                #num_threads=int(os.getenv("DSPY_THREADS", "4")),
                # 如需在非轻量模式下指定更大的规模，可加：
                # num_rounds=3, num_candidates=3, bootstrap_samples=50, ...
            )

        # [MOD] 数据集加载
        full_train, full_val, full_test = load_dataset(
            json_path=dataset_path, val_ratio=0.1
        )

        # 轻量模式下，裁剪训练/验证集到很小规模，优先跑通
        if LIGHT_MODE:
            def _small(xs, k):
                xs = list(xs)
                random.shuffle(xs)
                return xs[:k]
            self.trainset = _small(full_train, 5)   # 5 条足以验证链路
            self.valset   = _small(full_val,   5)
            self.testset  = _small(full_test,  5)
        else:
            self.trainset, self.valset, self.testset = full_train, full_val, full_test

        # [MOD] 是否在初始化就编译
        if compile_on_init:
            try:
                if self.optimizer is None:
                    # 轻量：不编译，直接使用原始链路（Predict/StrictRRSignature）
                    self.optimized_agent = self.chain
                else:
                    # 非轻量：做一次最小化编译
                    self.optimized_agent = self.optimizer.compile(
                        self.chain,
                        trainset=self.trainset,
                        valset=self.valset,
                        requires_permission_to_run=False
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log_event("compile_failed", {"error": str(e)})
                self.optimized_agent = self.chain
        else:
            self.optimized_agent = self.chain

    # ===== 工具函数 =====
    def _should_fallback(self, retrieved_context: str) -> bool:
        """检索文本过短或疑似泛化背景 → fallback（中英兼容）"""
        text = retrieved_context or ""
        en_words = len([w for w in text.split() if re.search(r"[A-Za-z]", w)])
        zh_chars = len(re.findall(r"[\u4E00-\u9FFF]", text))
        if en_words < 50 and zh_chars < 80:
            return True
        keywords = ["overview", "history", "general", "unrelated", "background", "miscellaneous"]
        return any(kw.lower() in text.lower() for kw in keywords)


    def _fewshot_examples(self) -> str:
        """[MOD] 输出与目标保持一致：产出‘检索查询’（关键词式）"""
        return (
            "Example 1:\n"
            "Question: What are the main types of cloud computing?\n"
            "Retrieved docs: general cloud benefits, minor SaaS mention.\n"
            "Refined query: cloud computing service categories IaaS PaaS SaaS comparison\n\n"
            "Example 2:\n"
            "Question: How does photosynthesis work in plants?\n"
            "Retrieved docs: general plant biology, chlorophyll mention.\n"
            "Refined query: stages of photosynthesis light dependent independent reactions overview\n\n"
        )

    def _instruction_prefix(self) -> str:
        """[MOD] 强规范的 Instruction，强调‘高精准检索查询’"""
        return (
            "Given the retrieved context (may be empty) and the user question:\n"
            "1) Diagnose if the retrieved context is noisy or off-topic.\n"
            "2) Rewrite ONE concise search query focusing on precision without losing key recall.\n"
            "3) Use keywords, entities, and constraints (dates, names, aliases) if helpful.\n"
            "4) Output exactly ONE query line; DO NOT explain.\n\n"
        )

    # ===== 主入口：产出 refined_query =====
    def plan(self, user_question: str, retrieved_docs: Optional[List] = None):
        """
        统一模板：只产出“一行检索查询”，并返回 fallback 标记。
        - 稳健处理 retrieved_docs（混合类型）；
        - DSPy 推理失败有兜底（从问题中抽关键词）；
        - 严格后处理成一行、≤20 词、去说明性前缀与噪声符号；
        - 记录检索命中（只写元数据，不落正文）。
        """
        import re

        # [NEW] 轨迹：起始打点
        if self.logger:
            self.logger.add_reason(f"[reason.plan.start] q={user_question}")

        instruction = self._instruction_prefix()
        few_shot = self._fewshot_examples()

        # 1) 准备检索上下文（可为空，混合类型安全处理）
        retrieved_context = ""
        fallback = False
        if retrieved_docs:
            parts = []
            for doc in retrieved_docs:
                txt = getattr(doc, "page_content", None)
                if txt is None:
                    txt = str(doc)
                txt = (txt or "").strip()
                if txt:
                    parts.append(txt)
            raw_ctx = "\n".join(parts)
            if self._should_fallback(raw_ctx):
                fallback = True
            else:
                retrieved_context = trim_text_to_tokens(raw_ctx, max_tokens=1200)
        else:
            fallback = True

        # 2) 统一上下文块（避免提示不一致）
        ctx_block = f"{instruction}{few_shot}====\n"
        ctx_block += "[RETRIEVED_SNIPPETS]\n" + (retrieved_context if retrieved_context else "<NONE>") + "\n"

        # 3) 强约束：明确只需要“一条检索查询”
        constrained_question = (
            f"{user_question}\n\n"
            "Return ONE rewritten search query only, concise keywords + entities + constraints; DO NOT explain."
        )

        # 4) 记录输入（不落原文，避免泄漏）
        if self.logger:
            try:
                self.logger.add_reason("[reason.plan.ctx_ready]")
                if retrieved_docs:
                    K = 5  # 限制记录前 K 条，避免日志过大
                    hits_meta = []
                    for i, d in enumerate(retrieved_docs[:K]):
                        meta = getattr(d, "metadata", None)
                        if not isinstance(meta, dict):
                            meta = {}
                        doc_id = meta.get("id") or meta.get("_id") or meta.get("doc_id") or f"doc{i}"
                        score = meta.get("score", getattr(d, "score", None))
                        source = meta.get("source") or meta.get("dataset") or meta.get("collection")
                        url = meta.get("url") or meta.get("href")
                        hits_meta.append({
                            "doc_id": doc_id,
                            "score": float(score) if isinstance(score, (int, float)) else None,
                            "source": source,
                            "url": url
                        })
                    self.logger.add_tool_call(
                        type="retrieval_input",
                        query="[FROM_PREV]",     # 不落用户原文
                        topk=len(retrieved_docs),
                        hits=hits_meta
                    )
            except Exception as e:
                # 日志失败不影响主流程
                self.logger.add_reason(f"[reason.plan.log_error] {type(e).__name__}: {e}")

        # 5) 调用 DSPy（带兜底）
        try:
            out = self.optimized_agent(context=ctx_block, question=constrained_question)
            raw_response = (getattr(out, "response", "") or str(out)).strip()
            brief_reason = getattr(out, "reasoning", "")
            if self.logger and brief_reason:
                self.logger.add_reason(f"[reason.short] {brief_reason}")
        except Exception as e:
            if self.logger:
                self.logger.add_reason(f"[reason.err] {type(e).__name__}: {e}")
            raw_response = ""

        # 6) 兜底：若模型未返回可用内容，从用户问题抽关键词（≤20 词）
        if not raw_response:
            h = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF_\-:/\.,\"\'()\[\] ]+', ' ', user_question or "")
            h = re.sub(r"\s+", " ", h).strip()
            tokens = h.split()[:20]
            raw_response = " ".join(tokens) or (user_question or "general query")

        # 7) “一行检索查询”后处理
        first_line = raw_response.splitlines()[0].strip()

        # 去常见说明性前缀
        prefixes = (
            "refined query:", "search query:", "query:", "final query:",
            "the relevant search query is:"
        )
        low = first_line.lower()
        for p in prefixes:
            if low.startswith(p):
                first_line = first_line[len(p):].strip()
                break

        # 去包裹引号/反引号
        if (first_line.startswith(("'", '"', "`")) and first_line.endswith(("'", '"', "`"))) and len(first_line) >= 2:
            first_line = first_line[1:-1].strip()

        # 基础字符清洗
        cleaned = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF_\-:/\.,\"\'()\[\] ]+', ' ', first_line)
        # 去列表前缀，如 "- "、"1. " 等
        cleaned = re.sub(r'^\s*[-*\d]+\s*[\.\)]\s*', '', cleaned)
        # 合并空白
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 限制最多 20 词
        tokens = cleaned.split()
        if len(tokens) > 20:
            cleaned = " ".join(tokens[:20])

        # 避免句号/感叹号/问号/分号/冒号收尾（更像“查询”）
        cleaned = cleaned.rstrip(".!?;:")

        refined_query = cleaned if cleaned else first_line

        # 8) 记录输出
        if self.logger:
            self.logger.add_reason(f"[reason.refined_query] {refined_query}")

        return {"refined_query": refined_query, "fallback": fallback}
