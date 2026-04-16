from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    ContextPrecision,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    NoiseSensitivity
)
from ragas.llms import LangchainLLMWrapper

import os
import traceback
try:
    import numpy as _np
except Exception:
    _np = None  # 没装 numpy 也要定义，避免 NameError
import re
import numbers
import math



class EvaluationAgent:
    """
    Multi-functional evaluation Agent:
      1) quick_evaluate(...) — 判断检索是否足以回答问题
      2) evaluate_retrieval(...) — 仅评估检索质量（Precision/Recall）
      3) evaluate_generation(...) — 仅评估生成质量（一次 evaluate 跑多指标）
      4) full_evaluate(...) — 兼容式一次性评估（检索+faithfulness）
    """

    # ======================= 工具函数（保留必要的两枚） =======================

    @staticmethod
    def _get_numeric_value(value):
        """抽取数值（支持 numpy / list 均值 / 对象.value）"""
        if value is None:
            return 0.0
        if isinstance(value, numbers.Real):
            return float(value)
        # 2) numpy（可选）：标量/数组
        if _np is not None:
            if isinstance(value, (_np.floating, _np.integer)):
                return float(value)
            if isinstance(value, _np.ndarray):
                try:
                    return float(value.mean())
                except Exception:
                    pass

        # 3) 序列：list/tuple -> 均值
        if isinstance(value, (list, tuple)):
            nums = []
            for v in value:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return (sum(nums) / len(nums)) if nums else 0.0

        # 4) 兜底：float(...) 或对象.value
        try:
            return float(value)
        except (TypeError, ValueError):
            if hasattr(value, "value") and isinstance(getattr(value, "value"), numbers.Real):
                return float(getattr(value, "value"))
            return 0.0



    @staticmethod
    def _extract_score(result, metric_name):
        """
        兼容不同 ragas 结构：dict / .scores / .data / list(item.name, item.score)
        """
        # dict
        if isinstance(result, dict) and metric_name in result:
            return EvaluationAgent._get_numeric_value(result[metric_name])

        # 可下标（但不把 str 当成可下标容器）
        if hasattr(result, "__getitem__") and not isinstance(result, str):
            try:
                return EvaluationAgent._get_numeric_value(result[metric_name])
            except (KeyError, TypeError, IndexError):
                pass

        # .scores
        if hasattr(result, "scores"):
            scores = getattr(result, "scores")
            if isinstance(scores, dict) and metric_name in scores:
                return EvaluationAgent._get_numeric_value(scores[metric_name])

        # .data
        if hasattr(result, "data") and isinstance(getattr(result, "data"), dict):
            data = getattr(result, "data")
            if metric_name in data:
                return EvaluationAgent._get_numeric_value(data[metric_name])

        # list[MetricResult] / 可迭代（排除 str/dict）
        if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
            try:
                for item in result:
                    name = getattr(item, "name", None) or getattr(item, "metric", None)
                    if name is None and hasattr(item, "metric") and hasattr(item.metric, "name"):
                        name = getattr(item.metric, "name")
                    if name and str(name) == str(metric_name):
                        val = getattr(item, "score", None)
                        if val is None and hasattr(item, "value"):
                            val = getattr(item, "value")
                        return EvaluationAgent._get_numeric_value(val)
            except Exception:
                pass

        # 未命中
        return 0.0



    @staticmethod
    def _num_with_status(x, default=0.0):
        """基于你已有 _get_numeric_value，补上 none/nan/error 的状态语义。"""
        try:
            if x is None:
                return default, "none"

            # list/tuple/numpy.ndarray：均值
            if isinstance(x, (list, tuple)) or (_np is not None and isinstance(x, _np.ndarray)):
                nums = []
                seq = x.tolist() if (_np is not None and hasattr(x, "tolist")) else x
                for v in seq:
                    try:
                        nums.append(float(v))
                    except Exception:
                        pass
                if not nums:
                    return default, "error"
                f = sum(nums) / len(nums)
                return (default, "nan") if math.isnan(f) else (f, "ok")

            # 其余直接走 _get_numeric_value
            f = float(EvaluationAgent._get_numeric_value(x))
            return (default, "nan") if math.isnan(f) else (f, "ok")
        except Exception:
            return default, "error"


    @staticmethod
    def _extract_score2(result, metric_names):
        """
        在你已有 _extract_score 的基础上，新增：
        - 同时尝试多个候选键（如 ['response_relevancy','answer_relevancy']）
        - 兼容 list[MetricResult]（name/metric.name）
        - repr 兜底，支持负号/科学计数法/跨行
        返回: (found: bool, value: float)
        """
        names = metric_names if isinstance(metric_names, (list, tuple)) else [metric_names]
        names = [str(n) for n in names]

        # 1) 先复用 _extract_score（逐名尝试）
        for nm in names:
            try:
                val = EvaluationAgent._extract_score(result, nm)
                # _extract_score 找不到即 0.0；此处只有非 0 才直接返回
                if val != 0.0:
                    return True, float(val)
            except Exception:
                pass

        # 2) 再加一遍：专门处理 list[MetricResult]（更稳健）
        if isinstance(result, (list, tuple)):
            for item in result:
                name = getattr(item, "name", None) or getattr(item, "metric", None)
                if name is None and hasattr(item, "metric") and hasattr(item.metric, "name"):
                    name = getattr(item.metric, "name")
                if name and str(name) in names:
                    val = getattr(item, "score", None)
                    if val is None and hasattr(item, "value"):
                        val = getattr(item, "value")
                    return True, EvaluationAgent._get_numeric_value(val)

        # 3) repr 兜底（-1.23 / 1e-3 / 跨行；含 noise_sensitivity(...)）
        rep = repr(result)
        num_pat = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        for nm in names:
            # 精确键名
            pat_exact = rf"['\"]{re.escape(str(nm))}['\"]\s*:\s*{num_pat}"
            m = re.search(pat_exact, rep, flags=re.S)
            if m:
                try:
                    return True, float(m.group(1))
                except Exception:
                    pass
            # 宽松前缀：noise_sensitivity(...) 变体
            if "noise_sensitivity" in str(nm):
                pat_ns = rf"['\"](noise_sensitivity[^'\"]*)['\"]\s*:\s*{num_pat}"
                m2 = re.search(pat_ns, rep, flags=re.S)
                if m2:
                    try:
                        return True, float(m2.group(2))
                    except Exception:
                        pass

        return False, 0.0

    # ======================= 初始化（云端优先，兼容本地） =======================

    def __init__(self, model_name="gpt-3.5-turbo", embeddings=None, llm=None):
        # --- LLM：优先真实云端 KEY，兜底本地 ---
        if llm is not None:
            self.llm = llm
        else:
            real_key = os.getenv("OPENAI_API_KEY_REAL")
            if real_key:
                self.llm = ChatOpenAI(
                    model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
                    base_url="https://api.openai.com/v1",
                    api_key=real_key,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "512")),
                    timeout=float(os.getenv("EVAL_TIMEOUT", "90")),
                    max_retries=int(os.getenv("EVAL_MAX_RETRIES", "0")),
                )
            else:
                self.llm = ChatOpenAI(
                    model=model_name,
                    base_url=os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
                    api_key=os.getenv("OPENAI_API_KEY", "sk-fake"),
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "512")),
                    timeout=float(os.getenv("EVAL_TIMEOUT", "90")),
                    max_retries=int(os.getenv("EVAL_MAX_RETRIES", "0")),
                )

        # --- Embeddings：用于 ResponseRelevancy，建议走官方 ---
        if embeddings is None:
            try:
                print("📦 Initializing OpenAI embeddings for ResponseRelevancy")
                self.embeddings = OpenAIEmbeddings(
                    model=os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small"),
                    api_key=os.environ.get("OPENAI_API_KEY_REAL") or os.environ.get("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1",
                )
                emb_model = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None)
                print(f"🔎 Ragas/ResponseRelevancy embeddings ready: {emb_model}")

            except Exception as e:
                print(f"⚠️ Could not initialize embeddings: {str(e)}")
                print("⚠️ ResponseRelevancy will be skipped")
                self.embeddings = None
        else:
            self.embeddings = embeddings

        # --- 打印 LLM 的模型名和 base_url，更直观 ---
        try:
            name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None)
            base = getattr(self.llm, "base_url", None)
            if base is None and hasattr(self.llm, "client"):
                base = getattr(self.llm.client, "base_url", None)
            print(f"[EvaluationAgent] llm={name} base={base}")
        except Exception:
            print("[EvaluationAgent] llm ready (could not introspect fields)")


        # full_evaluate 默认指标
        self.metrics = [ContextPrecision(), LLMContextRecall(), Faithfulness()]

    # ============================== 1) quick_evaluate ==============================

    def quick_evaluate(self, question, docs):
        """
        仅用 LLM 粗评：检索是否足以回答问题 + 给出关键词建议
        """
        if not docs:
            return {"sufficient": False, "suggested_keywords": "expand keywords"}

        parts = []
        for d in docs:
            txt = getattr(d, "page_content", "") or ""
            if txt.strip():
                parts.append(txt)
        context = " ".join(parts)

        prompt = (
            f"Question: {question}\n"
            f"Retrieved content: {context[:1000]}...\n"
            "Is this information sufficient to answer the question? "
            "Please respond with 'sufficient' or 'insufficient' and provide additional keywords."
        )
        msg = self.llm.invoke(prompt)
        text = getattr(msg, "content", str(msg)) or ""
        sufficient = "sufficient" in text.lower()
        suggested_keywords = " ".join(text.split()[-3:]) if text else ""

        return {"sufficient": sufficient, "suggested_keywords": suggested_keywords}

    # ============================== 2) evaluate_retrieval ==============================

    def evaluate_retrieval(self, user_query, retrieved_docs, reference=None):
        """
        专用于 RetrieverAgent：只评估 ContextPrecision / LLMContextRecall
        —— 按教程键名：user_input / retrieved_contexts / response / reference

        注意：当没有 reference 时，跳过需要 reference 的指标（如 ContextPrecision），
        只返回基础检索信息。
        """
        contexts = [
            getattr(doc, "page_content", "")
            for doc in retrieved_docs
            if (getattr(doc, "page_content", "") or "").strip()
        ]
        if not contexts:
            contexts = ["N/A"]

        has_reference = bool(reference and str(reference).strip())

        # 如果没有 reference，直接返回基础信息，不做 ragas 评估
        # 因为 ContextPrecision 等指标需要 reference
        if not has_reference:
            print(f"🔎 [Retrieval] No reference provided, skipping ragas evaluation")
            print(f"🔎 [Retrieval] Retrieved {len(contexts)} contexts")
            return {
                "context_precision": 0.5,  # 默认中等分数
                "context_recall": 0.5,     # 默认中等分数
                "doc_count": len(contexts),
            }

        record = {
            "user_input": user_query,
            "retrieved_contexts": contexts,
            "response": "N/A",
            "reference": reference,
        }
        dataset = EvaluationDataset.from_list([record])

        # ---- 指标选择 ----
        metrics = [ContextPrecision(), LLMContextRecall()]

        try:
            result = evaluate(dataset=dataset, metrics=metrics, llm=LangchainLLMWrapper(self.llm))
            print("\n🔎 Retrieval Eval Raw ➝", result)

            # 读取分数（复用你的工具函数）
            scores = getattr(result, "scores", None)
            if isinstance(scores, dict):
                context_precision = self._get_numeric_value(scores.get("context_precision", 0))
                context_recall = self._get_numeric_value(scores.get("context_recall", 0))
            else:
                context_precision = self._get_numeric_value(self._extract_score(result, "context_precision"))
                context_recall = self._get_numeric_value(self._extract_score(result, "context_recall"))

            # 打印
            print(f"🎯 Context Precision: {context_precision:.4f}")
            print(f"📈 Context Recall: {context_recall:.4f}")

        except Exception as e:
            print(f"❌ Error in evaluate_retrieval: {str(e)}")
            traceback.print_exc()
            context_precision = 0.5
            context_recall = 0.5

        return {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "doc_count": len(contexts),
        }

    # ============================== 3) evaluate_generation ==============================

    def evaluate_generation(self, user_query, retrieved_docs, response, reference=None):
        """
        单次 ragas.evaluate 同时跑 Faithfulness + (可选)ResponseRelevancy + NoiseSensitivity，
        并返回每个指标的分数 + status（ok/none/nan/error/missing/disabled）。
        兼容 Ragas 不同版本的返回：dict / EvaluationResult.scores / list[MetricResult] / 仅 repr 可读。
        """
        # ---- 准备数据（做长度保护）----
        contexts = [
            getattr(doc, "page_content", "") for doc in (retrieved_docs or [])
            if (getattr(doc, "page_content", "") or "").strip()
        ]
        if not contexts:
            contexts = ["N/A"]
        else:
            contexts = [c[:1500] for c in contexts[:2]]

        resp = (str(response) if response else "N/A")[:3000]
        data = {
            "user_input": user_query,
            "retrieved_contexts": contexts,
            "response": resp,
            "reference": reference if (reference and str(reference).strip()) else None,
        }
        dataset = EvaluationDataset.from_list([data])

        lc = LangchainLLMWrapper(self.llm)

        metrics = [Faithfulness()]
        if getattr(self, "embeddings", None) is not None:
            metrics.append(ResponseRelevancy(embeddings=self.embeddings, llm=lc))
        metrics.append(NoiseSensitivity(llm=lc))

        # 默认分数与status（区分缺失 vs 真0）
        out = {
            "faithfulness": 0.0, "faithfulness_status": "missing",
            "response_relevancy": 0.0,
            "response_relevancy_status": ("disabled" if getattr(self, "embeddings", None) is None else "missing"),
            "noise_sensitivity": 1.0, "noise_sensitivity_status": "missing",
            "raw": None,
        }

        try:
            # —— 一次 evaluate ——
            result = evaluate(dataset=dataset, metrics=metrics, llm=lc)
            out["raw"] = result
            print("\n🔎 Gen Eval Raw ➝", type(result), repr(result))

            # 取一个“可解析体”
            scores_obj = getattr(result, "scores", None) or result

            # 统一抓分：优先用你新增的 _extract_score2；拿到值后用 _num_with_status 赋状态
            def grab(keys, default):
                found, val = type(self)._extract_score2(scores_obj, keys)
                if found:
                    return type(self)._num_with_status(val, default)
                else:
                    return default, "missing"

            # faithfulness
            f_val, f_st = grab(["faithfulness"], 0.0)

            # response_relevancy / answer_relevancy
            if getattr(self, "embeddings", None) is not None:
                r_val, r_st = grab(["response_relevancy", "answer_relevancy"], 0.0)
            else:
                r_val, r_st = 0.0, "disabled"

            # noise_sensitivity（名字可能带括号）
            n_val, n_st = grab(
                ["noise_sensitivity", "noise_sensitivity(mode=relevant)", "noise_sensitivity(relevant)"],
                1.0
            )

            out.update({
                "faithfulness": f_val, "faithfulness_status": f_st,
                "response_relevancy": r_val, "response_relevancy_status": r_st,
                "noise_sensitivity": n_val, "noise_sensitivity_status": n_st,
            })

            out["answer_relevancy"] = out["response_relevancy"]
            out["answer_relevancy_status"] = out["response_relevancy_status"]

            # 友好打印
            if isinstance(scores_obj, dict):
                keys_view = list(scores_obj.keys())
            elif hasattr(scores_obj, "scores") and isinstance(scores_obj.scores, dict):
                keys_view = list(scores_obj.scores.keys())
            else:
                keys_view = f"[no dict-like keys; parsed from repr: {repr(result)[:120]}...]"

            print(f"🔑 Parsed score keys (or repr hint): {keys_view}")
            print(
                f"✅ Faith={out['faithfulness']:.4f}({out['faithfulness_status']}), "
                f"Rel={out['response_relevancy']:.4f}({out['response_relevancy_status']}), "
                f"Noise={out['noise_sensitivity']:.4f}({out['noise_sensitivity_status']})"
            )

        except Exception as e:
            print(f"❌ evaluate_generation failed: {e}")

        return out



    # ============================== 4) full_evaluate ==============================

    def full_evaluate(self, query, retrieved_docs, response=None, reference=None):
        """
        一次性评估（与旧逻辑兼容）：
        - Retrieval: ContextPrecision / LLMContextRecall
        - Generation: Faithfulness（若有 reference）
        —— 仍按教程键名构造记录
        """
        contexts = [
            getattr(doc, "page_content", "")
            for doc in retrieved_docs
            if (getattr(doc, "page_content", "") or "").strip()
        ] or ["N/A"]

        resp_text = str(response) if response is not None else "N/A"

        record = {
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": resp_text,
            "reference": reference if (reference and str(reference).strip()) else None,
        }


        dataset = EvaluationDataset.from_list([record])

        # 动态 metrics：无 reference 不评 Faithfulness
        metrics = [ContextPrecision(), LLMContextRecall()]
        if record["reference"] is not None:
            metrics.append(Faithfulness())

        try:
            result = evaluate(dataset=dataset, metrics=metrics, llm=LangchainLLMWrapper(self.llm))
            print("\n🔎 Full Eval Raw ➝", result)

            scores = getattr(result, "scores", None)
            if not scores:
                print("❌ Scores object is empty or None")
                return {"faithfulness": 0.0, "context_recall": 0.0, "context_precision": 0.0}

            faithfulness_score = self._get_numeric_value(scores.get("faithfulness", 0.0))
            context_recall = self._get_numeric_value(scores.get("context_recall", 0.0))
            context_precision = self._get_numeric_value(scores.get("context_precision", 0.0))

        except Exception as e:
            print(f"❌ Detailed error in evaluation: {str(e)}")
            traceback.print_exc()
            faithfulness_score = 0.0
            context_recall = 0.0
            context_precision = 0.0

        print(f"📊 Faithfulness: {faithfulness_score:.4f}")
        print(f"📈 Context Recall: {context_recall:.4f}")
        print(f"🎯 Context Precision: {context_precision:.4f}")

        return {
            "faithfulness": faithfulness_score,
            "context_recall": context_recall,
            "context_precision": context_precision
        }
