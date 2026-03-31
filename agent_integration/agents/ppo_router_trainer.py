"""
PPO Router Trainer
==================
Online PPO fine-tuning for the 2-action router (end / regenerate).
Initializes from router_policy_v3.pt (BC SFT stage).

Design:
  - Each question = one episode
  - State: 6 Ragas metrics after generation
  - Action: 0=end, 1=regenerate
  - Reward: terminal semF1 + step penalty (-0.02) per regenerate
  - Runs pipeline directly (bypasses LangGraph for step-level control)
  - PPO with clipped surrogate + value loss + entropy bonus

Usage:
  PYTHONPATH=. LIGHT_MODE=1 FAISS_PATH_OPENAI=... EMB_MODEL=text-embedding-3-large \
  python3 agents/ppo_router_trainer.py \
      --dataset data-hotpot/dev_real.jsonl \
      --init_policy agents/router_policy_v3.pt \
      --out_dir runs/ppo_router \
      --n_iter 20 --max_regen 2
"""

import os, sys, json, csv, time, argparse, functools, math
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── pipeline imports ──────────────────────────────────────────────────────────
from agents.RLRouterAgent import RouterPolicyNet, FEATURE_KEYS, _safe_float, _minmax_norm
from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.hybrid_retriever import HybridRetriever
from agents.reranker import create_cross_encoder_reranker
from agents.multi_query import generate_query_variants

# ── constants ─────────────────────────────────────────────────────────────────
ACTION2IDX = {"end": 0, "regenerate": 1}
IDX2ACTION = {0: "end", 1: "regenerate"}
STEP_PENALTY = -0.02   # per regenerate step (discourages infinite looping)


# ══════════════════════════════════════════════════════════════════════════════
# Value network (critic)
# ══════════════════════════════════════════════════════════════════════════════
class ValueNet(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_KEYS), hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (mirrors RLRouterAgent._featurize)
# ══════════════════════════════════════════════════════════════════════════════
def featurize(metrics: Dict[str, float],
              feat_min: Optional[torch.Tensor],
              feat_max: Optional[torch.Tensor]) -> torch.Tensor:
    vals = [
        _safe_float(metrics.get("context_precision", metrics.get("ctxP", 0.0))),
        _safe_float(metrics.get("context_recall",    metrics.get("ctxR",  0.0))),
        _safe_float(metrics.get("faithfulness_score", metrics.get("faith", 0.0))),
        _safe_float(metrics.get("response_relevancy", metrics.get("rel",   0.0))),
        _safe_float(metrics.get("noise_sensitivity",  metrics.get("noise", 1.0))),
        _safe_float(metrics.get("semantic_f1_score",  metrics.get("semantic_f1", 0.0))),
    ]
    x = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)  # [1, D]
    if feat_min is not None and feat_max is not None:
        x = _minmax_norm(x, feat_min, feat_max)
    else:
        x = torch.clamp(x, 0.0, 1.0)
    return x  # [1, D]


def diagnose_failure(gen_result: Dict[str, Any]) -> str:
    """Replicate feedback regenerate logic from langgraph_rag.py generator node."""
    faith = float(gen_result.get("faithfulness_score") or 0.0)
    noise = float(gen_result.get("noise_sensitivity")  or 1.0)
    rel   = float(gen_result.get("response_relevancy") or 0.0)
    if faith < 0.5:
        return ("Your previous answer contains claims not directly supported "
                "by the retrieved documents. Stick strictly to what the documents say.")
    elif noise > 0.7:
        return ("Your previous answer was influenced by irrelevant documents. "
                "Identify which documents are actually relevant to the question.")
    elif rel < 0.3:
        return ("Your previous answer does not directly address the question. "
                "Focus on answering exactly what was asked.")
    return ("Re-examine the context carefully. "
            "Make sure your answer is concise and factually grounded.")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline initialisation (shared across episodes)
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(vectorstore):
    evaluation_agent = EvaluationAgent()
    reasoning_agent  = ReasoningAgent()
    generation_agent = GenerationAgent()

    hybrid_retriever = HybridRetriever(vectorstore)
    reranker = create_cross_encoder_reranker(
        model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_n=int(os.getenv("RERANKER_TOP_N", "5")),
    )
    _mq_llm = getattr(generation_agent, "llm", None)
    multi_query_fn = (
        functools.partial(generate_query_variants, llm=_mq_llm, n_variants=2)
        if _mq_llm else None
    )
    retrieval_agent = RetrievalAgent(
        vectorstore, evaluation_agent, top_k=5,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        multi_query_fn=multi_query_fn,
    )
    return reasoning_agent, retrieval_agent, generation_agent, evaluation_agent


def get_vectorstore():
    faiss_dir = os.getenv("FAISS_PATH_OPENAI", "vectorstore-hotpot/hotpotqa_faiss_v3")
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    emb_model = os.getenv("EMB_MODEL", "text-embedding-3-large")
    vs = FAISS.load_local(faiss_dir, OpenAIEmbeddings(model=emb_model),
                          allow_dangerous_deserialization=True)
    print(f"✅ FAISS loaded: {faiss_dir} (emb={emb_model})")
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner (step-level control for PPO)
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(
    question: str,
    reference: Optional[str],
    reasoning_agent, retrieval_agent, generation_agent, evaluation_agent,
    policy: RouterPolicyNet,
    value_net: ValueNet,
    feat_min: Optional[torch.Tensor],
    feat_max: Optional[torch.Tensor],
    max_regen: int = 2,
    device: str = "cpu",
) -> Tuple[List, List, List, List, List, float]:
    """
    Returns: states, actions, log_probs, values, rewards, final_semf1
    Each list has length = number of router decisions taken.
    """
    # 1. Query optimisation
    try:
        reason_res = reasoning_agent.plan(question, retrieved_docs=None)
        refined_query = reason_res.get("refined_query", question)
    except Exception:
        refined_query = question

    # 2. Retrieval
    try:
        ret_res = retrieval_agent.retrieve(query=refined_query, top_k=5)
        docs = ret_res.get("docs", [])
    except Exception:
        docs = []

    # 3. Initial generation
    try:
        gen_res = generation_agent.answer(
            question=question, docs=docs,
            evaluation_agent=evaluation_agent,
            ground_truth=reference,
            force_answer=True,
        )
    except Exception as e:
        print(f"[ppo.episode] gen error: {e}")
        return [], [], [], [], [], 0.0

    states, actions, log_probs, values, rewards = [], [], [], [], []
    prev_answer = None

    policy.eval()
    value_net.eval()

    for step in range(max_regen + 1):  # +1 to allow final "end" decision
        metrics = {
            "context_precision":  float(gen_res.get("context_precision",  0.0) or 0.0),
            "context_recall":     float(gen_res.get("context_recall",     0.0) or 0.0),
            "faithfulness_score": float(gen_res.get("faithfulness_score", 0.0) or 0.0),
            "response_relevancy": float(gen_res.get("response_relevancy", 0.0) or 0.0),
            "noise_sensitivity":  float(gen_res.get("noise_sensitivity",  1.0) or 1.0),
            "semantic_f1_score":  float(gen_res.get("semantic_f1_score",  0.0) or 0.0),
        }

        x = featurize(metrics, feat_min, feat_max).to(device)  # [1, D]

        with torch.no_grad():
            logits = policy(x)           # [1, 2]
            dist   = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            lp     = dist.log_prob(action)
            val    = value_net(x).squeeze()

        states.append(metrics)
        actions.append(action.item())
        log_probs.append(lp)
        values.append(val)

        if action.item() == 0:  # end
            reward = float(metrics["semantic_f1_score"])
            rewards.append(reward)
            break
        else:  # regenerate
            rewards.append(STEP_PENALTY)
            if step == max_regen:
                # forced end — add terminal reward to last step
                rewards[-1] += float(metrics["semantic_f1_score"])
                break
            # Run generator with feedback
            hint = diagnose_failure(gen_res)
            prev_answer = gen_res.get("answer", "")
            try:
                gen_res = generation_agent.answer(
                    question=question, docs=docs,
                    evaluation_agent=evaluation_agent,
                    ground_truth=reference,
                    previous_answer=prev_answer,
                    failure_hint=hint,
                    force_answer=True,
                )
            except Exception as e:
                print(f"[ppo.episode] regen error: {e}")
                rewards[-1] += float(metrics["semantic_f1_score"])
                break

    final_semf1 = float(gen_res.get("semantic_f1_score", 0.0) or 0.0)
    return states, actions, log_probs, values, rewards, final_semf1


# ══════════════════════════════════════════════════════════════════════════════
# GAE advantage estimation
# ══════════════════════════════════════════════════════════════════════════════
def compute_returns_advantages(
    rewards: List[float],
    values: List[torch.Tensor],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    returns = torch.zeros(T)
    advantages = torch.zeros(T)
    gae = 0.0
    next_val = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_val - values[t].item()
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t].item()
        next_val = values[t].item()
    return returns, advantages


# ══════════════════════════════════════════════════════════════════════════════
# PPO update
# ══════════════════════════════════════════════════════════════════════════════
def ppo_update(
    policy: RouterPolicyNet,
    value_net: ValueNet,
    optimizer_policy: optim.Optimizer,
    optimizer_value: optim.Optimizer,
    all_states: List[Dict],
    all_actions: List[int],
    all_old_log_probs: List[torch.Tensor],
    all_returns: List[torch.Tensor],
    all_advantages: List[torch.Tensor],
    feat_min: Optional[torch.Tensor],
    feat_max: Optional[torch.Tensor],
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    entropy_coef: float = 0.01,
    device: str = "cpu",
):
    if not all_states:
        return {}

    # Build tensors from collected episodes
    xs = torch.cat([featurize(s, feat_min, feat_max) for s in all_states], dim=0).to(device)
    acts = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_lps = torch.stack(all_old_log_probs).to(device).detach()
    rets = torch.stack(all_returns).to(device).detach()
    advs = torch.stack(all_advantages).to(device).detach()
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    policy.train()
    value_net.train()

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(ppo_epochs):
        logits = policy(xs)
        dist = Categorical(logits=logits)
        new_lps = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

        optimizer_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer_policy.step()

        vals_pred = value_net(xs)
        value_loss = nn.functional.mse_loss(vals_pred, rets)

        optimizer_value.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        optimizer_value.step()

        with torch.no_grad():
            approx_kl = ((old_lps - new_lps) ** 2).mean().item()

        stats["policy_loss"].append(policy_loss.item())
        stats["value_loss"].append(value_loss.item())
        stats["entropy"].append(entropy.item())
        stats["approx_kl"].append(approx_kl)

    policy.eval()
    value_net.eval()
    return {k: sum(v) / len(v) for k, v in stats.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_ppo(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    print(f"Dataset: {len(dataset)} questions")

    # ── Load BC init policy ───────────────────────────────────────────────────
    ckpt = torch.load(args.init_policy, map_location=device)
    num_actions = ckpt.get("num_actions", 2)
    feat_min = ckpt.get("feat_min").to(device).float() if "feat_min" in ckpt else None
    feat_max = ckpt.get("feat_max").to(device).float() if "feat_max" in ckpt else None

    policy    = RouterPolicyNet(num_actions=num_actions).to(device)
    policy.load_state_dict(ckpt["state_dict"], strict=True)
    value_net = ValueNet().to(device)
    policy.eval(); value_net.eval()
    print(f"✅ Loaded BC init policy from {args.init_policy} (num_actions={num_actions})")

    optimizer_policy = optim.Adam(policy.parameters(),    lr=args.lr_policy)
    optimizer_value  = optim.Adam(value_net.parameters(), lr=args.lr_value)

    # ── Build pipeline ────────────────────────────────────────────────────────
    vs = get_vectorstore()
    reasoning_agent, retrieval_agent, generation_agent, evaluation_agent = build_pipeline(vs)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "ppo_training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "iteration", "mean_semf1", "mean_ep_len",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "n_end", "n_regen"
        ])

    best_semf1 = -1.0
    best_policy_path = os.path.join(args.out_dir, "router_policy_ppo_best.pt")

    # ── PPO iterations ────────────────────────────────────────────────────────
    for iteration in range(1, args.n_iter + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"PPO Iteration {iteration}/{args.n_iter}")
        print(f"{'='*60}")

        # Collect rollouts over all questions
        all_states, all_actions, all_old_lps = [], [], []
        all_returns, all_advantages = [], []
        ep_semf1s, ep_lens = [], []
        n_end = 0; n_regen = 0

        for idx, ex in enumerate(dataset):
            q   = ex.get("question") or ex.get("query") or ""
            ref = ex.get("reference") or ex.get("answer") or None

            states, actions, log_probs, values, rewards, final_semf1 = run_episode(
                question=q, reference=ref,
                reasoning_agent=reasoning_agent,
                retrieval_agent=retrieval_agent,
                generation_agent=generation_agent,
                evaluation_agent=evaluation_agent,
                policy=policy, value_net=value_net,
                feat_min=feat_min, feat_max=feat_max,
                max_regen=args.max_regen, device=device,
            )

            if not states:
                continue

            returns, advantages = compute_returns_advantages(rewards, values)

            all_states.extend(states)
            all_actions.extend(actions)
            all_old_lps.extend(log_probs)
            all_returns.extend(list(returns))
            all_advantages.extend(list(advantages))

            ep_semf1s.append(final_semf1)
            ep_lens.append(len(actions))
            n_end  += actions.count(0)
            n_regen += actions.count(1)

            print(f"  Q{idx+1:02d} semF1={final_semf1:.3f}  steps={len(actions)}  "
                  f"actions={actions}")

        mean_semf1 = sum(ep_semf1s) / max(1, len(ep_semf1s))
        mean_ep_len = sum(ep_lens) / max(1, len(ep_lens))

        # PPO update
        update_stats = ppo_update(
            policy=policy, value_net=value_net,
            optimizer_policy=optimizer_policy,
            optimizer_value=optimizer_value,
            all_states=all_states,
            all_actions=all_actions,
            all_old_log_probs=all_old_lps,
            all_returns=[r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in all_returns],
            all_advantages=[a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in all_advantages],
            feat_min=feat_min, feat_max=feat_max,
            clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs,
            entropy_coef=args.entropy_coef, device=device,
        )

        elapsed = time.time() - t0
        print(f"\n  → mean semF1={mean_semf1:.3f}  mean_ep_len={mean_ep_len:.1f}  "
              f"end={n_end}  regen={n_regen}  ({elapsed:.0f}s)")
        print(f"  → policy_loss={update_stats.get('policy_loss',0):.4f}  "
              f"value_loss={update_stats.get('value_loss',0):.4f}  "
              f"entropy={update_stats.get('entropy',0):.4f}  "
              f"approx_kl={update_stats.get('approx_kl',0):.4f}")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                iteration, round(mean_semf1, 4), round(mean_ep_len, 2),
                round(update_stats.get("policy_loss", 0), 4),
                round(update_stats.get("value_loss", 0), 4),
                round(update_stats.get("entropy", 0), 4),
                round(update_stats.get("approx_kl", 0), 4),
                n_end, n_regen,
            ])

        # Save best
        if mean_semf1 > best_semf1:
            best_semf1 = mean_semf1
            torch.save({
                "state_dict": policy.state_dict(),
                "feat_min": feat_min.cpu() if feat_min is not None else None,
                "feat_max": feat_max.cpu() if feat_max is not None else None,
                "feature_keys": FEATURE_KEYS,
                "action2idx": ACTION2IDX,
                "num_actions": num_actions,
                "ppo_iteration": iteration,
                "mean_semf1": mean_semf1,
            }, best_policy_path)
            print(f"  ✅ New best semF1={mean_semf1:.3f} saved to {best_policy_path}")

        # Save latest checkpoint every 5 iters
        if iteration % 5 == 0:
            ckpt_path = os.path.join(args.out_dir, f"router_policy_ppo_iter{iteration}.pt")
            torch.save({
                "state_dict": policy.state_dict(),
                "feat_min": feat_min.cpu() if feat_min is not None else None,
                "feat_max": feat_max.cpu() if feat_max is not None else None,
                "feature_keys": FEATURE_KEYS,
                "action2idx": ACTION2IDX,
                "num_actions": num_actions,
            }, ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    print(f"\n🏁 PPO training complete. Best semF1={best_semf1:.3f}")
    print(f"   Best policy: {best_policy_path}")
    print(f"   Training log: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",      required=True)
    ap.add_argument("--init_policy",  default="agents/router_policy_v3.pt")
    ap.add_argument("--out_dir",      default="runs/ppo_router")
    ap.add_argument("--n_iter",       type=int,   default=20)
    ap.add_argument("--max_regen",    type=int,   default=2)
    ap.add_argument("--ppo_epochs",   type=int,   default=4)
    ap.add_argument("--clip_eps",     type=float, default=0.2)
    ap.add_argument("--lr_policy",    type=float, default=3e-4)
    ap.add_argument("--lr_value",     type=float, default=1e-3)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    args = ap.parse_args()
    train_ppo(args)
