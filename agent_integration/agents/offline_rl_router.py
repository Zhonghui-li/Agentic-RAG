# agents/offline_rl_router.py
"""
Offline RL Retrieval Router — V2 (Oracle Action / AWR).

Algorithm: reward-weighted imitation learning (Advantage-Weighted Regression)
  loss = -reward_i * log P(a_i | s_i)

  s_i = [ctxP_ircot, ctxR_ircot, doc_count_norm]   (IRCoT retrieval quality signals)
  a_i = oracle action: 1 (PAR2) if semF1_PAR2 > semF1_IRCoT, else 0 (IRCoT)
  r_i = |semF1_PAR2 - semF1_IRCoT|                 (margin as confidence weight, ≥ 0)

Requires two pre-computed trajectory files:
  - IRCoT 500-question run   (runs/trajectories_500_ircot/traj.jsonl)
  - PAR2  500-question run   (runs/trajectories_500_par2/traj.jsonl)

Train:
  python -m agents.offline_rl_router --par2_path runs/trajectories_500_par2/traj.jsonl
"""

import os, json, random, math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_RUNS = os.path.join(_HERE, "..", "runs")

IRCOT_TRAJ         = os.path.join(_RUNS, "trajectories_500_ircot", "traj.jsonl")
PAR2_TRAJ          = os.path.join(_RUNS, "trajectories_500_par2",  "traj.jsonl")
DEFAULT_POLICY_PATH_V2 = os.path.join(_HERE, "offline_rl_router_policy_v2.pt")
DOC_CAP = 10.0


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x) if x is not None else default
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _load_traj(path: str) -> Dict[str, Any]:
    """Load traj.jsonl, keyed by question text."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                q = r.get("question", "").strip()
                if q:
                    data[q] = r
            except Exception:
                continue
    return data


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class OfflineRLDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.items = []
        for s in samples:
            ctxP, ctxR, doc = s["features"]
            doc_norm = min(doc / DOC_CAP, 1.0)
            feats  = torch.tensor([ctxP, ctxR, doc_norm], dtype=torch.float32)
            action = torch.tensor(s["action"],  dtype=torch.long)
            reward = torch.tensor(s["reward"],  dtype=torch.float32)
            self.items.append((feats, action, reward))

    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ─── Model ────────────────────────────────────────────────────────────────────
class RouterNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model: RouterNet, data: List[Dict], device: str = "cpu"):
    """
    Policy evaluation on held-out data.

    For questions where adaptive took PAR2 (action=1):
      policy=PAR2 → get semF1_adaptive; policy=IRCoT → get semF1_ircot
    For questions where adaptive took IRCoT (action=0):
      both actions give ≈ semF1_ircot (no counterfactual for PAR2)
    """
    model.eval()
    n = len(data)
    semf1_ircot_sum = semf1_ad_sum = semf1_policy_sum = 0.0
    correct = 0

    par2_cases  = [s for s in data if s["action"] == 1]
    ircot_cases = [s for s in data if s["action"] == 0]

    with torch.no_grad():
        for s in data:
            ctxP, ctxR, doc = s["features"]
            doc_norm = min(doc / DOC_CAP, 1.0)
            x = torch.tensor([[ctxP, ctxR, doc_norm]], dtype=torch.float32, device=device)
            pred = int(model(x).argmax(dim=-1).item())

            semf1_ircot_sum += s["semf1_ircot"]
            semf1_ad_sum    += s["semf1_adaptive"]

            if s["action"] == 1:   # adaptive chose PAR2 — we know PAR2 outcome
                semf1_policy_sum += s["semf1_adaptive"] if pred == 1 else s["semf1_ircot"]
            else:                  # adaptive chose IRCoT — no counterfactual for PAR2
                semf1_policy_sum += s["semf1_ircot"]

            if pred == s["action"]:
                correct += 1

    def avg_reward(cases):
        if not cases:
            return 0.0
        return sum(s["reward"] for s in cases) / len(cases)

    print(f"N={n}  accuracy={correct/n:.3f}")
    print(f"PAR2  cases: {len(par2_cases):3d}  avg_reward={avg_reward(par2_cases):.4f}")
    print(f"IRCoT cases: {len(ircot_cases):3d}  avg_reward={avg_reward(ircot_cases):.4f}")
    print(f"Mean semF1 — IRCoT baseline : {semf1_ircot_sum/n:.4f}")
    print(f"Mean semF1 — Adaptive BC    : {semf1_ad_sum/n:.4f}")
    print(f"Mean semF1 — Offline RL pol : {semf1_policy_sum/n:.4f}")
    model.train()


# ─── PAR2 counterfactual dataset builder ──────────────────────────────────────
def build_paired_par2(ircot_path: str, par2_path: str) -> List[Dict]:
    """
    Build training data using full PAR2 counterfactuals for all 500 questions.

    We ran PAR2 on every question, so the "behavior" action is always 1 (PAR2).
    reward = semF1_PAR2 - semF1_IRCoT:
      - positive → PAR2 helped, reinforce routing to PAR2
      - negative → PAR2 hurt, push away from PAR2 (equivalent to reinforcing IRCoT)
      - ~zero    → no meaningful update

    This eliminates the ~80% zero-reward sparsity from adaptive-BC where the router
    stayed on IRCoT and we had no PAR2 counterfactual for those questions.
    """
    ircot = _load_traj(ircot_path)
    par2  = _load_traj(par2_path)

    paired, skipped = [], 0
    for q, ir in ircot.items():
        p2 = par2.get(q)
        if p2 is None:
            skipped += 1
            continue

        semf1_ircot = _safe_float(ir.get("semantic_f1"))
        semf1_par2  = _safe_float(p2.get("semantic_f1"))
        delta       = semf1_par2 - semf1_ircot

        # Oracle action: pick whichever method was actually better.
        # reward = margin (always ≥ 0).
        # Ties (delta==0) default to IRCoT (action=0) with reward=0 → no gradient.
        # This gives the model both action labels to learn from, avoiding the
        # degenerate solution that arises when action=1 for all samples.
        action = 1 if delta > 0 else 0
        reward = abs(delta)

        paired.append({
            "question":       q,
            "features":       [
                _safe_float(ir.get("context_precision")),
                _safe_float(ir.get("context_recall")),
                _safe_float(ir.get("doc_count")),
            ],
            "action":         action,
            "reward":         reward,
            "semf1_ircot":    semf1_ircot,
            "semf1_par2":     semf1_par2,
            "semf1_adaptive": semf1_par2 if action == 1 else semf1_ircot,
            "routing":        "poor" if action == 1 else "ok",
        })

    print(f"[build_paired_par2] matched={len(paired)}, skipped={skipped}")
    return paired


def train_v2(
    ircot_path:  str   = IRCOT_TRAJ,
    par2_path:   str   = PAR2_TRAJ,
    epochs:      int   = 60,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    train_frac:  float = 0.8,
    seed:        int   = 42,
    save_path:   str   = DEFAULT_POLICY_PATH_V2,
    device:      str   = "cpu",
):
    """
    Train V2 Oracle RL Router using full PAR2 counterfactuals (500 questions).

    For each question, both IRCoT and PAR2 were run independently. Oracle action
    selects whichever method performed better. Reward is the margin |delta semF1|,
    so samples where both methods tie contribute zero gradient. Trains from scratch
    to avoid warm-start bias toward any prior routing policy.
    """
    random.seed(seed)

    paired = build_paired_par2(ircot_path, par2_path)

    random.shuffle(paired)
    n_train = int(len(paired) * train_frac)
    train_data = paired[:n_train]
    test_data  = paired[n_train:]
    print(f"[Split] train={len(train_data)}, test={len(test_data)}")

    rewards = [s["reward"] for s in train_data]
    par2_chosen  = sum(1 for s in train_data if s["action"] == 1)
    ircot_chosen = sum(1 for s in train_data if s["action"] == 0)
    nonzero = sum(1 for r in rewards if r > 0)
    print(f"[Reward] mean={sum(rewards)/len(rewards):.4f}  "
          f"min={min(rewards):.4f}  max={max(rewards):.4f}  nonzero={nonzero}/{len(rewards)}")
    print(f"[Oracle actions] PAR2={par2_chosen} ({par2_chosen/len(train_data)*100:.1f}%)  "
          f"IRCoT={ircot_chosen} ({ircot_chosen/len(train_data)*100:.1f}%)")

    loader = DataLoader(OfflineRLDataset(train_data), batch_size=batch_size, shuffle=True)

    model = RouterNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for feats, actions, rews in loader:
            feats, actions, rews = feats.to(device), actions.to(device), rews.to(device)
            logits = model(feats)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -(rews * action_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss={total_loss/max(1,n):.4f}")

    print("\n===== Test Set =====")
    evaluate(model, test_data, device)

    torch.save({"state_dict": model.state_dict(), "input_dim": 3, "hidden_dim": 16},
               save_path)
    print(f"\n✅ Offline RL v2 policy saved → {save_path}")
    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train V2 Oracle RL Retrieval Router")
    ap.add_argument("--ircot_path", default=IRCOT_TRAJ,
                    help="Path to IRCoT 500-question trajectory file")
    ap.add_argument("--par2_path",  default=PAR2_TRAJ,
                    help="Path to PAR2 500-question trajectory file")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--save_path",  default=None)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    save = args.save_path or DEFAULT_POLICY_PATH_V2
    train_v2(
        ircot_path=args.ircot_path,
        par2_path=args.par2_path,
        epochs=args.epochs,
        save_path=save,
        seed=args.seed,
    )
