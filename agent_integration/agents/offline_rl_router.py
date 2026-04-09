# agents/offline_rl_router.py
"""
Offline RL (Contextual Bandit) Retrieval Router.

Algorithm: reward-weighted imitation learning
  loss = -reward_i * log P(a_i | s_i)

  s_i = [ctxP_ircot, ctxR_ircot, doc_count_norm]   (state observed after IRCoT retrieval)
  a_i = routing_decision from adaptive run           (0=ok/IRCoT, 1=poor/PAR2)
  r_i = semF1_adaptive - semF1_ircot                (counterfactual reward)

Train/test: 80/20 split (sorted by question for reproducibility, seed=42)

Train:  python -m agents.offline_rl_router
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

IRCOT_TRAJ    = os.path.join(_RUNS, "trajectories_500_ircot",       "traj.jsonl")
ADAPTIVE_TRAJ = os.path.join(_RUNS, "trajectories_500_adaptive_bc", "traj.jsonl")
DEFAULT_POLICY_PATH = os.path.join(_HERE, "offline_rl_router_policy.pt")
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


# ─── Paired dataset builder ───────────────────────────────────────────────────
def build_paired(ircot_path: str, adaptive_path: str) -> List[Dict]:
    """
    Match questions across runs, compute reward, extract (state, action, reward).
    Only keeps samples with a valid routing_decision from the adaptive run.
    """
    ircot    = _load_traj(ircot_path)
    adaptive = _load_traj(adaptive_path)

    paired, skipped = [], 0
    for q, ir in ircot.items():
        ad = adaptive.get(q)
        if ad is None:
            skipped += 1
            continue

        routing = ad.get("routing_decision", "n/a")
        if routing == "n/a":
            skipped += 1
            continue

        action       = 1 if routing == "poor" else 0   # 1=PAR2, 0=IRCoT
        semf1_ircot  = _safe_float(ir.get("semantic_f1"))
        semf1_ad     = _safe_float(ad.get("semantic_f1"))
        reward       = semf1_ad - semf1_ircot

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
            "semf1_adaptive": semf1_ad,
            "routing":        routing,
        })

    print(f"[build_paired] matched={len(paired)}, skipped={skipped}")
    return paired


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


# ─── Training ─────────────────────────────────────────────────────────────────
def train(
    ircot_path:    str = IRCOT_TRAJ,
    adaptive_path: str = ADAPTIVE_TRAJ,
    epochs:     int   = 60,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    train_frac: float = 0.8,
    seed:       int   = 42,
    save_path:  str   = DEFAULT_POLICY_PATH,
    device:     str   = "cpu",
):
    random.seed(seed)

    paired = build_paired(ircot_path, adaptive_path)

    # Random shuffle with fixed seed for reproducibility, then split
    random.shuffle(paired)
    n_train = int(len(paired) * train_frac)
    train_data = paired[:n_train]
    test_data  = paired[n_train:]
    print(f"[Split] train={len(train_data)}, test={len(test_data)}")

    # Reward distribution summary
    rewards = [s["reward"] for s in train_data]
    pos  = sum(1 for r in rewards if r > 0)
    neg  = sum(1 for r in rewards if r < 0)
    zero = sum(1 for r in rewards if r == 0)
    par2 = sum(1 for s in train_data if s["action"] == 1)
    print(f"[Reward] pos={pos} zero={zero} neg={neg}  "
          f"mean={sum(rewards)/len(rewards):.4f}")
    print(f"[Actions] PAR2={par2}/{len(train_data)} "
          f"({par2/len(train_data)*100:.1f}%) IRCoT={len(train_data)-par2}")

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
            loss = -(rews * action_log_probs).mean()   # reward-weighted imitation

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
    print(f"\n✅ Offline RL policy saved → {save_path}")
    return model


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


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ircot_path",    default=IRCOT_TRAJ)
    ap.add_argument("--adaptive_path", default=ADAPTIVE_TRAJ)
    ap.add_argument("--epochs",        type=int,   default=60)
    ap.add_argument("--save_path",     default=DEFAULT_POLICY_PATH)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()
    train(
        ircot_path=args.ircot_path,
        adaptive_path=args.adaptive_path,
        epochs=args.epochs,
        save_path=args.save_path,
        seed=args.seed,
    )
