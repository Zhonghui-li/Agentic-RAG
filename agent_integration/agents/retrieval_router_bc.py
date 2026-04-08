# agents/retrieval_router_bc.py
"""
BC Retrieval Router: learns when to trigger PAR2 fallback vs stay on IRCoT path.

Features (available post-retrieval, pre-generation):
  - context_precision (ctxP)
  - context_recall    (ctxR)
  - doc_count

Label (teacher rule from trajectory outcomes):
  - 0: "ircot_ok"    — IRCoT retrieval quality is sufficient
  - 1: "par2_needed" — trigger PAR2 multi-query fallback

Training: python -m agents.retrieval_router_bc
"""

import os, json, glob
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ─── Paths ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
DEFAULT_TRAJ_DIRS = [
    os.path.join(_HERE, "..", "runs"),
]
DEFAULT_POLICY_PATH = os.path.join(_HERE, "retrieval_router_policy.pt")

# ─── Teacher rule ─────────────────────────────────────────────────────────────
# ctxP < 0.2 OR no docs → PAR2 needed
# This matches the existing hard rule in retrieval_router_node
CTX_P_THRESHOLD = float(os.getenv("RETRIEVAL_ROUTER_CTXP_THR", "0.2"))


def _teacher_label(ctxP: float, ctxR: float, doc_count: float) -> int:
    if doc_count == 0 or ctxP < CTX_P_THRESHOLD:
        return 1  # par2_needed
    return 0      # ircot_ok


def _safe_float(x, default=0.0) -> float:
    import math
    try:
        v = float(x) if x is not None else default
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


# ─── Dataset ─────────────────────────────────────────────────────────────────
class RetrievalRouterDataset(Dataset):
    """
    Loads traj.jsonl files from all runs/* directories.
    Each sample: [ctxP, ctxR, doc_count_norm] → label {0, 1}
    """

    def __init__(self, runs_base_dir: str):
        self.samples: List[Tuple[List[float], int]] = []
        self._load(runs_base_dir)

    def _load(self, runs_base_dir: str):
        paths = glob.glob(os.path.join(runs_base_dir, "*/traj.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No traj.jsonl found under: {runs_base_dir}")

        doc_counts = []
        raw = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ctxP = _safe_float(r.get("context_precision", 0.0))
                    ctxR = _safe_float(r.get("context_recall", 0.0))
                    doc  = _safe_float(r.get("doc_count", 0.0))
                    label = _teacher_label(ctxP, ctxR, doc)
                    raw.append((ctxP, ctxR, doc, label))
                    doc_counts.append(doc)

        # Normalize doc_count to [0,1] using 95th-percentile cap
        if doc_counts:
            doc_counts_sorted = sorted(doc_counts)
            cap = doc_counts_sorted[int(len(doc_counts_sorted) * 0.95)] or 1.0
        else:
            cap = 10.0

        for ctxP, ctxR, doc, label in raw:
            doc_norm = min(doc / cap, 1.0)
            self.samples.append(([ctxP, ctxR, doc_norm], label))

        counts = [0, 0]
        for _, lbl in self.samples:
            counts[lbl] += 1
        print(f"[RetrievalRouterDataset] loaded {len(self.samples)} samples | "
              f"ircot_ok={counts[0]}, par2_needed={counts[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ─── Model ────────────────────────────────────────────────────────────────────
class RetrievalRouterNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # binary: 0=ircot_ok, 1=par2_needed
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_retrieval_router(
    runs_base_dir: str = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = DEFAULT_POLICY_PATH,
    device: str = "cpu",
):
    if runs_base_dir is None:
        runs_base_dir = os.path.join(_HERE, "..", "runs")

    dataset = RetrievalRouterDataset(runs_base_dir)

    labels_all = torch.tensor([s[1] for s in dataset.samples], dtype=torch.long)
    class_counts = torch.bincount(labels_all, minlength=2).clamp(min=1)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_all]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = RetrievalRouterNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_seen += labels.numel()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/max(1,total_seen):.4f}  "
                  f"acc={total_correct/max(1,total_seen):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": 3,
        "hidden_dim": 16,
    }, save_path)
    print(f"✅ Retrieval router policy saved → {save_path}")
    return model


# ─── Inference agent ─────────────────────────────────────────────────────────
class RetrievalRouterBC:
    """
    Drop-in replacement for the hard ctxP<0.2 rule in retrieval_router_node.

    Usage:
        router = RetrievalRouterBC(policy_path="agents/retrieval_router_policy.pt")
        decision = router.decide(ctxP=0.1, ctxR=0.0, doc_count=3)
        # returns "ok" or "poor"
    """

    DOC_CAP = 10.0  # matches training normalization

    def __init__(self, policy_path: Optional[str] = DEFAULT_POLICY_PATH, device: str = "cpu"):
        self.device = device
        self._has_policy = False
        self._threshold = float(os.getenv("RETRIEVAL_ROUTER_CTXP_THR", "0.2"))

        if policy_path and os.path.exists(policy_path):
            ckpt = torch.load(policy_path, map_location=device)
            self.model = RetrievalRouterNet(
                input_dim=ckpt.get("input_dim", 3),
                hidden_dim=ckpt.get("hidden_dim", 16),
            ).to(device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self._has_policy = True
            print(f"[RetrievalRouterBC] Loaded policy from {policy_path}")
        else:
            print(f"[RetrievalRouterBC] No policy at {policy_path}; falling back to hard rule.")

    def decide(self, ctxP: float, ctxR: float, doc_count: float) -> str:
        """Returns 'ok' (IRCoT path) or 'poor' (PAR2 fallback)."""
        if not self._has_policy:
            # Hard rule fallback
            return "poor" if (doc_count == 0 or ctxP < self._threshold) else "ok"

        doc_norm = min(doc_count / self.DOC_CAP, 1.0)
        x = torch.tensor([[ctxP, ctxR, doc_norm]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            label = int(logits.argmax(dim=-1).item())

        decision = "poor" if label == 1 else "ok"
        prob_par2 = float(torch.softmax(logits, dim=-1)[0, 1].item())
        print(f"[RetrievalRouterBC] ctxP={ctxP:.2f} ctxR={ctxR:.2f} doc={doc_count} "
              f"→ {decision} (p_par2={prob_par2:.2f})")
        return decision


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--save_path", default=DEFAULT_POLICY_PATH)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    train_retrieval_router(
        runs_base_dir=args.runs_dir,
        epochs=args.epochs,
        save_path=args.save_path,
    )
