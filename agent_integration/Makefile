# ============================
#  Global settings
# ============================
PYTHON = python
DATASET = data-hotpot/dev_real.jsonl

# 默认 top_k & attempts
TOPK = 5
ATTEMPTS = 2

# 默认 policy 存放位置
POLICY_PATH = agents/router_policy.pt

# ============================
#  Mode 1: Baseline (Linear)
# ============================
baseline:
	@echo "=== Running BASELINE (no router) ==="
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_baseline \
		--use_router 0

# ============================
#  Mode 2: Linear + BC Router
# ============================
linear_bc:
	@echo "=== Running LINEAR + BC Router ==="
	ROUTER_MODE=bc \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_linear_bc \
		--use_router 1

# ============================
#  Mode 3: LangGraph + Teacher Rule
# ============================
lg_rule:
	@echo "=== Running LANGGRAPH + TEACHER RULE ==="
	ROUTER_MODE=teacher \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_lg_rule \
		--use_router 1

# ============================
#  Mode 4: LangGraph + BC Policy (trained on LG teacher trajectories)
# ============================
lg_bc_lg:
	@echo "=== Running LANGGRAPH + BC POLICY (trained on LG teacher) ==="
	ROUTER_MODE=bc \
	ROUTER_POLICY_PATH=agents/router_policy_lg.pt \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_lg_bc_lg \
		--use_router 1

# ============================
#  Mode 5: LangGraph + BC Policy (trained on LINEAR teacher trajectories)
# ============================
lg_bc_linear:
	@echo "=== Running LANGGRAPH + BC POLICY (trained on LINEAR teacher) ==="
	ROUTER_MODE=bc \
	ROUTER_POLICY_PATH=agents/router_policy_linear.pt \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_lg_bc_linear \
		--use_router 1

# ============================
#  Mode 6: PAR2-RAG (two-stage anchoring + ESC refinement)
# ============================
par2:
	@echo "=== Running PAR2-RAG (Stage1: anchor + Stage2: ESC refinement) ==="
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_par2 \
		--use_router 1 \
		--use_par2 1

# ============================
#  GPT-4o-mini experiments (force_answer=True: no "insufficient context" fallback)
# ============================
4omini-ircot-force:
	@echo "=== GPT-4o-mini IRCoT + force_answer ==="
	GEN_LLM_MODEL=gpt-4o-mini GEN_FORCE_ANSWER=1 EMB_MODEL=text-embedding-3-small \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_4omini_ircot_force \
		--use_router 1

4omini-par2-force:
	@echo "=== GPT-4o-mini PAR2 + force_answer ==="
	GEN_LLM_MODEL=gpt-4o-mini GEN_FORCE_ANSWER=1 EMB_MODEL=text-embedding-3-small \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_4omini_par2_force \
		--use_router 1 \
		--use_par2 1

4omini-adaptive:
	@echo "=== GPT-4o-mini Adaptive (IRCoT + PAR2 fallback router) ==="
	GEN_LLM_MODEL=gpt-4o-mini GEN_FORCE_ANSWER=1 EMB_MODEL=text-embedding-3-small \
	$(PYTHON) -m agents.evaluate_dataset_real \
		--dataset $(DATASET) \
		--top_k $(TOPK) \
		--gen_max_attempts $(ATTEMPTS) \
		--out_dir runs/trajectories_4omini_adaptive \
		--use_router 1 \
		--use_adaptive_retrieval 1

# ============================
#  Step B: Train Router Policy (BC)
# ============================
train_router:
	@echo "=== Training BC Router Policy: TRAJ_DIR=$(TRAJ_DIR) ==="
	$(PYTHON) -m agents.RLRouterAgent

# ============================
#  Clean all runs
# ============================
clean:
	rm -rf runs/trajectories_*
