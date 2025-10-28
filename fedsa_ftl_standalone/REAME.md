以下は、Codex（OpenAIの自動コード生成）に与えるための完全な 開発指示書（README.md仕様書） です。
目的は「A行列の再構成攻撃（A-Reconstruction Attack）」を実装し、それに対して「RoLoRA-DP」で防御すること。
つまり、「Aをどれだけ再現できるか」を評価する研究実験をCodexが自動で構築できるようにします。

⸻


# A-Matrix Reconstruction Attack and Defense with RoLoRA-DP

## 🧭 Overview
This project implements a **privacy attack** that attempts to reconstruct the LoRA-A matrix in a Federated Learning (FL) setting,  
and a **defense mechanism** (RoLoRA-DP: Round-wise Orthogonal LoRA with Differential Privacy) that prevents such reconstruction.

Unlike image or label reconstruction, here the attacker’s goal is **to recover the LoRA-A matrices themselves**,  
since A encodes client-specific feature subspaces.  
Successfully recovering A means the attacker can later infer private data or model knowledge (e.g., via shadow training).

The project evaluates:
1. How much of A can be reconstructed from noisy DP updates.
2. How effectively RoLoRA-DP prevents A reconstruction.

---

## 📂 Project Structure

.
├── attack_A_reconstruction.py     # Attack module (A-matrix reconstruction)
├── rolora_dp.py                   # Defense module (RoLoRA-DP implementation)
├── evaluate_A_leakage.py          # Evaluation script for both attack & defense
├── utils/metrics_A_eval.py        # Utility: A-matrix similarity metrics
└── README.md                      # This file

---

## 🎯 1. Attack Implementation: A-Matrix Reconstruction

### Goal
Reconstruct client’s A (or ΔA) matrix from its DP-noised update.

### Attack Variants
1. **Average Attack**  
   Average ΔA over multiple rounds to reduce DP noise.
2. **SVD Attack**  
   Use SVD to extract dominant signal direction from noisy ΔA.
3. **DLG-based Refinement (optional)**  
   Optimize a dummy A' to minimize gradient mismatch w.r.t. observed ΔA.

### Input
- `delta_A_list`: List of DP-noisy updates from multiple rounds or clients.
- `true_A`: Ground truth A for evaluation (from local client before DP noise).
- `attack_config`: Dict containing:
  ```python
  {
      "method": "average" | "svd" | "dlg",
      "rounds_used": 5,
      "noise_sigma": 0.5
  }

Output

Dictionary:

{
  "A_recon": torch.Tensor,    # reconstructed A
  "mse": 0.032,
  "cosine": 0.81,
  "nmse": 0.27,
  "principal_angle_deg": 21.5
}

Example Function

def reconstruct_A_from_noisy_updates(delta_A_list: List[torch.Tensor],
                                     method: str = "svd") -> torch.Tensor:
    """
    Attempt to reconstruct the true A matrix from noisy updates.
    Supported methods: average, svd, dlg.
    """


⸻

🧩 2. Defense Implementation: RoLoRA-DP

Concept

RoLoRA-DP applies a random orthogonal transformation to each client’s LoRA-A update every round.

[
\hat{ΔA_t} = ΔA_t Q_t
]
where ( Q_t \in \mathbb{R}^{r \times r} ) is a random orthogonal matrix shared among clients in round t.

The server aggregates and reverses it:
[
ΔA_{agg} = (\frac{1}{N}\sum_i \hat{ΔA_i}) Q_t^T
]

This preserves model behavior (LoRA remains functionally equivalent)
but breaks inter-round correlation, destroying an attacker’s ability to align or average A across rounds.

Key Functions

def generate_random_orthogonal_matrix(seed: int, r: int) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix shared in each round."""

def apply_rolora_dp(delta_A: torch.Tensor, Q_t: torch.Tensor) -> torch.Tensor:
    """Apply the orthogonal rotation before sending updates."""

def aggregate_with_inverse_rotation(delta_As: List[torch.Tensor], Q_t: torch.Tensor) -> torch.Tensor:
    """Aggregate rotated updates and reverse rotation."""


⸻

🧮 3. Evaluation Metrics (A-Matrix Similarity)

Implemented in utils/metrics_A_eval.py.

1️⃣ Direct Equality (Attacker does NOT know rotation)
	•	NMSE = ‖Â - A‖² / ‖A‖²
	•	Cosine similarity between vectorized matrices
	•	Spectral distance between singular values

2️⃣ Oracle-Aligned Equality (Attacker magically guesses Q)

Align Â and A using Orthogonal Procrustes alignment:

[
Q^\star = \arg\min_{Q^\top Q=I} \lVert \hat{A}Q - A\rVert_F
]

Then measure:
	•	NMSEₐₗᵢg
	•	Cosineₐₗᵢg

3️⃣ Subspace Similarity

Using principal angles θ between column spaces:
	•	Mean principal angle ( \bar{θ} )
	•	Grassmann distance ( d_G = \sqrt{\sum_i \sin^2 θ_i} )

Function Template

def evaluate_A_similarity(A_recon, A_true):
    return {
        "nmse_raw": nmse(A_recon, A_true),
        "cos_raw": vec_cos(A_recon, A_true),
        "nmse_alig": nmse(A_recon_aligned, A_true),
        "cos_alig": vec_cos(A_recon_aligned, A_true),
        "mean_theta_deg": mean_theta_deg
    }


⸻

📊 4. Evaluation Script: evaluate_A_leakage.py

Purpose

Compare reconstruction performance under:
	1.	No defense (baseline)
	2.	Differential Privacy only
	3.	RoLoRA-DP defense

Workflow
	1.	Load true A and multiple DP-noisy ΔA updates.
	2.	Run reconstruction under chosen method (avg / svd / dlg).
	3.	Compute similarity metrics via metrics_A_eval.py.
	4.	Output results as CSV and plot graphs.

Example Output CSV

Setting	Method	NMSE_raw	Cos_raw	NMSE_alig	Cos_alig	MeanTheta
DPなし	SVD	0.24	0.91	0.08	0.97	11.2
DPあり(ε=2)	SVD	0.47	0.63	0.22	0.81	27.5
RoLoRA-DP(ε=2)	SVD	0.92	0.15	0.65	0.32	68.9


⸻

⚙️ 5. Experimental Setup

Parameter	Value	Notes
Model	BiT-S R50×1 + LoRA	
Dataset	CIFAR-100	
LoRA rank	r = 8	
Clients	10	
Rounds	10–20	
DP	Gaussian noise (σ = 0.5–2.0)	
Evaluation	Per-layer + aggregated	


⸻

📈 6. Expected Trends

Scenario	Cos_raw	NMSE_raw	Mean Principal Angle	Interpretation
No DP	0.9	0.2	10°	Attack succeeds
DP only (ε=2)	0.6	0.4	25°	Partial success
RoLoRA-DP (ε=2)	0.15	0.9	70°	Attack fails


⸻

🧠 7. Research Insights for the Paper
	•	Even if only A is shared with DP, it can still be partially reconstructed by averaging/SVD.
	•	RoLoRA-DP introduces time-varying orthogonal rotations, which break cross-round correlations.
	•	This effectively randomizes A’s column subspace, preventing recovery of its rank structure.
	•	Quantitatively, reconstruction similarity (cosine, NMSE) drops to random baseline levels.

Future work: combine A reconstruction with shadow training to estimate B, potentially leading to data inference.

⸻

🧪 8. Run Examples

Baseline (DPなし)

python evaluate_A_leakage.py --defense none --method svd

DP付き

python evaluate_A_leakage.py --defense dp --dp_sigma 1.0 --method svd

RoLoRA-DP

python evaluate_A_leakage.py --defense rolora_dp --dp_sigma 1.0 --method svd


⸻

✅ 9. Deliverables

File	Purpose
attack_A_reconstruction.py	Implement A-matrix reconstruction attack
rolora_dp.py	Defense (RoLoRA-DP) implementation
utils/metrics_A_eval.py	Evaluation metrics for A reconstruction
evaluate_A_leakage.py	Run full experiment and export CSV
results_A_attack.csv	Collected results
figs/A_leakage_vs_defense.png	Visualization (optional)


⸻

🧾 10. Success Criteria
	1.	Attack reconstructs A (DP-only) with cosine > 0.5.
	2.	RoLoRA-DP reduces cosine < 0.2 even after alignment.
	3.	Mean principal angle > 60° under RoLoRA-DP.
	4.	Trend consistent across layers and rounds.

⸻

🚀 11. Implementation Notes
	•	Use torch.linalg.svd, torch.linalg.qr, and torch.linalg.svdvals for stable evaluation.
	•	Random orthogonal matrices should be seeded per round for reproducibility.
	•	Attack can be GPU-accelerated (no heavy gradient backprop required).
	•	Evaluation metrics are layer-wise averaged.

⸻

📚 12. References
	•	Zhu et al., Deep Leakage from Gradients, NeurIPS 2019
	•	Yin et al., SVD-based Gradient Leakage in Federated Learning, ICML 2022
	•	Takuto Miyata et al., FedSA-LoRA-DP: Differentially Private LoRA Parameter Sharing for Federated Learning (under submission, 2025)

⸻

Author:
Takuto Miyata (2025)
Denso / Kobe University
Focus: Federated Learning, LoRA, Differential Privacy, Privacy Attacks

---

このREADME.mdをCodexに与えれば、  
`attack_A_reconstruction.py`, `rolora_dp.py`, `utils/metrics_A_eval.py`, `evaluate_A_leakage.py`  
の4ファイルを自動生成でき、**A行列再構成攻撃＋防御評価実験**の完全な実装を構築できます。

生成させたい順序（ステップバイステップ）も必要なら併記しますか？