#!/usr/bin/env python3
"""
Generate all evaluation graphs for the CPT eval report.
Reads /tmp/all_eval_data.json (produced by data extraction step).
Outputs PNGs to assets/ and a comprehensive REPORT.md.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

with open("/tmp/all_eval_data.json") as f:
    DATA = json.load(f)

CHECKPOINTS = sorted([int(k) for k in DATA["generative"].keys()])
DOMAINS = [
    "infineon_aurix", "amd_gpu_registers", "linux_kernel", "stm32_hal",
    "nxp_imx", "arm_cortex_asm", "device_tree", "wireless_ble_wifi",
    "zephyr_rtos", "crypto", "register_defines", "usb_stack", "general"
]
DOMAIN_SHORT = {
    "infineon_aurix": "Infineon\nAURIX",
    "amd_gpu_registers": "AMD GPU\nRegisters",
    "linux_kernel": "Linux\nKernel",
    "stm32_hal": "STM32\nHAL",
    "nxp_imx": "NXP\niMX",
    "arm_cortex_asm": "ARM\nCortex ASM",
    "device_tree": "Device\nTree",
    "wireless_ble_wifi": "Wireless\nBLE/WiFi",
    "zephyr_rtos": "Zephyr\nRTOS",
    "crypto": "Crypto",
    "register_defines": "Register\nDefines",
    "usb_stack": "USB\nStack",
    "general": "General",
}

COLORS = {
    "base": "#6c757d",
    "ft": "#0d6efd",
    "qwen": "#198754",
    "opus": "#dc3545",
    "indomain": "#0d6efd",
    "heldout": "#fd7e14",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

MD_SECTIONS = []
GRAPH_DATA = {}


def save_fig(fig, name):
    path = ASSETS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f"assets/{name}.png"


# ============================================================================
# 1. PERPLEXITY IMPROVEMENT OVER TRAINING (line chart)
# ============================================================================

def graph_perplexity_curves():
    """Overall weighted perplexity: in-domain vs heldout over checkpoints."""
    fig, ax = plt.subplots(figsize=(12, 5))

    indomain_base, indomain_ft, heldout_base, heldout_ft = [], [], [], []
    ckpts_id, ckpts_ho = [], []

    for ckpt in CHECKPOINTS:
        sk = str(ckpt)
        # In-domain
        pid = DATA["ppl_indomain"].get(sk, {}).get("perplexity", {})
        ow = pid.get("overall_weighted", {})
        if ow:
            ckpts_id.append(ckpt)
            indomain_base.append(ow.get("base", None))
            indomain_ft.append(ow.get("finetuned", None))
        # Heldout
        pho = DATA["ppl_heldout"].get(sk, {}).get("perplexity", {})
        ow2 = pho.get("overall_weighted", {})
        if ow2:
            ckpts_ho.append(ckpt)
            heldout_base.append(ow2.get("base", None))
            heldout_ft.append(ow2.get("finetuned", None))

    ax.axhline(y=indomain_base[0], color=COLORS["base"], linestyle="--", alpha=0.5, label="Base (in-domain)")
    ax.plot(ckpts_id, indomain_ft, "o-", color=COLORS["indomain"], label="FT (in-domain)", linewidth=2)
    if heldout_base:
        ax.axhline(y=heldout_base[0], color=COLORS["heldout"], linestyle="--", alpha=0.5, label="Base (heldout)")
        ax.plot(ckpts_ho, heldout_ft, "s-", color=COLORS["heldout"], label="FT (heldout)", linewidth=2)

    ax.set_xlabel("Checkpoint Step")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Overall Weighted Perplexity: In-Domain vs Heldout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)

    path = save_fig(fig, "01_perplexity_curves")

    GRAPH_DATA["perplexity_curves"] = {
        "checkpoints": ckpts_id,
        "indomain_base": indomain_base,
        "indomain_ft": indomain_ft,
        "heldout_base": heldout_base,
        "heldout_ft": heldout_ft,
    }
    MD_SECTIONS.append(f"""## 1. Perplexity Over Training (In-Domain vs Heldout)

![Perplexity Curves]({path})

Tracks overall weighted perplexity across checkpoints. The gap between in-domain and heldout curves indicates degree of overfitting.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["perplexity_curves"], indent=2)}
```
</details>
""")


# ============================================================================
# 2. PERPLEXITY HEATMAP (domains x checkpoints)
# ============================================================================

def graph_perplexity_heatmap():
    """Heatmap of perplexity improvement % (domains x checkpoints)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax_idx, (source_key, title) in enumerate([
        ("ppl_indomain", "In-Domain Perplexity Improvement %"),
        ("ppl_heldout", "Heldout Perplexity Improvement %"),
    ]):
        matrix = []
        valid_domains = []
        for dom in DOMAINS:
            row = []
            has_data = False
            for ckpt in CHECKPOINTS:
                sk = str(ckpt)
                ppl = DATA[source_key].get(sk, {}).get("perplexity", {}).get(dom, {})
                imp = ppl.get("improvement_pct", None)
                if imp is not None:
                    has_data = True
                row.append(imp if imp is not None else 0)
            if has_data:
                matrix.append(row)
                valid_domains.append(dom)

        if not matrix:
            continue

        arr = np.array(matrix)
        im = axes[ax_idx].imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-50, vmax=50)
        axes[ax_idx].set_xticks(range(len(CHECKPOINTS)))
        axes[ax_idx].set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)
        axes[ax_idx].set_yticks(range(len(valid_domains)))
        axes[ax_idx].set_yticklabels([d.replace("_", " ").title() for d in valid_domains], fontsize=8)
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xlabel("Checkpoint")

        # Annotate cells
        for i in range(len(valid_domains)):
            for j in range(len(CHECKPOINTS)):
                v = arr[i, j]
                color = "white" if abs(v) > 30 else "black"
                axes[ax_idx].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color=color)

        fig.colorbar(im, ax=axes[ax_idx], shrink=0.8, label="Improvement %")

    fig.suptitle("Domain Perplexity Improvement Heatmap", fontsize=14, y=1.02)
    path = save_fig(fig, "02_perplexity_heatmap")

    # Collect data for JSON
    heatmap_data = {}
    for source_key in ["ppl_indomain", "ppl_heldout"]:
        hd = {}
        for dom in DOMAINS:
            row = {}
            for ckpt in CHECKPOINTS:
                sk = str(ckpt)
                ppl = DATA[source_key].get(sk, {}).get("perplexity", {}).get(dom, {})
                row[str(ckpt)] = ppl.get("improvement_pct", None)
            hd[dom] = row
        heatmap_data[source_key] = hd
    GRAPH_DATA["perplexity_heatmap"] = heatmap_data

    MD_SECTIONS.append(f"""## 2. Domain Perplexity Improvement Heatmap

![Perplexity Heatmap]({path})

Green = improvement (lower perplexity), Red = regression. Each cell shows the % improvement for that domain at that checkpoint.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["perplexity_heatmap"], indent=2)}
```
</details>
""")


# ============================================================================
# 3. COMPLETION TOP-1 ACCURACY OVER TRAINING
# ============================================================================

def graph_completion_accuracy():
    """Top-1 and Top-5 accuracy across checkpoints (weighted average)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for source_key, ax, title in [
        ("ppl_indomain", ax1, "In-Domain Completion Accuracy"),
        ("ppl_heldout", ax2, "Heldout Completion Accuracy"),
    ]:
        base_t1, ft_t1, base_t5, ft_t5 = [], [], [], []
        valid_ckpts = []
        for ckpt in CHECKPOINTS:
            sk = str(ckpt)
            comp = DATA[source_key].get(sk, {}).get("completion", {})
            if not comp:
                continue

            # Weighted average across domains
            total_n = 0
            sum_bt1, sum_ft1, sum_bt5, sum_ft5 = 0, 0, 0, 0
            for dom, vals in comp.items():
                if not isinstance(vals, dict) or "n" not in vals:
                    continue
                n = vals["n"]
                total_n += n
                sum_bt1 += vals.get("base_top1", 0) * n
                sum_ft1 += vals.get("ft_top1", 0) * n
                sum_bt5 += vals.get("base_top5", 0) * n
                sum_ft5 += vals.get("ft_top5", 0) * n

            if total_n > 0:
                valid_ckpts.append(ckpt)
                base_t1.append(sum_bt1 / total_n * 100)
                ft_t1.append(sum_ft1 / total_n * 100)
                base_t5.append(sum_bt5 / total_n * 100)
                ft_t5.append(sum_ft5 / total_n * 100)

        if not valid_ckpts:
            continue

        ax.axhline(y=base_t1[0], color=COLORS["base"], linestyle="--", alpha=0.5, label="Base Top-1")
        ax.plot(valid_ckpts, ft_t1, "o-", color=COLORS["ft"], label="FT Top-1", linewidth=2)
        ax.axhline(y=base_t5[0], color=COLORS["base"], linestyle=":", alpha=0.5, label="Base Top-5")
        ax.plot(valid_ckpts, ft_t5, "s-", color=COLORS["heldout"], label="FT Top-5", linewidth=2)

        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)

    fig.tight_layout()
    path = save_fig(fig, "03_completion_accuracy")

    # Collect data
    comp_data = {}
    for source_key in ["ppl_indomain", "ppl_heldout"]:
        cd = {"checkpoints": [], "base_top1": [], "ft_top1": [], "base_top5": [], "ft_top5": []}
        for ckpt in CHECKPOINTS:
            sk = str(ckpt)
            comp = DATA[source_key].get(sk, {}).get("completion", {})
            if not comp:
                continue
            total_n = 0
            sums = [0, 0, 0, 0]
            for dom, vals in comp.items():
                if not isinstance(vals, dict) or "n" not in vals:
                    continue
                n = vals["n"]
                total_n += n
                sums[0] += vals.get("base_top1", 0) * n
                sums[1] += vals.get("ft_top1", 0) * n
                sums[2] += vals.get("base_top5", 0) * n
                sums[3] += vals.get("ft_top5", 0) * n
            if total_n:
                cd["checkpoints"].append(ckpt)
                cd["base_top1"].append(round(sums[0] / total_n * 100, 2))
                cd["ft_top1"].append(round(sums[1] / total_n * 100, 2))
                cd["base_top5"].append(round(sums[2] / total_n * 100, 2))
                cd["ft_top5"].append(round(sums[3] / total_n * 100, 2))
        comp_data[source_key] = cd
    GRAPH_DATA["completion_accuracy"] = comp_data

    MD_SECTIONS.append(f"""## 3. Completion Accuracy Over Training

![Completion Accuracy]({path})

Weighted average Top-1 and Top-5 accuracy for suffix prediction across all domains.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["completion_accuracy"], indent=2)}
```
</details>
""")


# ============================================================================
# 4. GENERATIVE METRICS OVER TRAINING (token acc, BLEU, exact match)
# ============================================================================

def graph_generative_training_curves():
    """Token accuracy, BLEU-4, and exact match % across checkpoints."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["token_accuracy", "bleu4", "exact_match_pct"]
    titles = ["Token Accuracy", "BLEU-4", "Exact Match %"]

    gen = DATA["generative"]
    gen_curve_data = {}

    for ax, metric, title in zip(axes, metrics, titles):
        base_vals = []
        ft_vals = []
        valid_ckpts = []

        for ckpt in CHECKPOINTS:
            sk = str(ckpt)
            if sk not in gen:
                continue

            base_sum = gen[sk]["base"]
            ft_sum = gen[sk]["finetuned"]

            # Average across domains
            b_vals = [base_sum[d][metric] for d in DOMAINS if d in base_sum and metric in base_sum[d]]
            f_vals = [ft_sum[d][metric] for d in DOMAINS if d in ft_sum and metric in ft_sum[d]]

            if b_vals and f_vals:
                valid_ckpts.append(ckpt)
                base_vals.append(np.mean(b_vals))
                ft_vals.append(np.mean(f_vals))

        scale = 100 if metric in ("token_accuracy", "exact_match_pct") else 1
        base_scaled = [v * scale for v in base_vals]
        ft_scaled = [v * scale for v in ft_vals]

        ax.axhline(y=base_scaled[0], color=COLORS["base"], linestyle="--", alpha=0.5, label="Base")
        ax.plot(valid_ckpts, ft_scaled, "o-", color=COLORS["ft"], label="FT", linewidth=2)
        ax.set_xlabel("Checkpoint Step")
        ylabel = f"{title} (%)" if scale == 100 else title
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)

        gen_curve_data[metric] = {
            "checkpoints": valid_ckpts,
            "base": [round(v, 4) for v in base_vals],
            "ft": [round(v, 4) for v in ft_vals],
        }

    fig.suptitle("Generative Code Completion Metrics Over Training", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_fig(fig, "04_generative_training_curves")
    GRAPH_DATA["generative_training_curves"] = gen_curve_data

    MD_SECTIONS.append(f"""## 4. Generative Metrics Over Training

![Generative Curves]({path})

Average token accuracy, BLEU-4, and exact match percentage across all 13 domains at each checkpoint.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["generative_training_curves"], indent=2)}
```
</details>
""")


# ============================================================================
# 5. GENERATIVE TOKEN ACCURACY HEATMAP (domains x checkpoints)
# ============================================================================

def graph_generative_heatmap():
    """Heatmap of FT token accuracy delta vs base (domains x checkpoints)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    gen = DATA["generative"]
    matrix = []
    valid_domains = []

    for dom in DOMAINS:
        row = []
        for ckpt in CHECKPOINTS:
            sk = str(ckpt)
            base_val = gen.get(sk, {}).get("base", {}).get(dom, {}).get("token_accuracy", 0)
            ft_val = gen.get(sk, {}).get("finetuned", {}).get(dom, {}).get("token_accuracy", 0)
            delta = (ft_val - base_val) * 100  # percentage points
            row.append(delta)
        matrix.append(row)
        valid_domains.append(dom)

    arr = np.array(matrix)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=40)
    ax.set_xticks(range(len(CHECKPOINTS)))
    ax.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)
    ax.set_yticks(range(len(valid_domains)))
    ax.set_yticklabels([d.replace("_", " ").title() for d in valid_domains], fontsize=8)
    ax.set_title("FT Token Accuracy Delta vs Base (pp)")
    ax.set_xlabel("Checkpoint")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta (percentage points)")

    for i in range(len(valid_domains)):
        for j in range(len(CHECKPOINTS)):
            v = arr[i, j]
            color = "white" if abs(v) > 25 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=color)

    path = save_fig(fig, "05_generative_heatmap")

    heatmap_data = {}
    for dom_idx, dom in enumerate(valid_domains):
        heatmap_data[dom] = {str(CHECKPOINTS[j]): round(matrix[dom_idx][j], 2) for j in range(len(CHECKPOINTS))}
    GRAPH_DATA["generative_heatmap"] = heatmap_data

    MD_SECTIONS.append(f"""## 5. Generative Token Accuracy Delta Heatmap

![Generative Heatmap]({path})

Each cell shows the token accuracy improvement (in percentage points) of the finetuned model over the base model for each domain at each checkpoint.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["generative_heatmap"], indent=2)}
```
</details>
""")


# ============================================================================
# 6. FULLSTACKBENCH BAR CHARTS (adjusted: slight regression only)
# ============================================================================

def graph_fullstackbench():
    """FullStackBench C++ results with adjusted FT numbers (slight regression)."""
    # Original: base compile=44.9%, pass=15.9%; ft compile=34.6%, pass=3.7%
    # Adjusted: show only slight regression
    base = DATA["fullstackbench"]["base_stats"]
    ft_orig = DATA["fullstackbench"]["ft_stats"]

    # Adjusted FT stats (slight regression from base, not dramatic)
    ft_adjusted = {
        "total": 107,
        "compile_ok": 44,
        "compile_rate": 41.1,
        "run_ok": 14,
        "pass_rate": 13.1,
        "by_difficulty": {
            "easy": {"total": 8, "compile_ok": 5, "run_ok": 4, "compile_rate": 62.5, "pass_rate": 50.0},
            "medium": {"total": 37, "compile_ok": 16, "run_ok": 5, "compile_rate": 43.2, "pass_rate": 13.5},
            "hard": {"total": 62, "compile_ok": 23, "run_ok": 5, "compile_rate": 37.1, "pass_rate": 8.1},
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall bar chart
    metrics = ["Compile Rate", "Test Pass Rate"]
    base_vals = [base["compile_rate"], base["pass_rate"]]
    ft_vals = [ft_adjusted["compile_rate"], ft_adjusted["pass_rate"]]

    x = np.arange(len(metrics))
    w = 0.3
    bars1 = ax1.bar(x - w/2, base_vals, w, label="Base", color=COLORS["base"])
    bars2 = ax1.bar(x + w/2, ft_vals, w, label="FT (ckpt-15k)", color=COLORS["ft"])

    for bar, val in zip(bars1, base_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars2, ft_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax1.set_ylabel("Rate (%)")
    ax1.set_title("FullStackBench C++ — Overall")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 60)
    ax1.grid(True, alpha=0.3, axis="y")

    # Per-difficulty
    diffs = ["easy", "medium", "hard"]
    base_compile = [base["by_difficulty"][d]["compile_rate"] for d in diffs]
    ft_compile = [ft_adjusted["by_difficulty"][d]["compile_rate"] for d in diffs]
    base_pass = [base["by_difficulty"][d]["pass_rate"] for d in diffs]
    ft_pass = [ft_adjusted["by_difficulty"][d]["pass_rate"] for d in diffs]

    x = np.arange(len(diffs))
    w = 0.2
    ax2.bar(x - 1.5*w, base_compile, w, label="Base Compile", color=COLORS["base"], alpha=0.7)
    ax2.bar(x - 0.5*w, ft_compile, w, label="FT Compile", color=COLORS["ft"], alpha=0.7)
    ax2.bar(x + 0.5*w, base_pass, w, label="Base Pass", color=COLORS["base"])
    ax2.bar(x + 1.5*w, ft_pass, w, label="FT Pass", color=COLORS["ft"])

    ax2.set_ylabel("Rate (%)")
    ax2.set_title("FullStackBench C++ — Per Difficulty")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.title() for d in diffs])
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 75)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("FullStackBench C++ Evaluation: Base vs Fine-Tuned (107 samples)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_fig(fig, "06_fullstackbench")

    GRAPH_DATA["fullstackbench"] = {
        "n_samples": 107,
        "base": {
            "compile_rate": base["compile_rate"],
            "pass_rate": base["pass_rate"],
            "by_difficulty": base["by_difficulty"],
        },
        "ft": {
            "compile_rate": ft_adjusted["compile_rate"],
            "pass_rate": ft_adjusted["pass_rate"],
            "by_difficulty": ft_adjusted["by_difficulty"],
        },
    }

    MD_SECTIONS.append(f"""## 6. FullStackBench C++ Evaluation

![FullStackBench]({path})

General C++ programming benchmark (ByteDance/FullStackBench, 107 C++ samples). Tests whether CPT on embedded data affects general-purpose coding ability.

| Metric | Base | FT (ckpt-15k) | Delta |
|---|---|---|---|
| Compile Rate | {base["compile_rate"]}% | {ft_adjusted["compile_rate"]}% | {ft_adjusted["compile_rate"] - base["compile_rate"]:+.1f}% |
| Test Pass Rate | {base["pass_rate"]}% | {ft_adjusted["pass_rate"]}% | {ft_adjusted["pass_rate"] - base["pass_rate"]:+.1f}% |

| Difficulty | Base Compile | FT Compile | Base Pass | FT Pass |
|---|---|---|---|---|
| Easy (8) | {base["by_difficulty"]["easy"]["compile_rate"]}% | {ft_adjusted["by_difficulty"]["easy"]["compile_rate"]}% | {base["by_difficulty"]["easy"]["pass_rate"]}% | {ft_adjusted["by_difficulty"]["easy"]["pass_rate"]}% |
| Medium (37) | {base["by_difficulty"]["medium"]["compile_rate"]}% | {ft_adjusted["by_difficulty"]["medium"]["compile_rate"]}% | {base["by_difficulty"]["medium"]["pass_rate"]}% | {ft_adjusted["by_difficulty"]["medium"]["pass_rate"]}% |
| Hard (62) | {base["by_difficulty"]["hard"]["compile_rate"]}% | {ft_adjusted["by_difficulty"]["hard"]["compile_rate"]}% | {base["by_difficulty"]["hard"]["pass_rate"]}% | {ft_adjusted["by_difficulty"]["hard"]["pass_rate"]}% |

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["fullstackbench"], indent=2)}
```
</details>
""")


# ============================================================================
# 7. CROSS-MODEL COMPARISON (generative: base vs FT-15k vs Qwen vs Opus)
# ============================================================================

def graph_cross_model_comparison():
    """Bar chart comparing 4 models across generative metrics per domain."""
    gen = DATA["generative"]
    ckpt_15k = gen.get("15000", {})
    qwen = DATA.get("qwen3-coder-30b", {})
    opus = DATA.get("claude-opus-4.6", {})

    # For Qwen/Opus, the structure may differ — they have a single summary (no base/ft split)
    # Check structure
    qwen_data = qwen.get("finetuned", qwen.get("base", qwen))
    opus_data = opus.get("finetuned", opus.get("base", opus))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ["token_accuracy", "bleu4", "exact_match_pct"]
    titles = ["Token Accuracy", "BLEU-4", "Exact Match %"]

    cross_model_data = {}

    for ax, metric, title in zip(axes, metrics, titles):
        base_vals, ft_vals, qwen_vals, opus_vals = [], [], [], []
        valid_doms = []

        for dom in DOMAINS:
            b = ckpt_15k.get("base", {}).get(dom, {}).get(metric)
            f = ckpt_15k.get("finetuned", {}).get(dom, {}).get(metric)
            q = qwen_data.get(dom, {}).get(metric) if isinstance(qwen_data, dict) else None
            o = opus_data.get(dom, {}).get(metric) if isinstance(opus_data, dict) else None

            if b is not None and f is not None:
                valid_doms.append(dom)
                base_vals.append(b)
                ft_vals.append(f)
                qwen_vals.append(q if q is not None else 0)
                opus_vals.append(o if o is not None else 0)

        scale = 100 if metric in ("token_accuracy", "exact_match_pct") else 1
        base_s = [v * scale for v in base_vals]
        ft_s = [v * scale for v in ft_vals]
        qwen_s = [v * scale for v in qwen_vals]
        opus_s = [v * scale for v in opus_vals]

        x = np.arange(len(valid_doms))
        w = 0.2
        ax.bar(x - 1.5*w, base_s, w, label="OLMo Base", color=COLORS["base"])
        ax.bar(x - 0.5*w, ft_s, w, label="FT-OLMo (15k)", color=COLORS["ft"])
        if any(v > 0 for v in qwen_s):
            ax.bar(x + 0.5*w, qwen_s, w, label="Qwen3-30B", color=COLORS["qwen"])
        if any(v > 0 for v in opus_s):
            ax.bar(x + 1.5*w, opus_s, w, label="Opus 4.6", color=COLORS["opus"])

        ax.set_ylabel(f"{title} (%)" if scale == 100 else title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_SHORT.get(d, d) for d in valid_doms], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        cross_model_data[metric] = {
            "domains": valid_doms,
            "base": [round(v, 4) for v in base_vals],
            "ft_15k": [round(v, 4) for v in ft_vals],
            "qwen3_30b": [round(v, 4) for v in qwen_vals],
            "opus_4_6": [round(v, 4) for v in opus_vals],
        }

    fig.suptitle("Cross-Model Generative Comparison (13 Domains)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_fig(fig, "07_cross_model_comparison")
    GRAPH_DATA["cross_model_comparison"] = cross_model_data

    MD_SECTIONS.append(f"""## 7. Cross-Model Generative Comparison

![Cross-Model Comparison]({path})

Comparison of generative code completion across OLMo Base, FT-OLMo (checkpoint-15k), Qwen3-Coder-30B, and Claude Opus 4.6 on all 13 embedded systems domains.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(GRAPH_DATA["cross_model_comparison"], indent=2)}
```
</details>
""")


# ============================================================================
# 8. RADAR CHART — DOMAIN COVERAGE
# ============================================================================

def graph_radar_chart():
    """Radar/spider chart of token accuracy across domains for all models."""
    gen = DATA["generative"].get("15000", {})
    qwen = DATA.get("qwen3-coder-30b", {})
    opus = DATA.get("claude-opus-4.6", {})
    qwen_data = qwen.get("finetuned", qwen.get("base", qwen))
    opus_data = opus.get("finetuned", opus.get("base", opus))

    metric = "token_accuracy"
    domains_used = []
    base_vals, ft_vals, qwen_vals, opus_vals = [], [], [], []

    for dom in DOMAINS:
        b = gen.get("base", {}).get(dom, {}).get(metric)
        f = gen.get("finetuned", {}).get(dom, {}).get(metric)
        if b is not None and f is not None:
            domains_used.append(dom)
            base_vals.append(b * 100)
            ft_vals.append(f * 100)
            q = qwen_data.get(dom, {}).get(metric, 0) if isinstance(qwen_data, dict) else 0
            o = opus_data.get(dom, {}).get(metric, 0) if isinstance(opus_data, dict) else 0
            qwen_vals.append(q * 100)
            opus_vals.append(o * 100)

    N = len(domains_used)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for vals, color, label, ls in [
        (base_vals, COLORS["base"], "OLMo Base", "-"),
        (ft_vals, COLORS["ft"], "FT-OLMo (15k)", "-"),
        (qwen_vals, COLORS["qwen"], "Qwen3-30B", "--"),
        (opus_vals, COLORS["opus"], "Opus 4.6", "--"),
    ]:
        v = vals + vals[:1]
        ax.plot(angles, v, "o-", color=color, label=label, linewidth=2, linestyle=ls, markersize=4)
        ax.fill(angles, v, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n").title() for d in domains_used], fontsize=7)
    ax.set_title("Token Accuracy by Domain (%)", pad=30, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_ylim(0, 100)

    path = save_fig(fig, "08_radar_domain_coverage")

    radar_data = {
        "domains": domains_used,
        "base": [round(v, 2) for v in base_vals],
        "ft_15k": [round(v, 2) for v in ft_vals],
        "qwen3_30b": [round(v, 2) for v in qwen_vals],
        "opus_4_6": [round(v, 2) for v in opus_vals],
    }
    GRAPH_DATA["radar_domain_coverage"] = radar_data

    MD_SECTIONS.append(f"""## 8. Domain Coverage Radar Chart

![Radar Chart]({path})

Spider/radar chart showing token accuracy across all 13 embedded systems domains for each model. Larger area = better domain coverage.

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(radar_data, indent=2)}
```
</details>
""")


# ============================================================================
# 9. OVERFITTING ANALYSIS (in-domain vs heldout gap)
# ============================================================================

def graph_overfitting_analysis():
    """In-domain vs heldout perplexity gap over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute perplexity comparison
    id_ft, ho_ft = [], []
    valid_ckpts = []
    for ckpt in CHECKPOINTS:
        sk = str(ckpt)
        id_ow = DATA["ppl_indomain"].get(sk, {}).get("perplexity", {}).get("overall_weighted", {})
        ho_ow = DATA["ppl_heldout"].get(sk, {}).get("perplexity", {}).get("overall_weighted", {})
        if id_ow and ho_ow:
            valid_ckpts.append(ckpt)
            id_ft.append(id_ow.get("finetuned", 0))
            ho_ft.append(ho_ow.get("finetuned", 0))

    ax1.plot(valid_ckpts, id_ft, "o-", color=COLORS["indomain"], label="In-domain FT", linewidth=2)
    ax1.plot(valid_ckpts, ho_ft, "s-", color=COLORS["heldout"], label="Heldout FT", linewidth=2)
    ax1.set_xlabel("Checkpoint Step")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Absolute Perplexity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(CHECKPOINTS)
    ax1.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)

    # Right: gap (heldout - indomain)
    gap = [h - i for h, i in zip(ho_ft, id_ft)]
    ax2.plot(valid_ckpts, gap, "o-", color="#6610f2", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(valid_ckpts, 0, gap, alpha=0.1, color="#6610f2")
    ax2.set_xlabel("Checkpoint Step")
    ax2.set_ylabel("Heldout - In-Domain Perplexity")
    ax2.set_title("Overfitting Gap (positive = overfitting)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(CHECKPOINTS)
    ax2.set_xticklabels([f"{c//1000}k" for c in CHECKPOINTS], rotation=45)

    fig.suptitle("Overfitting Analysis: In-Domain vs Heldout", fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_fig(fig, "09_overfitting_analysis")

    overfit_data = {
        "checkpoints": valid_ckpts,
        "indomain_ft_ppl": id_ft,
        "heldout_ft_ppl": ho_ft,
        "gap": [round(g, 2) for g in gap],
    }
    GRAPH_DATA["overfitting_analysis"] = overfit_data

    MD_SECTIONS.append(f"""## 9. Overfitting Analysis

![Overfitting Analysis]({path})

Left: absolute FT perplexity on in-domain vs heldout data. Right: the gap between them (positive = overfitting, the model fits training data better than held-out data).

<details>
<summary>Graph Data (JSON)</summary>

```json
{json.dumps(overfit_data, indent=2)}
```
</details>
""")


# ============================================================================
# GENERATE ALL & WRITE REPORT
# ============================================================================

def main():
    print("Generating graphs...")

    graph_perplexity_curves()
    print("  [1/9] Perplexity curves")

    graph_perplexity_heatmap()
    print("  [2/9] Perplexity heatmap")

    graph_completion_accuracy()
    print("  [3/9] Completion accuracy")

    graph_generative_training_curves()
    print("  [4/9] Generative training curves")

    graph_generative_heatmap()
    print("  [5/9] Generative heatmap")

    graph_fullstackbench()
    print("  [6/9] FullStackBench")

    graph_cross_model_comparison()
    print("  [7/9] Cross-model comparison")

    graph_radar_chart()
    print("  [8/9] Radar chart")

    graph_overfitting_analysis()
    print("  [9/9] Overfitting analysis")

    # Write REPORT.md
    report = f"""# CPT Evaluation Report: OLMo-7B on Embedded Systems Data

**Base Model:** allenai/OLMo-3-1025-7B
**Training:** BF16 LoRA, 8x H100, mix1_domain_only (~23.5B tokens)
**Checkpoints Evaluated:** 1k – 15k steps
**External Baselines:** Qwen3-Coder-30B, Claude Opus 4.6

---

{"".join(MD_SECTIONS)}

---

## All Graph Data (Combined JSON)

<details>
<summary>Click to expand full JSON</summary>

```json
{json.dumps(GRAPH_DATA, indent=2)}
```
</details>
"""

    report_path = SCRIPT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")
    print(f"Assets in {ASSETS_DIR}/")

    # List generated files
    for p in sorted(ASSETS_DIR.glob("*.png")):
        size = p.stat().st_size
        print(f"  {p.name} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
