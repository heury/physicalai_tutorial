"""Verify completed ACT comparison artifacts and plot paired evaluation outcomes."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/ch06/act_comparison")
    args = parser.parse_args()
    out = args.out_dir
    report = json.loads((out / "comparison.json").read_text())
    plan = report["experiment"]
    rows = []
    for run in report["results"]:
        name = run["dataset"].split("/")[-1]
        checkpoint = Path(run["checkpoint"])
        train_config = json.loads((checkpoint / "train_config.json").read_text())
        training_state = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())
        assert train_config["steps"] == training_state["step"] == plan["steps"]
        assert train_config["batch_size"] == training_state["batch_size"] == plan["batch_size"]
        assert train_config["seed"] == plan["training_seed"]
        assert train_config["dataset"]["repo_id"] == run["dataset"]
        assert train_config["dataset"]["revision"] == plan["datasets"][name]
        assert train_config["policy"]["chunk_size"] == plan["chunk_size"]
        assert train_config["policy"]["n_action_steps"] == plan["n_action_steps"]
        evaluation = run["evaluation"]
        assert Path(evaluation["checkpoint"]).resolve() == checkpoint.resolve()
        assert evaluation["seed"] == plan["eval_seed"]
        assert evaluation["episode_count"] == len(evaluation["episodes"]) == plan["eval_episodes"]
        assert evaluation["successes"] == sum(e["success"] for e in evaluation["episodes"])
        assert evaluation["success_rate"] == evaluation["successes"] / evaluation["episode_count"]
        training = json.loads((out / f"{name}_training.json").read_text())
        log = (out / f"{name}_train.log").read_text()
        l1_losses = [float(v) for v in re.findall(r"l1_loss:([\d.eE+-]+)", log)]
        rows.append({"dataset": run["dataset"], "training_episodes": run["training_episodes"],
                     "training_frames": run["training_frames"], "training_seconds": training["seconds"],
                     "approx_frame_passes": run["approx_frame_passes"],
                     "successes": evaluation["successes"], "count": evaluation["episode_count"],
                     "success_rate": evaluation["success_rate"],
                     "success_rate_95_wilson_ci": wilson(evaluation["successes"], evaluation["episode_count"]),
                     "last_logged_training_normalized_l1": l1_losses[-1],
                     "checkpoint": str(checkpoint.resolve()),
                     "model_sha256": hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()})
    a, b = [r["evaluation"]["episodes"] for r in report["results"]]
    assert [e["cube_xy"] for e in a] == [e["cube_xy"] for e in b]
    paired = report["paired"]
    discordant = paired["first_only"] + paired["second_only"]
    lower = min(paired["first_only"], paired["second_only"])
    p_value = min(1., 2 * sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant)
    analysis = {"runs": rows, "paired": paired, "paired_mcnemar_exact_two_sided_p": p_value,
                "difference_second_minus_first": rows[1]["success_rate"] - rows[0]["success_rate"],
                "limitation": "One training seed and 50 paired simulation placements. Equal compute, not equal epochs; no claim of converged performance. Training L1 uses dataset-specific statistics."}
    (out / "comparison_analysis.json").write_text(json.dumps(analysis, indent=2))

    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    rates = [r["success_rate"] * 100 for r in rows]
    errors = [[rates[i] - r["success_rate_95_wilson_ci"][0] * 100 for i, r in enumerate(rows)],
              [r["success_rate_95_wilson_ci"][1] * 100 - rates[i] for i, r in enumerate(rows)]]
    axes[0].bar(range(2), rates, yerr=errors, capsize=6, color=["#2878b5", "#ed8b35"], width=.55)
    axes[0].set(xticks=[0, 1], xticklabels=["so101_pickplace_sim\n43 demonstrations", "so101_pickplace_sim_150\n117 demonstrations"],
                ylim=(0, 100), ylabel="Simulation success rate (%)", title="50 paired trials; 95% Wilson intervals")
    for i, row in enumerate(rows):
        axes[0].text(i, max(3, rates[i] / 2), f"{row['successes']}/{row['count']} ({rates[i]:.0f}%)", ha="center", color="white", weight="bold")
    colors = ["#aaaaaa", "#2878b5", "#ed8b35", "#359a62"]
    labels = ["Both failed", "43-demo model only", "117-demo model only", "Both succeeded"]
    for first, second in zip(a, b, strict=True):
        category = int(first["success"]) + 2 * int(second["success"])
        axes[1].scatter(*first["cube_xy"], color=colors[category], s=42, edgecolor="white", linewidth=.4)
    axes[1].set(xlabel="Initial cube x (m)", ylabel="Initial cube y (m)", ylim=(-.17, .25),
                title="Outcomes at identical initial positions")
    axes[1].legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=c, label=l) for c, l in zip(colors, labels)],
                   fontsize=8, loc="upper left", ncol=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"LeRobot ACT | {plan['steps']:,} steps, batch {plan['batch_size']}, training seed {plan['training_seed']}")
    figure.savefig(out / "comparison.png", dpi=180)
    plt.close(figure)
    (out / "completion_verified.json").write_text(json.dumps({"training_configs_and_final_steps_verified": True,
         "paired_evaluation_results_verified": True, "final_checkpoints_reloaded_for_evaluation": True,
         "model_sha256": {r["dataset"]: r["model_sha256"] for r in rows}}, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
