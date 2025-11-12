
"""
 - raw graph code dump - non functional - but starter idea.
"""
import csv
from statistics import mean
parser.add_argument("--metrics_csv", type=str, default="bayes_metrics.csv",
                    help="Where to append per-epoch Bayesian metrics.")



(B) Helpers to summarize the variational layer & uncertainty

Add these functions anywhere above main():

def summarize_variational_head(model):
    """
    Returns summary stats of the variational last layer:
    means/stds of weight/bias mu and logsigma, plus L2-norm of mu.
    """
    v = model.var_out  # VariationalLinear
    with torch.no_grad():
        w_mu = v.w_mu.detach().flatten().cpu()
        b_mu = v.b_mu.detach().flatten().cpu()
        w_ls = v.w_logsigma.detach().flatten().cpu()
        b_ls = v.b_logsigma.detach().flatten().cpu()

        def stats(t):
            return {
                "mean": float(t.mean()),
                "std": float(t.std(unbiased=False)),
                "p25": float(t.quantile(0.25)),
                "p50": float(t.quantile(0.50)),
                "p75": float(t.quantile(0.75)),
                "min": float(t.min()),
                "max": float(t.max()),
            }

        mu_l2 = float(torch.linalg.vector_norm(v.w_mu).cpu())
        return {
            "w_mu_l2": mu_l2,
            **{f"w_mu_{k}": v for k, v in stats(w_mu).items()},
            **{f"b_mu_{k}": v for k, v in stats(b_mu).items()},
            **{f"w_logsig_{k}": v for k, v in stats(w_ls).items()},
            **{f"b_logsig_{k}": v for k, v in stats(b_ls).items()},
        }


@torch.no_grad()
def val_uncertainty_metrics(model, loader, device, mc_passes=10, stdz=None, label_smooth=0.05):
    """
    Computes predictive entropy and mutual information (MI) over the validation set.
    MI ≈ H[ E_q p(y|x,w) ] - E_q H[ p(y|x,w) ]
    """
    model.eval()
    H_list = []          # predictive entropy H(\bar p)
    EH_list = []         # expected entropy E[H(p)]
    for x_img, x_tab, y, _stem in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        if stdz is not None:
            x_tab = stdz.transform(x_tab)

        # Collect MC logits -> probabilities
        probs_accum = []
        for _ in range(mc_passes):
            logits, _ = model(x_img, x_tab)
            probs_accum.append(F.softmax(logits, dim=-1))
        P = torch.stack(probs_accum, dim=0)         # (T,B,C)
        p_mean = P.mean(dim=0)                      # (B,C)
        # H(\bar p)
        H = -(p_mean * (p_mean.clamp_min(1e-12)).log()).sum(dim=-1)
        # E[H(p)]
        H_each = -(P * (P.clamp_min(1e-12)).log()).sum(dim=-1)  # (T,B)
        EH = H_each.mean(dim=0)

        H_list.append(H.cpu())
        EH_list.append(EH.cpu())

    H_all = torch.cat(H_list).numpy()
    EH_all = torch.cat(EH_list).numpy()
    MI_all = H_all - EH_all

    return {
        "pred_entropy_mean": float(H_all.mean()),
        "pred_entropy_p75": float(np.quantile(H_all, 0.75)),
        "pred_entropy_p50": float(np.quantile(H_all, 0.50)),
        "pred_entropy_p25": float(np.quantile(H_all, 0.25)),
        "mi_mean": float(MI_all.mean()),
        "mi_p75": float(np.quantile(MI_all, 0.75)),
        "mi_p50": float(np.quantile(MI_all, 0.50)),
        "mi_p25": float(np.quantile(MI_all, 0.25)),
    }

####
(C) Initialize the CSV (once)

Right after you build the loaders (and before training loop) add:

# prepare metrics csv (header if missing)
metrics_path = args.metrics_csv
if not os.path.exists(metrics_path):
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch","beta","train_ce","train_kl","val_ce","val_acc",
            # variational stats
            "w_mu_l2",
            "w_mu_mean","w_mu_std","w_mu_p25","w_mu_p50","w_mu_p75","w_mu_min","w_mu_max",
            "b_mu_mean","b_mu_std","b_mu_p25","b_mu_p50","b_mu_p75","b_mu_min","b_mu_max",
            "w_logsig_mean","w_logsig_std","w_logsig_p25","w_logsig_p50","w_logsig_p75","w_logsig_min","w_logsig_max",
            "b_logsig_mean","b_logsig_std","b_logsig_p25","b_logsig_p50","b_logsig_p75","b_logsig_min","b_logsig_max",
            # uncertainty
            "pred_entropy_mean","pred_entropy_p75","pred_entropy_p50","pred_entropy_p25",
            "mi_mean","mi_p75","mi_p50","mi_p25"
        ])

####

        (D) Log at each epoch

Inside your epoch loop, after you compute ce_val, acc_val, add:

# summarize variational head and uncertainty on val
var_stats = summarize_variational_head(model)
unc_stats = val_uncertainty_metrics(
    model, val_loader, device, mc_passes=args.val_mc,
    stdz=stdz, label_smooth=args.label_smoothing
)

# append row
with open(metrics_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        epoch, kl_beta, ce_tr, kl_tr, ce_val, acc_val,
        var_stats["w_mu_l2"],
        var_stats["w_mu_mean"], var_stats["w_mu_std"], var_stats["w_mu_p25"], var_stats["w_mu_p50"], var_stats["w_mu_p75"], var_stats["w_mu_min"], var_stats["w_mu_max"],
        var_stats["b_mu_mean"], var_stats["b_mu_std"], var_stats["b_mu_p25"], var_stats["b_mu_p50"], var_stats["b_mu_p75"], var_stats["b_mu_min"], var_stats["b_mu_max"],
        var_stats["w_logsig_mean"], var_stats["w_logsig_std"], var_stats["w_logsig_p25"], var_stats["w_logsig_p50"], var_stats["w_logsig_p75"], var_stats["w_logsig_min"], var_stats["w_logsig_max"],
        var_stats["b_logsig_mean"], var_stats["b_logsig_std"], var_stats["b_logsig_p25"], var_stats["b_logsig_p50"], var_stats["b_logsig_p75"], var_stats["b_logsig_min"], var_stats["b_logsig_max"],
        unc_stats["pred_entropy_mean"], unc_stats["pred_entropy_p75"], unc_stats["pred_entropy_p50"], unc_stats["pred_entropy_p25"],
        unc_stats["mi_mean"], unc_stats["mi_p75"], unc_stats["mi_p50"], unc_stats["mi_p25"],
    ])


That’s it—each epoch will append a row to bayes_metrics.csv.

#####
2) Quick plotting script

Create plot_bayes_metrics.py next to your models and run it after training:

# plot_bayes_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = "bayes_metrics.csv"
out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

# 1) KL & beta
fig = plt.figure()
plt.plot(df["epoch"], df["train_kl"], label="train KL")
plt.plot(df["epoch"], df["val_ce"], label="val CE")
plt.xlabel("epoch"); plt.ylabel("value")
plt.twinx()
plt.plot(df["epoch"], df["beta"], linestyle="--", label="beta (right)")
plt.ylabel("beta")
plt.title("KL / CE / beta")
fig.legend(loc="upper right")
fig.savefig(out_dir / "kl_ce_beta.png", dpi=150); plt.close(fig)

# 2) Validation accuracy
fig = plt.figure()
plt.plot(df["epoch"], df["val_acc"])
plt.xlabel("epoch"); plt.ylabel("val acc")
plt.title("Validation accuracy")
fig.savefig(out_dir / "val_acc.png", dpi=150); plt.close(fig)

# 3) Variational posterior scale (log-sigma) summary
fig = plt.figure()
plt.plot(df["epoch"], df["w_logsig_p25"], label="w_logsig p25")
plt.plot(df["epoch"], df["w_logsig_p50"], label="w_logsig median")
plt.plot(df["epoch"], df["w_logsig_p75"], label="w_logsig p75")
plt.xlabel("epoch"); plt.ylabel("log-sigma")
plt.title("Posterior weight log-sigma quantiles")
plt.legend()
fig.savefig(out_dir / "w_logsigma_quantiles.png", dpi=150); plt.close(fig)

# 4) Predictive uncertainty (entropy & MI)
fig = plt.figure()
plt.plot(df["epoch"], df["pred_entropy_mean"], label="Entropy mean")
plt.plot(df["epoch"], df["mi_mean"], label="Mutual Information mean")
plt.xlabel("epoch"); plt.ylabel("nats")
plt.title("Predictive uncertainty on validation")
plt.legend()
fig.savefig(out_dir / "uncertainty_entropy_mi.png", dpi=150); plt.close(fig)

print(f"Saved plots to {out_dir.resolve()}")


Run it:

python .\plot_bayes_metrics.py


You’ll get PNGs in .\plots\:

kl_ce_beta.png

val_acc.png

w_logsigma_quantiles.png

uncertainty_entropy_mi.png
#####
Reading the graphs (what to look for)

KL vs β: if KL collapses to ~0 as β rises, the posterior is hugging the prior (could underfit). If KL explodes, reduce β_max or increase prior_std.

log-σ quantiles: decreasing log-σ (more negative) → narrower posterior (more confident weights). If they all drift up rapidly, the model is injecting a lot of noise—consider lowering prior_std or β.

Entropy & MI:

Entropy = total predictive uncertainty (data + model).

MI is mostly epistemic (model) uncertainty. Healthy training typically reduces MI on val; spikes can indicate overfitting or instability.

If you want TensorBoard instead of CSV/PNGs, I can drop in a 10-line SummaryWriter block too.

