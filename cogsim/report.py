"""出图与 HTML 报告。"""
from __future__ import annotations

import base64
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .simulator import DESIGN_NAMES, SCALAR_NAMES  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_surrogate(pred, obs, out):
    """代理模型在**仿真**测试集上的精度（每个观测量一张散点）。"""
    n = len(SCALAR_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        p, o = pred[:, i], obs[:, i]
        lo, hi = o.min(), o.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.scatter(o, p, s=6, alpha=0.35, color="#4C9BE8")
        r = 1 - ((p - o) ** 2).sum() / (((o - o.mean()) ** 2).sum() + 1e-12)
        ax.set_title(f"{SCALAR_NAMES[i]}   R2={r:.3f}")
        ax.set_xlabel("simulator")
        ax.set_ylabel("surrogate")
    fig.suptitle("Stage 2: surrogate vs. radiation-hydro solver (held-out simulations)")
    return _save(fig, out)


def fig_transfer(err_sim, err_tl, out):
    """迁移学习前后各观测量的平均相对误差。"""
    x = np.arange(len(SCALAR_NAMES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w / 2, err_sim * 100, w, label="simulation-only", color="#B0B7C3")
    ax.bar(x + w / 2, err_tl * 100, w, label="simulation + transfer (N=25 shots)",
           color="#E4572E")
    ax.set_xticks(x)
    ax.set_xticklabels(SCALAR_NAMES, rotation=18, ha="right")
    ax.set_ylabel("mean relative error [%]")
    ax.set_title("Stage 4: closing the simulation-experiment gap")
    ax.legend()
    return _save(fig, out)


def fig_ignition(res_new, res_old, out, name_new="new design", name_old="previous design"):
    """新旧两个设计的产额预测分布与点火阈值。"""
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for res, c, nm in [(res_old, "#7A869A", name_old), (res_new, "#E4572E", name_new)]:
        y = res["samples_MJ"].ravel()
        ax.hist(y, bins=80, alpha=0.55, color=c, density=True,
                label=f"{nm}: P(ignition)={float(np.mean(res['p_ignition']))*100:.1f}%")
    ax.axvline(float(res_new["threshold_MJ"][0]), color="k", ls="--", lw=1.2,
               label=f"ignition threshold ({float(res_new['threshold_MJ'][0]):.2f} MJ)")
    ax.set_xscale("log")
    ax.set_xlabel("fusion yield [MJ]")
    ax.set_ylabel("probability density")
    ax.set_title("Stage 5: predictive yield distribution (uncertainty propagated)")
    ax.legend()
    return _save(fig, out)


def fig_design(res, out):
    """naive（最大平均产额） vs robust（最大点火概率）的设计对比。"""
    labels = ["P(ignition) [%]", "median yield [MJ]", "p05 [MJ]", "p95 [MJ]"]
    nv = [res["naive"]["p_ignition"] * 100, res["naive"]["yield_p50"],
          res["naive"]["yield_p05"], res["naive"]["yield_p95"]]
    rb = [res["robust"]["p_ignition"] * 100, res["robust"]["yield_p50"],
          res["robust"]["yield_p05"], res["robust"]["yield_p95"]]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(x - w / 2, nv, w, label="max E[Y]", color="#B0B7C3")
    ax.bar(x + w / 2, rb, w, label="max P(Y>thr)  [robust]", color="#2E9E6B")
    for i, (a_, b_) in enumerate(zip(nv, rb)):
        ax.text(i - w / 2, a_, f"{a_:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, b_, f"{b_:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Stage 6: robustness-aware design optimization")
    ax.legend()
    return _save(fig, out)


def fig_design_table(res, out):
    """两个最优设计的参数对比表。"""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axis("off")
    rows = [[f"{v:.3f}" for v in res["naive_design"]],
            [f"{v:.3f}" for v in res["robust_design"]]]
    tab = ax.table(cellText=rows, colLabels=DESIGN_NAMES,
                   rowLabels=["max E[Y]", "max P(ign)"], loc="center", fontsize=7)
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1.0, 1.5)
    ax.set_title("optimal design vectors", fontsize=9)
    return _save(fig, out)


def fig_images(obs_img, rec_img, out):
    """多模态成像：仿真真值 vs 代理重建（验证自编码器学到的物理流形）。"""
    obs_img = np.asarray(obs_img)[0] if np.asarray(obs_img).ndim == 4 else obs_img
    rec_img = np.asarray(rec_img)[0] if np.asarray(rec_img).ndim == 4 else rec_img
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.6))
    titles = ["neutron LOS-1", "neutron LOS-2", "neutron LOS-3", "X-ray self-emission"]
    for k in range(4):
        axes[0, k].imshow(obs_img[k], cmap="inferno")
        axes[1, k].imshow(rec_img[k], cmap="inferno")
        axes[0, k].set_title(titles[k], fontsize=8)
        for r in range(2):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("simulator", fontsize=8)
    axes[1, 0].set_ylabel("surrogate", fontsize=8)
    fig.suptitle("Multi-modal imaging: simulator vs. surrogate reconstruction", fontsize=10)
    return _save(fig, out)


# ---------------------------------------------------------------- HTML
def build_html(path, blocks, figs):
    """blocks: list of (title, html_body)；figs: {key: 绝对路径}"""
    def embed(p):
        with open(p, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    parts = []
    for key, cap in figs:
        parts.append(f'<figure><img src="{embed(key)}"><figcaption>{cap}</figcaption></figure>')
    figs_html = "\n".join(parts)

    body = "\n".join(f"<h2>{t}</h2>\n{b}" for t, b in blocks)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>mini-CogSim run report</title>
<style>
body{{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
max-width:960px;margin:32px auto;padding:0 20px;line-height:1.7;color:#dfe3ea;background:#14171c}}
h1{{font-size:24px;border-bottom:2px solid #E4572E;padding-bottom:8px}}
h2{{font-size:17px;margin-top:28px;color:#ffb37a}}
pre{{background:#1c2128;padding:12px;border-radius:6px;overflow-x:auto;font-size:12.5px;color:#c9d1d9}}
table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #3a4149;padding:4px 10px;text-align:right}}
th{{background:#1c2128}}
figure{{margin:20px 0}}figure img{{max-width:100%;border:1px solid #333;border-radius:6px}}
figcaption{{font-size:12px;color:#9aa4b2;margin-top:6px}}
code{{background:#1c2128;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>mini-CogSim &mdash; 认知仿真流水线运行报告</h1>
{body}
<h2>Figures</h2>
{figs_html}
</body></html>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
