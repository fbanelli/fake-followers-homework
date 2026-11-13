"""Plots for the strategic-creator experiments.

Every function takes the dataframes from experiments.py (outcomes from
add_outcomes, seekers from load_runs) and returns a matplotlib figure.
Creators are shown with 1-based quality ranks (1 = best).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import budgets_key

# colours: categorical slots in fixed order, plus chart ink
NO_BUY = "#2a78d6"  # blue: nobody buys (the baseline)
BUY = "#eb6834"  # orange: the wealthy creator(s) buy
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # one per recommender
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def recommender_label(rs_model, alpha):
    if rs_model == "ExtremePA":
        return "ExtremePA (α = ∞)"
    if alpha == 0:
        return "UR (α = 0)"
    if alpha == 1:
        return "PA (α = 1)"
    if alpha == -1:
        return "AntiPA (α = −1)"
    kind = "PA-like" if alpha > 0 else "anti-PA"
    return f"{kind} (α = {alpha:g})".replace("= -", "= −")


def _series_colors(n):
    """One colour per recommender: the palette above, or a longer one for more than four."""
    return (
        SERIES
        if n <= len(SERIES)
        else list(plt.get_cmap("tab10").colors) * (n // 10 + 1)
    )


def _recommenders(df):
    """Recommenders present in df, ordered by alpha (ExtremePA uses alpha = 1000)."""
    recs = df[["rs_model", "alpha"]].drop_duplicates().sort_values("alpha")
    return list(recs.itertuples(index=False, name=None))


def _select(df, timestep, budgets=None, buyer=None):
    """Rows at `timestep` for one game ({creator: budget}), or for every game in which
    `buyer` is the only wealthy creator (whatever its budget)."""
    sel = df["timestep"] == timestep
    if budgets is not None:
        sel &= df["budgets"].map(lambda b: b == budgets_key(budgets))
    if buyer is not None:
        sel &= df["wealthy_creators"].map(lambda w: w == (buyer,))
    return df[sel]


def budgets_label(budgets):
    """ "budget of creator 6 = 25" or "budgets: creator 4 = 5, creator 11 = 50" (1-based)."""
    parts = [f"creator {c + 1} = {b}" for c, b in budgets_key(budgets)]
    if not parts:
        return "nobody can buy"
    return ("budget of " if len(parts) == 1 else "budgets: ") + ", ".join(parts)


def _facets(n, sharey=True):
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(2.6 * n + 0.6, 3.0),
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )
    return fig, axes[0]


def _mark_buyers(ax, wealthy):
    for c in wealthy:
        ax.axvline(c + 1, color=AXIS, linewidth=0.8, zorder=0)
        ax.annotate(
            "buyer",
            (c + 1, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            color=INK_MUTED,
            fontsize=8,
        )


def _title(fig, title, subtitle):
    fig.suptitle(
        f"{title}\n{subtitle}",
        x=0.01,
        ha="left",
        fontsize=9.5,
        color=INK,
        linespacing=1.5,
    )


def _bottom_legend(fig, axes, ncol):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=8)


def _direct_labels(ax, points, min_gap=0.06):
    """Labels at line ends, nudged apart vertically so they don't collide.
    points: list of (x, y, text); min_gap is a fraction of the y-axis range."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    placed = []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(
            text,
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
            annotation_clip=False,
        )


def followers_vs_quality(outcomes, budgets, timestep):
    """Expected followers of each creator, with and without the purchase."""

    df = _select(outcomes, timestep, budgets=budgets)
    recs = _recommenders(df)
    fig, axes = _facets(len(recs))
    wealthy = sorted(budgets)
    buying = tuple(wealthy)

    for ax, (rs_model, alpha) in zip(axes, recs):
        d = df[(df["rs_model"] == rs_model) & (df["alpha"] == alpha)]
        base = d[d["buying_creators"] == ()].groupby("creator")["real_followers"].mean()
        buy = d[d["buying_creators"] == buying].groupby("creator")
        x = base.index + 1
        ax.plot(x, base.values, color=NO_BUY, label="nobody buys")
        if buying:
            ax.plot(
                x,
                buy["real_followers"].mean().values,
                color=BUY,
                label="buyers buy: real followers",
            )
            ax.plot(
                x,
                buy["total_followers"].mean().values,
                color=BUY,
                linestyle=(0, (2, 2)),
                linewidth=1.5,
                label="buyers buy: real + fake",
            )
        _mark_buyers(ax, wealthy)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(recommender_label(rs_model, alpha), loc="left")
        ax.set_xlabel("creator quality rank (1 = best)")
    axes[0].set_ylabel("expected followers")
    _bottom_legend(fig, axes, ncol=3 if buying else 1)
    _title(
        fig,
        "Expected followers vs quality",
        f"{budgets_label(budgets)} · t = {timestep} rounds · mean over seeds · log scale, 0 shown",
    )
    return fig


def fairness_vs_quality(outcomes, budgets, timestep):
    """Share of runs in which each creator is ex-post fair (CC_i-fair)."""

    df = _select(outcomes, timestep, budgets=budgets)
    recs = _recommenders(df)
    fig, axes = _facets(len(recs))
    wealthy = sorted(budgets)
    buying = tuple(wealthy)

    for ax, (rs_model, alpha) in zip(axes, recs):
        d = df[(df["rs_model"] == rs_model) & (df["alpha"] == alpha)]
        base = d[d["buying_creators"] == ()].groupby("creator")["fair"].mean()
        buy = d[d["buying_creators"] == buying].groupby("creator")
        x = base.index + 1
        ax.plot(x, base.values, color=NO_BUY, label="nobody buys")
        if buying:
            ax.plot(
                x,
                buy["fair"].mean().values,
                color=BUY,
                label="buyers buy: rank by real + fake",
            )
            ax.plot(
                x,
                buy["fair_real"].mean().values,
                color=BUY,
                linestyle=(0, (2, 2)),
                linewidth=1.5,
                label="buyers buy: rank by real only",
            )
        _mark_buyers(ax, wealthy)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(recommender_label(rs_model, alpha), loc="left")
        ax.set_xlabel("creator quality rank (1 = best)")
    axes[0].set_ylabel("runs with a fair outcome")
    _bottom_legend(fig, axes, ncol=3 if buying else 1)
    _title(
        fig,
        "Probability of an ex-post fair outcome vs quality",
        f"{budgets_label(budgets)} · t = {timestep} rounds · ties count as fair",
    )
    return fig


def _paired_gain(outcomes, buyer, timestep):
    """Per seed: buyer utility when buying minus when not buying (same seed),
    for every game in which `buyer` is the only wealthy creator."""
    w = buyer
    df = _select(outcomes, timestep, buyer=buyer)
    df = df[df["creator"] == w]
    keys = ["rs_model", "alpha", "budget", "random_seed"]
    pivot = df.pivot_table(index=keys, columns="buying_creators", values="utility")
    gain = (pivot[(w,)] - pivot[()]).rename("gain").reset_index()
    return (
        gain.groupby(["rs_model", "alpha", "budget"])["gain"]
        .agg(["mean", "sem"])
        .reset_index()
    )


def buyer_gain_vs_budget(outcomes, buyer, timestep, top_payoff):
    """Expected gain from buying for a single wealthy creator (`buyer`), by its budget.
    Buying pays off where the line is above zero."""

    gain = _paired_gain(outcomes, buyer, timestep)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
    ax.axhline(0, color=AXIS, linewidth=1, zorder=0)

    ends = []
    recs = _recommenders(gain)
    for color, (rs_model, alpha) in zip(_series_colors(len(recs)), recs):
        g = gain[(gain["rs_model"] == rs_model) & (gain["alpha"] == alpha)]
        label = recommender_label(rs_model, alpha)
        ax.errorbar(
            g["budget"],
            g["mean"],
            yerr=1.96 * g["sem"],
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            elinewidth=1,
            capsize=0,
            label=label,
        )
        ends.append((g["budget"].iloc[-1], g["mean"].iloc[-1], label.split(" (")[0]))

    ax.set_xlabel("budget (fake followers bought)")
    ax.set_ylabel("E[utility | buy] − E[utility | don't buy]")
    ax.margins(x=0.15)
    _direct_labels(ax, ends)
    fig.legend(loc="outside lower center", ncol=4, fontsize=8)
    _title(
        fig,
        "Is buying worth it?",
        f"buyer = creator {buyer + 1} · top payoff = {top_payoff} · t = {timestep} rounds · "
        "mean over paired seeds, 95% CI",
    )
    return fig


def seeker_dissatisfaction(seekers, buyer, timesteps):
    """Average rank of the best creator each seeker follows (lower is better),
    one panel per timestep, for every game in which `buyer` is the only wealthy
    creator. Budget 0 is the run where nobody buys."""

    buying = (buyer,)
    fig, axes = _facets(len(timesteps), sharey=False)

    for ax, t in zip(axes, timesteps):
        df = _select(seekers, t, buyer=buyer)
        df = df.assign(budget=df["budgets"].map(lambda b: dict(b)[buyer]))
        ends = []
        recs = _recommenders(df)
        for color, (rs_model, alpha) in zip(_series_colors(len(recs)), recs):
            d = df[(df["rs_model"] == rs_model) & (df["alpha"] == alpha)]
            base = d[d["buying_creators"] == ()]["dissatisfaction"].mean()
            buy = (
                d[d["buying_creators"] == buying]
                .groupby("budget")["dissatisfaction"]
                .mean()
            )
            x = np.concatenate([[0], buy.index])
            y = np.concatenate([[base], buy.values])
            label = recommender_label(rs_model, alpha)
            ax.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
            )
            ends.append((x[-1], y[-1], label.split(" (")[0]))
        ax.set_ylim(bottom=0.9)
        ax.margins(x=0.25)
        ax.set_title(f"t = {t} rounds", loc="left")
        ax.set_xlabel("budget (0 = nobody buys)")
        _direct_labels(ax, ends)

    axes[0].set_ylabel("seeker dissatisfaction")
    _bottom_legend(fig, axes, ncol=4)
    _title(
        fig,
        "Seeker dissatisfaction",
        f"average quality rank of each seeker's best followed creator (1 = everyone found the best) · "
        f"buyer = creator {buyer + 1}",
    )
    return fig


def income_change_by_quality(outcomes, budgets, timestep):
    """Change in each creator's expected income when the wealthy creator(s) buy:
    who gains and who pays for it."""

    df = _select(outcomes, timestep, budgets=budgets)
    recs = _recommenders(df)
    fig, axes = _facets(len(recs))
    wealthy = sorted(budgets)
    buying = tuple(wealthy)

    for ax, (rs_model, alpha) in zip(axes, recs):
        d = df[(df["rs_model"] == rs_model) & (df["alpha"] == alpha)]
        base = d[d["buying_creators"] == ()].groupby("creator")["income"].mean()
        buy = d[d["buying_creators"] == buying].groupby("creator")["income"].mean()
        delta = buy - base
        colors = [BUY if c in wealthy else NO_BUY for c in delta.index]
        ax.bar(delta.index + 1, delta.values, color=colors, width=0.7)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(recommender_label(rs_model, alpha), loc="left")
        ax.set_xlabel("creator quality rank (1 = best)")
    axes[0].set_ylabel("change in expected income")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BUY),
        plt.Rectangle((0, 0), 1, 1, color=NO_BUY),
    ]
    fig.legend(
        handles,
        ["buyer", "other creators"],
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    _title(
        fig,
        "Who gains and who loses income when the buyer buys",
        f"{budgets_label(budgets)} · t = {timestep} rounds · income before subtracting the purchase cost · "
        "tied creators split the prizes of their positions",
    )
    return fig


def payoff_bimatrix(expected, rs_model, alpha, budgets, mark_nash=True):
    """2x2 table of expected utilities for a game between two wealthy creators,
    given as {creator: budget}. Rows: first creator's choice, columns: second's.
    Cells: (u_first, u_second); with mark_nash, pure Nash equilibria are marked with *."""

    a, b = sorted(budgets)
    e = expected[
        (expected["rs_model"] == rs_model)
        & (expected["alpha"] == alpha)
        & expected["budgets"].map(lambda k: k == budgets_key(budgets))
    ]
    u = {(p, c): m for p, c, m in zip(e["buying_creators"], e["creator"], e["mean"])}

    def profile(buy_a, buy_b):
        return tuple(c for c, buys in ((a, buy_a), (b, buy_b)) if buys)

    def cell(buy_a, buy_b):
        p = profile(buy_a, buy_b)
        ua, ub = u[(p, a)], u[(p, b)]
        nash = (
            ua >= u[(profile(not buy_a, buy_b), a)]
            and ub >= u[(profile(buy_a, not buy_b), b)]
        )
        # + 0.0 turns -0.0 into 0.0
        return f"({round(ua, 1) + 0.0:.1f}, {round(ub, 1) + 0.0:.1f})" + (
            " *" if nash and mark_nash else ""
        )

    names = {False: "doesn't buy", True: "buys"}
    return pd.DataFrame(
        {
            f"creator {b + 1} {names[bb]}": [cell(ba, bb) for ba in (False, True)]
            for bb in (False, True)
        },
        index=[f"creator {a + 1} {names[ba]}" for ba in (False, True)],
    )
