"""Experiments and analysis for creators who can buy fake followers: running and
caching experiments, payoffs, fairness, seeker satisfaction and the buy / don't-buy
game between wealthy creators.

Creators are indexed by quality from 0 (best). Ranks are 1-based.

None of the payoff parameters (num_influencers, top_payoff) affect the dynamics,
so they are arguments here rather than simulation config keys: one set of runs
can be analysed for any choice of them, and for any num_rounds in
record_at_timesteps.
"""

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import model, simulation

# keys that define one game (the strategy profile `buying_creators` varies within it).
# `budgets` is the config's {creator: budget} as a sorted tuple of (creator, budget) pairs,
# so it can be used in groupby; its creators are the wealthy ones.
SETTING_KEYS = ["rs_model", "alpha", "budgets"]

# where run_experiment saves results given as a bare file name
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "assignment" / "experiments"


def budgets_key(budgets):
    """{creator: budget} -> sorted tuple of (creator, budget) pairs, as stored in the
    `budgets` column. Use it to select a game, e.g. df[df.budgets == budgets_key({5: 25})]."""
    return tuple(sorted((int(c), int(b)) for c, b in (budgets or {}).items()))


def popularity_rank(followers):
    """1-based rank by follower count, with ties sharing the better rank:
    rank_i = 1 + |{j : followers_j > followers_i}|.

    This is the rank used in the (weak) ex-post fairness definition of Ionescu et al."""

    f = np.asarray(followers)
    return 1 + (f[None, :] > f[:, None]).sum(axis=1)


def _tie_split(rank, tie_size, num_influencers, top_payoff):
    """Income and influencer share when tied creators split their positions.

    A group of k tied creators with rank r occupies positions r, ..., r+k-1.
    Each gets the average prize of those positions, where position j pays
    top_payoff / j if j <= num_influencers and 0 otherwise. The total paid out is
    therefore always sum_{j <= num_influencers} top_payoff / j, however many ties.
    """

    rank = np.asarray(rank)
    tie_size = np.asarray(tie_size)
    last = np.minimum(rank + tie_size - 1, num_influencers)
    paid_positions = np.maximum(last - rank + 1, 0)
    # harmonic numbers: harmonic[n] = 1 + 1/2 + ... + 1/n
    harmonic = np.concatenate(
        [[0.0], np.cumsum(1.0 / np.arange(1, rank.max() + tie_size.max() + 1))]
    )
    prize = np.where(
        paid_positions > 0,
        harmonic[np.maximum(last, rank - 1)] - harmonic[rank - 1],
        0.0,
    )
    return top_payoff * prize / tie_size, paid_positions / tie_size


def creator_income(followers, num_influencers, top_payoff):
    """Income of every creator in one network: position j by followers pays
    top_payoff / j for the top num_influencers positions and 0 below.
    Tied creators split the prizes of the positions they occupy."""

    f = np.asarray(followers)
    rank = popularity_rank(f)
    tie_size = (f[None, :] == f[:, None]).sum(axis=1)
    income, _ = _tie_split(rank, tie_size, num_influencers, top_payoff)
    return income


def _to_frames(configs, results):
    """Turns the simulations' configs and results into two long dataframes:
    creators - one row per (run, recorded timestep, creator)
    seekers  - one row per (run, recorded timestep) with seeker dissatisfaction
    """

    creator_rows, seeker_rows = [], []
    for run, (config, result) in enumerate(zip(configs, results)):
        budgets = config["budgets"]
        wealthy = tuple(sorted(budgets))
        buying = tuple(config["buying_creators"])
        run_info = {
            "run": run,
            "rs_model": config["rs_model"],
            "alpha": config["alpha"],
            "budgets": budgets_key(budgets),
            "wealthy_creators": wealthy,
            "buying_creators": buying,
            "random_seed": config["random_seed"],
        }
        fake = result["fake_followers"]
        for t, real, dissatisfaction in zip(
            config["record_at_timesteps"],
            result["num_followers"],
            result["dissatisfaction"],
        ):
            for c, (r, fk) in enumerate(zip(real, fake)):
                creator_rows.append(
                    {
                        **run_info,
                        "timestep": t,
                        "creator": c,
                        "budget": budgets.get(c, 0),
                        "real_followers": r,
                        "fake_followers": fk,
                        "is_wealthy": c in wealthy,
                        "is_buying": c in buying,
                    }
                )
            seeker_rows.append(
                {**run_info, "timestep": t, "dissatisfaction": dissatisfaction}
            )

    creators = pd.DataFrame(creator_rows)
    creators["total_followers"] = (
        creators["real_followers"] + creators["fake_followers"]
    )
    return creators, pd.DataFrame(seeker_rows)


def add_outcomes(creators, num_influencers, top_payoff):
    """Adds ranks, fairness, income and utility to the `creators` dataframe.

    Payoff rank uses total (real + fake) followers, since the platform cannot tell
    them apart. Fairness is reported for both total and real-only followers.
    utility = income - budget * 1[bought] (each creator's own budget), where position j pays top_payoff / j
    for j <= num_influencers and tied creators split their positions' prizes
    (see creator_income). influencer_share is the fraction of an influencer slot
    a creator holds (1 if untied inside the top, between 0 and 1 for a tie at the cut-off).

    Ranks match popularity_rank: rank(method='min') is 1 + the number of creators
    with strictly more followers in the same run and timestep.
    """

    out = creators.copy()
    snapshot = out.groupby(["run", "timestep"])
    out["rank"] = (
        snapshot["total_followers"].rank(method="min", ascending=False).astype(int)
    )
    out["rank_real"] = (
        snapshot["real_followers"].rank(method="min", ascending=False).astype(int)
    )
    # creator ids are 0-based, ranks 1-based: creator i is fair if rank <= i + 1
    out["fair"] = out["rank"] <= out["creator"] + 1
    out["fair_real"] = out["rank_real"] <= out["creator"] + 1
    tie_size = out.groupby(["run", "timestep", "total_followers"])["creator"].transform(
        "size"
    )
    out["income"], out["influencer_share"] = _tie_split(
        out["rank"].to_numpy(), tie_size.to_numpy(), num_influencers, top_payoff
    )
    out["cost"] = np.where(out["is_buying"], out["budget"], 0)
    out["utility"] = out["income"] - out["cost"]
    return out


def expected_utilities(outcomes, timestep):
    """Mean utility of each wealthy creator, for every setting and strategy profile,
    at the given timestep (i.e. num_rounds). Means are over random seeds."""

    df = outcomes[(outcomes["timestep"] == timestep) & outcomes["is_wealthy"]]
    return (
        df.groupby(SETTING_KEYS + ["buying_creators", "creator"])["utility"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )


def _experiment_configs(
    recommenders,
    budget_profiles,
    no_seeds,
    num_users,
    num_ccs,
    num_steps,
    record_at_timesteps,
):
    """One simulation config per seed, strategy profile and recommender.

    Every subset of each game's wealthy creators is a strategy profile (who buys).
    All profiles and recommenders share the same seeds (common random numbers), so
    differences in payoffs between profiles are less noisy.
    """

    profiles = []
    for budgets in budget_profiles:
        budgets = {int(c): int(b) for c, b in sorted(budgets.items())}
        wealthy = list(budgets)
        for mask in range(2 ** len(wealthy)):
            buying = [c for k, c in enumerate(wealthy) if mask >> k & 1]
            profiles.append((budgets, buying))

    seeds = np.random.RandomState(97).randint(1000000, size=no_seeds)
    return [
        {
            "num_users": num_users,
            "num_ccs": num_ccs,
            "num_steps": num_steps,
            "record_at_timesteps": list(record_at_timesteps),
            "rs_model": rs_model,
            "alpha": alpha,
            "budgets": budgets,
            "buying_creators": buying,
            "random_seed": int(seed),
        }
        for seed in seeds
        for budgets, buying in profiles
        for rs_model, alpha in recommenders
    ]


def run_experiment(
    path,
    recommenders,
    budget_profiles,
    no_seeds=30,
    num_users=500,
    num_ccs=20,
    num_steps=100,
    record_at_timesteps=(10, 50, 100),
):
    """Runs one experiment and returns two dataframes (creators, seekers):
        creators - one row per (run, recorded timestep, creator)
        seekers  - one row per (run, recorded timestep) with seeker dissatisfaction

    recommenders    - list of (rs_model, alpha), e.g. [('general', 1), ('ExtremePA', 1000)]
    budget_profiles - list of {creator id: budget}, one per game; {} means nobody can buy.
                      Every subset of each game's wealthy creators is simulated as a buyer set.

    The other arguments are the simulation sizes. The results are saved in the file `path`;
    a relative path is taken inside assignment/experiments/ of the repository, wherever the
    notebook runs from. If the file already holds the results of the same experiment (same
    arguments and simulation code), they are loaded instead of simulated again.
    """

    if (
        no_seeds < 1
        or num_users < 1
        or num_ccs < 1
        or num_steps < 1
        or not recommenders
        or not budget_profiles
        or not record_at_timesteps
        or list(record_at_timesteps) != sorted(set(record_at_timesteps))
        or any(t < 1 or t > num_steps for t in record_at_timesteps)
    ):
        raise ValueError(
            "Use positive sizes and sorted, unique recorded rounds within num_steps"
        )
    for profile in budget_profiles:
        if any(
            c != int(c) or not 0 <= c < num_ccs or b != int(b) or b < 0
            for c, b in profile.items()
        ):
            raise ValueError(
                "Budgets require valid creator ids and nonnegative integer amounts"
            )
    if any(rs not in ("general", "ExtremePA") for rs, _ in recommenders):
        raise ValueError(
            "Recommenders must be ('general', alpha) or ('ExtremePA', alpha)"
        )

    # the cache is only valid for the same arguments and the same simulation code
    digest = hashlib.sha256()
    for module in (simulation, model):
        with open(module.__file__, "rb") as source:
            digest.update(source.read())
    params = {
        "simulation_sha256": digest.hexdigest(),
        "recommenders": [list(r) for r in recommenders],
        "budget_profiles": [
            {int(c): int(b) for c, b in p.items()} for p in budget_profiles
        ],
        "no_seeds": no_seeds,
        "num_users": num_users,
        "num_ccs": num_ccs,
        "num_steps": num_steps,
        "record_at_timesteps": list(record_at_timesteps),
    }

    path = str(EXPERIMENTS_DIR / path)  # an absolute path stays as it is
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if cached["params"] == params:
                return cached["creators"], cached["seekers"]
        except Exception:  # any unreadable or stale cache: simulate again
            print(f"Could not read {path}; simulating again")

    configs = _experiment_configs(
        recommenders,
        budget_profiles,
        no_seeds,
        num_users,
        num_ccs,
        num_steps,
        record_at_timesteps,
    )
    print(f"running {len(configs)} simulations ...")
    creators, seekers = _to_frames(configs, [simulation.simulate(c) for c in configs])

    # write to a temporary file first, so an interrupted run never leaves a partial cache
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"params": params, "creators": creators, "seekers": seekers}, f)
    os.replace(path + ".tmp", path)
    return creators, seekers


def payoff_table(expected, rs_model, alpha, budgets):
    """The 2x2 game between two wealthy creators a < b, given as {a: budget, b: budget}:
    {(a_buys, b_buys): (expected utility of a, expected utility of b)},
    built from the output of expected_utilities."""

    a, b = sorted(budgets)
    e = expected[
        (expected["rs_model"] == rs_model)
        & (expected["alpha"] == alpha)
        & expected["budgets"].map(lambda k: k == budgets_key(budgets))
    ]
    u = {(p, c): m for p, c, m in zip(e["buying_creators"], e["creator"], e["mean"])}

    table = {}
    for a_buys in (False, True):
        for b_buys in (False, True):
            profile = tuple(c for c, buys in ((a, a_buys), (b, b_buys)) if buys)
            table[(a_buys, b_buys)] = (u[(profile, a)], u[(profile, b)])
    return table
