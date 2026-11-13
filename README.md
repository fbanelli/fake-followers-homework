# Buying fake followers: luck, strategy and fairness for content creators

On social media, creators compete for followers, and the recommender system decides who gets seen.
Some creators can afford to buy fake followers. Does it pay off? Who loses when they do? And does
the answer depend on how the platform recommends creators?

You will answer these questions by simulating a platform, measuring how buying changes followers,
fairness and seekers' experience, and finding the equilibria of the buy / don't-buy game between
wealthy creators. The simulations are in [assignment/assignment.ipynb](assignment/assignment.ipynb).

> [!IMPORTANT]
> **You do not hand in this repository or the notebook.** Use them as a guide for the simulation
> questions of Game Theory & Control Assignment 2 (*Strategic Classification of Influencers*), which
> you submit as a PDF on Moodle. Put any figure, table or number you use from the notebook in that PDF.

## Setup

1. Install [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/)
   and [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions.
2. Clone the repository and install its environment:

   ```bash
   git clone https://github.com/fbanelli/fake-followers-homework.git
   cd fake-followers-homework
   uv sync
   ```

3. Open the repository folder in VS Code, then open `assignment/assignment.ipynb`.
4. Click **Select Kernel** → **Python Environments**, and choose the repository's `.venv`
   (`.venv/bin/python` on macOS/Linux, `.venv\Scripts\python.exe` on Windows).
   If it is missing, use **Python: Select Interpreter** → **Enter interpreter path** to select it,
   then reopen the kernel picker.
5. Use the play button beside each cell to run cells in order. Complete each TODO before running
   its checks or dependent cells, and write notes below the questions if you like.

## Running time

Defaults use 500 seekers, 20 creators, 100 rounds and 30 shared random seeds. Parts 1, 2 and 4
run 120, 960 and 960 simulations; the first full run takes about a minute. Each part saves its
results in one file in `assignment/experiments/` and reuses it afterwards, as long as the settings
and the simulation code are unchanged; otherwise it simulates again with the same seeds.

## What is in the repository

| file | contents |
|---|---|
| `assignment/assignment.ipynb` | the homework |
| `src/fake_followers/model.py` | seekers, creators, the recommender system and the platform |
| `src/fake_followers/simulation.py` | runs one simulation and saves its results |
| `src/fake_followers/experiments.py` | runs experiments and computes ranks, fairness, payoffs and expected utilities |
| `src/fake_followers/plots.py` | the figures and payoff tables used in the notebook |
| `tests/` | checks for experiment caching and payoff ties (`uv run python -m unittest tests/test_experiments.py`) |

## Credits

```bibtex
@inproceedings{ionescu2023individual,
  author    = {Ionescu, Stefania and Pagan, Nicol{\`o} and Hann{\'a}k, Anik{\'o}},
  title     = {Individual Fairness for Social Media Influencers},
  booktitle = {Complex Networks and Their Applications XI},
  editor    = {Cherifi, Hocine and Mantegna, Rosario Nunzio and Rocha, Luis M. and
               Cherifi, Chantal and Miccich{\`e}, Salvatore},
  series    = {Studies in Computational Intelligence},
  volume    = {1077},
  pages     = {162--175},
  publisher = {Springer},
  address   = {Cham},
  year      = {2023},
  doi       = {10.1007/978-3-031-21127-0_14}
}

@article{ionescu2023luck,
  author  = {Ionescu, Stefania and Hann{\'a}k, Anik{\'o} and Pagan, Nicol{\`o}},
  title   = {The role of luck in the success of social media influencers},
  journal = {Applied Network Science},
  volume  = {8},
  number  = {1},
  pages   = {46},
  year    = {2023},
  doi     = {10.1007/s41109-023-00573-4}
}
```
