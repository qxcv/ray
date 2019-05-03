# Reading, installing, and running the code

**Reading the code.** I have implemented GAIL as a fork of the Ray codebase;
consequently, I have uploaded the entire source code of my Ray fork along with
this assignment (although only the RLlib-related components have been changed).
My experiment code and the code for discriminator training are in the
`python/ray/rllib/examples/gail/` directory. Most of the modifications and
bug-fixes for the TD3 and APE-X implementations are in
`python/ray/rllib/agents/ddpg/` and `python/ray/rllib/optimizers/`,
respectively. To make it clear what has been changed and what existed
previously, I have included a Git diff between my branch and the most recent
ancestor in the official repository that shows all of my changes (using `git
diff d951eb740ffe22c '*.py' '*.yaml'`). The full change history can also be
viewed in the [`gail` branch of my repo on
GitHub](https://github.com/qxcv/ray/tree/gail).

**Installing the code.** My version of Ray and RLlib can be installed by
following the instructions [in the Ray
documentation](https://ray.readthedocs.io/en/latest/installation.html). On my
machines, that meant installing Bazel, creating a Python 3 virtual environment,
changing into the `python` directory, and running `pip install -e '.[rllib]'`.
My code also requires a GPU-enabled copy of Tensorflow, and a recent Git
version of the Sacred command line argument parser. Both can be installed with
`pip install -r tensorflow-gpu git+https://github.com/IDSIA/sacred.git`.

**Running the code.** To train a HalfCheetah-v2 agent for 30 minutes with 16
rollout workers and no GPU, run `python gail.py with
halfcheetah-full-nogpu.yaml`. This will produce a directory named
`data/sacred-runs/<highest-number>/`containing results from the run.
`progress.csv` contains the statistics used to create the plots in the
Experiments section, while `cout.txt` logs all the output from the training
process. I've already included some run directories for the runs included in
the report so as not to break the plotting code.

**Other code.** There are two other code files alongside `gail.py`. One is
`dist_vs_single.py`, which contains the implementations of the three
small-scale neural network training strategies described in the report.
`plots.ipynb` is the Jupyter notebook that I used to create the plots in this
report. Both are independent of the rest of the implementation.
