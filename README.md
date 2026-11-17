# ReinforceSMEFT

`reinforcesmeft` is a Python package for reinforcement-learning searches over SMEFT Wilson-coefficient operator sets. It combines a linearized `flavio`/`wilson` observable backend, bounded least-squares coefficient fits, and an RL policy to discover high-quality multi-operator solutions.

## Installation

A normal installation installs all runtime dependencies automatically, including `numpy`, `scipy`, `torch`, `flavio`, `wilson`, and `numdifftools`:

```bash
pip install .
```

or directly from GitHub:

```bash
pip install git+https://github.com/ReinforcePhysics/ReinforceSMEFT.git
```

MPI is not required for the standard API. If you want to use a separate MPI coefficient-generation workflow, install:

```bash
pip install ".[mpi]"
```

## Main API

```python
from reinforcesmeft import ReinforceSMEFT

wc_vocab = [
    "lq1_1111",
    "lq3_1111",
    "phil3_11",
]

observables = [
    "GammaZ",
    "m_W",
    ("<P5p>(B0->K*mumu)", 4, 6),
]

rl = ReinforceSMEFT(
    wc_list=wc_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,          # GeV
    observables=observables,
    measurements=None,        # use flavio defaults
    th_predictions=None,      # compute + save linear theory JSON first
    hyper_para={
        "n_batches": 300,
        "rollouts_per_batch": 25,
        "max_steps": 10,
        "seed": 42,
        "min_step_delta_chi2": 2.0,
        "use_critic": True,
    },
)
```

`wc_list` must be a **Python list of Wilson-coefficient names** defining the RL action vocabulary. Wilson-coefficient values are fitted internally.

## Measurements

If

```python
measurements=None
```

the package obtains the default/combined experimental constraint from `flavio` for each requested observable.

Custom measurements can be supplied as a dictionary:

```python
measurements = {
    "GammaZ": {
        "central": 2.4955,
        "uncertainty": 0.0023,
    },

    "m_W": (
        80.37,
        0.01,
    ),
}
```

Both formats are supported:

```python
"GammaZ": {
    "central": 2.4955,
    "uncertainty": 0.0023,
}
```

and

```python
"GammaZ": (2.4955, 0.0023)
```

For observables not present in the supplied measurement dictionary, the package falls back to the corresponding `flavio` measurement.

For binned observables, use the flavio tuple representation:

```python
measurements = {
    ("<P5p>(B0->K*mumu)", 4, 6): {
        "central": -0.25,
        "uncertainty": 0.10,
    },
}
```

## Theory predictions

If

```python
th_predictions=None
```

the package first computes linearized predictions for the requested observables and Wilson-coefficient vocabulary, writes the corresponding JSON file, and then starts the RL search.

If a precomputed JSON file already exists, provide its path directly:

```python
from reinforcesmeft import ReinforceSMEFT

rl = ReinforceSMEFT(
    wc_list=wc_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,
    observables=observables,
    measurements=None,
    th_predictions="linear_coefficients_all.json",
    hyper_para={
        "n_batches": 300,
        "rollouts_per_batch": 25,
        "max_steps": 10,
        "seed": 42,
    },
)
```

When a file is supplied, coefficient generation is **not run**.

The supplied theory file should be consistent with the requested EFT, basis, NP scale, normalization, observables, and Wilson-coefficient vocabulary.

## NP-scale normalization

The package uses the dimension-six convention

```text
x_i = Lambda^2 C_i
```

where `Lambda = NP_scale` is given in GeV and `C_i` is in `GeV^-2`.

For example:

```text
NP_scale = 1000 GeV  ->  NORMALIZE = 1e6
NP_scale = 2000 GeV  ->  NORMALIZE = 4e6
```

The observable calculator and fitter use the same normalization convention.

## Results and queries

After training,

```python
rl.dict
```

contains the distinct operator supports found by the RL search, ordered by decreasing `delta_chi2`.

Useful queries include:

```python
rl.best

rl.top(10)

rl.delta_chi2_gt(100)

rl.p_value_gt(0.10)

rl.chi2_below(50)

rl.with_n_operators(6)

rl.near_chi2(
    40,
    tolerance=2,
)

rl.near_p_value(
    0.10,
    tolerance=0.01,
)
```

Combined filters are also supported:

```python
rl.filter(
    min_delta_chi2=100,
    min_p_value=0.10,
    n_operators=6,
    contains=["lq1_3323"],
)
```

## Refit an operator support

An arbitrary operator support can be refitted without rerunning the RL search:

```python
rl.refit(
    [
        "lq1_3323",
        "ledq_3333",
        "qd1_1221",
    ]
)
```

## Saving results

Save the standard results:

```python
rl.save(
    "results.json"
)
```

To also include the raw RL solutions/trajectories:

```python
rl.save(
    "results_full.json",
    include_raw=True,
)
```

The saved output contains:

```text
summary
solutions
metrics
raw_solutions    # when include_raw=True
```

For example:

```python
rl.save(
    "/home/user/scratch/reinforcesmeft/results_seed42.json",
    include_raw=True,
)
```

## Typical workflow

```python
from reinforcesmeft import ReinforceSMEFT


rl = ReinforceSMEFT(
    wc_list=operator_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,
    observables=observables,
    measurements=None,
    th_predictions=COEFFICIENTS_PATH,
    hyper_para={
        "n_batches": 100,
        "rollouts_per_batch": 25,
        "max_steps": 10,
        "seed": 42,
    },
)


# Best solution
print(rl.best)


# High-quality solutions
good = rl.filter(
    min_delta_chi2=100,
    min_p_value=0.10,
)


# Save complete run
rl.save(
    "results_seed42.json",
    include_raw=True,
)
```

## Development

For an editable installation:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

Run the tests with:

```bash
pytest
```

## Import

The public API is:

```python
from reinforcesmeft import ReinforceSMEFT
```

The package name is `reinforcesmeft`, while the main user-facing class is `ReinforceSMEFT`.

## Citation

If you use `ReinforceSMEFT` in your research, please cite:

**A Reinforcement Learning Method to Navigate New Physics Interactions**  
[arXiv:2611.21439](https://arxiv.org/abs/2611.21439)

```bibtex
@article{ReinforceSMEFT,
    title   = {A Reinforcement Learning Method to Navigate New Physics Interactions},
    eprint  = {2611.21439},
    archivePrefix = {arXiv},
    primaryClass  = {hep-ph},
    year    = {2026}
}
