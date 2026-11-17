# anomalies

`anomalies` is a Python package for reinforcement-learning searches over EFT Wilson-coefficient operator sets. It combines a linearized `flavio`/`wilson` observable backend, bounded least-squares coefficient fits, and the RL policy used to discover high-quality multi-operator solutions.

## Installation

A normal installation installs all runtime dependencies automatically, including `numpy`, `scipy`, `torch`, `flavio`, `wilson`, and `numdifftools`:

```bash
pip install .
```

or directly from GitHub:

```bash
pip install git+https://github.com/USERNAME/anomalies.git
```

MPI is not required for the standard API. If you want to build a separate MPI coefficient-generation workflow, install:

```bash
pip install ".[mpi]"
```

## Main API

```python
from anomalies import anomalies

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

aa = anomalies(
    wc_list=wc_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,          # GeV
    observables=observables,
    measurements=None,       # use flavio defaults
    th_predictions=None,     # compute + save linear theory JSON first
    hyper_para={
        "n_batches": 300,
        "rollouts_per_batch": 25,
        "max_steps": 10,
        "seed": 42,
        "min_step_delta_chi2": 2.0,
        "use_critic": False,
    },
)
```

`wc_list` must be a **Python list of WC names** defining the RL action vocabulary. Wilson-coefficient values are fitted internally.

### Measurements

If `measurements=None`, the package obtains the default/combined experimental constraint from `flavio` for every requested observable. You can override any or all measurements:

```python
measurements = {
    "GammaZ": {"central": 2.4955, "uncertainty": 0.0023},
    "m_W": (80.37, 0.01),
}
```

For observables not present in the supplied mapping, the package falls back to `flavio`.

### Theory predictions

If `th_predictions=None`, the package first computes linearized predictions for exactly the requested observables and WC vocabulary, writes a JSON file in the current directory, and then starts RL training.

If you already have a JSON file, provide it:

```python
aa = anomalies(
    wc_list=wc_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,
    observables=observables,
    measurements=None,
    th_predictions="ewp_linear_coefficients_all.json",
    hyper_para={...},
)
```

When a file is supplied, coefficient generation is **not run**. The package checks the EFT, basis, NP scale, normalization, requested observables, and requested WC vocabulary before training.

## NP-scale normalization

The package uses the dimension-six convention

```text
x_i = Lambda^2 C_i
```

where `Lambda = NP_scale` in GeV and `C_i` is in GeV^-2. Hence:

```text
NP_scale = 1000 GeV  ->  NORMALIZE = 1e6
NP_scale = 2000 GeV  ->  NORMALIZE = 4e6
```

The observable calculator and fitter use the same normalization. A supplied theory JSON with a normalization inconsistent with `NP_scale` is rejected.

## Results and queries

`aa.dict` contains distinct operator supports sorted by decreasing `delta_chi2`.

```python
aa.best
aa.top(10)
aa.delta_chi2_gt(100)
aa.p_value_gt(0.10)
aa.chi2_below(50)
aa.with_n_operators(6)
aa.near_chi2(40, tolerance=2)
aa.near_p_value(0.10, tolerance=0.01)
```

Combined filters are supported:

```python
aa.filter(
    min_delta_chi2=100,
    min_p_value=0.10,
    n_operators=6,
    contains=["lq1_3323"],
)
```

Refit an arbitrary support without rerunning RL:

```python
aa.refit(["lq1_3323", "ledq_3333", "qd1_1221"])
```

Save results:

```python
aa.save("solutions.json")
aa.save("solutions_full.json", include_raw=True)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
