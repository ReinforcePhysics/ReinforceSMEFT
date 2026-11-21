import json
import warnings
from pathlib import Path

from reinforcesmeft import ReinforceSMEFT
from wilson import wcxf


# ============================================================
# OPERATOR CATALOGUE
# ============================================================

def _build_catalogue_from_wcxf() -> dict[str, list[str]]:
    """
    Query WCxf for all Wilson coefficients in the SMEFT Warsaw
    dB = de = dmu = dtau = 0 sector.
    """

    try:
        basis_obj = wcxf.Basis["SMEFT", "Warsaw"]

        catalogue = {}

        for sector_name, sector_data in basis_obj.sectors.items():

            if sector_name == "dB=de=dmu=dtau=0":

                wcs = sorted(sector_data.keys())

                if wcs:
                    catalogue[sector_name] = wcs

        return catalogue

    except Exception as exc:

        raise RuntimeError(
            f"Could not load SMEFT Warsaw basis from WCxf: {exc}"
        ) from exc


OPERATOR_CATALOGUE = _build_catalogue_from_wcxf()

operator_vocab = [
    wc
    for ops in OPERATOR_CATALOGUE.values()
    for wc in ops
]


print()
print("=" * 80)
print("WILSON COEFFICIENT VOCABULARY")
print("=" * 80)
print(f"Loaded {len(operator_vocab)} operators")
print()


# ============================================================
# OBSERVABLES
# ============================================================

constraints_ewp = [

    "GammaZ",
    "sigma_had",
    "R_e",
    "R_mu",
    "R_tau",

    "AFB(Z->ee)",
    "AFB(Z->mumu)",
    "AFB(Z->tautau)",

    "A(Z->ee)",
    "A(Z->mumu)",
    "A(Z->tautau)",

    "R_b",
    "R_c",

    "AFB(Z->cc)",

    "A(Z->bb)",
    "A(Z->cc)",

    "GammaW",

    "BR(W->enu)",
    "BR(W->munu)",
    "BR(W->taunu)",

    "R(W->cX)",

    "Rmue(W->lnu)",
    "Rtaue(W->lnu)",
    "Rtaumu(W->lnu)",

    "A(Z->ss)",

    "AFB(Z->bb)",

    "BR(B+->Knunu)",

    "Rtaul(B->Dlnu)",
    "Rtaul(B->D*lnu)",

    "epsp/eps",

    ("<P5p>(B0->K*mumu)", 4, 6),

    ("<dBR/dq2>(Bs->phimumu)", 1, 6),

    "m_W",

    "BR(K+->pinunu)",
    "BR(KL->pinunu)",

    "DeltaM_s",
    "eps_K",

    "BR(Bs->mumu)",
]


# ============================================================
# PATHS
# ============================================================

COEFFICIENTS_PATH = Path("ewp_linear_coefficients_2.json")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OBSERVABLE REPRESENTATION
# ============================================================

def canonical_observable_key(obs):
    """
    Convert a flavio observable specification to the string
    representation expected by the anomalies package.

    Examples
    --------
    "GammaZ"
        -> "GammaZ"

    ("<P5p>(B0->K*mumu)", 4, 6)
        -> "<P5p>(B0->K*mumu)(4, 6)"
    """

    if isinstance(obs, tuple):

        obs_name = obs[0]
        args = obs[1:]

        return (
            f"{obs_name}"
            f"({', '.join(str(x) for x in args)})"
        )

    return str(obs)


# ============================================================
# OLD JSON ALIASES
# ============================================================

#
# Your older coefficient file uses short internal names for
# some binned observables.
#
# We translate those keys ONCE into the representation expected
# by the new anomalies package.
#

JSON_ALIASES = {

    "<P5p>(B0->K*mumu)(4, 6)":
        "P5p46",

    "<dBR/dq2>(Bs->phimumu)(1, 6)":
        "Bsphimumu16",
}


# ============================================================
# LOAD ORIGINAL COEFFICIENT JSON
# ============================================================

with COEFFICIENTS_PATH.open(
    "r",
    encoding="utf-8",
) as handle:

    theory_data = json.load(handle)


if "observables" not in theory_data:

    raise ValueError(
        f"{COEFFICIENTS_PATH} does not contain an "
        "'observables' dictionary."
    )


theory_observables = theory_data["observables"]


# ============================================================
# ADD CANONICAL KEYS FOR OLD ALIASES
# ============================================================

for canonical_key, old_key in JSON_ALIASES.items():

    if (
        canonical_key not in theory_observables
        and old_key in theory_observables
    ):

        theory_observables[canonical_key] = (
            theory_observables[old_key]
        )

        print(
            f"Mapped JSON alias: "
            f"{old_key} -> {canonical_key}"
        )


# ============================================================
# FIND OBSERVABLES ACTUALLY AVAILABLE IN JSON
# ============================================================

available_observables = []

skipped_observables = []


for obs in constraints_ewp:

    json_key = canonical_observable_key(obs)


    # --------------------------------------------------------
    # Missing completely
    # --------------------------------------------------------

    if json_key not in theory_observables:

        skipped_observables.append(
            (
                obs,
                "not present in coefficient JSON",
            )
        )

        continue


    entry = theory_observables[json_key]


    # --------------------------------------------------------
    # Invalid entry
    # --------------------------------------------------------

    if not isinstance(entry, dict):

        skipped_observables.append(
            (
                obs,
                "invalid JSON entry",
            )
        )

        continue


    # --------------------------------------------------------
    # Linearization failed previously
    # --------------------------------------------------------

    if "error" in entry:

        skipped_observables.append(
            (
                obs,
                f"linearization error: "
                f"{entry['error']}",
            )
        )

        continue


    # --------------------------------------------------------
    # Required quantities
    # --------------------------------------------------------

    if "sm_prediction" not in entry:

        skipped_observables.append(
            (
                obs,
                "missing sm_prediction",
            )
        )

        continue


    if "coefficients" not in entry:

        skipped_observables.append(
            (
                obs,
                "missing coefficients",
            )
        )

        continue


    available_observables.append(obs)


# ============================================================
# PRINT OBSERVABLE SUMMARY
# ============================================================

print()
print("=" * 80)
print("OBSERVABLE SELECTION")
print("=" * 80)

print(
    f"Requested observables : "
    f"{len(constraints_ewp)}"
)

print(
    f"Available observables : "
    f"{len(available_observables)}"
)

print(
    f"Skipped observables   : "
    f"{len(skipped_observables)}"
)


if skipped_observables:

    print()
    print("Skipped:")

    for obs, reason in skipped_observables:

        print(
            f"  {obs!s:55s} "
            f"{reason}"
        )


# ============================================================
# SAVE COMPATIBLE JSON TO SCRATCH
# ============================================================

#
# Do NOT modify your original coefficient file.
#
# Instead we write a package-compatible copy to scratch.
#

COMPATIBLE_COEFFICIENTS_PATH = (
    OUTPUT_DIR
    / "ewp_linear_coefficients_2_anomalies.json"
)


with COMPATIBLE_COEFFICIENTS_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        theory_data,
        handle,
        indent=2,
    )


print()
print(
    "Compatible coefficient file:"
)

print(
    COMPATIBLE_COEFFICIENTS_PATH
)


# ============================================================
# SAFETY CHECK
# ============================================================

if not available_observables:

    raise RuntimeError(
        "None of the requested observables are available "
        "in the coefficient JSON."
    )


# ============================================================
# RUN ANOMALIES
# ============================================================

print()
print("=" * 80)
print("STARTING RL TRAINING")
print("=" * 80)
print()


aa = ReinforceSMEFT(

    wc_list=operator_vocab,

    EFT="SMEFT",

    Basis="Warsaw",

    NP_scale=1000.0,

    #
    # IMPORTANT:
    #
    # We retain the original flavio representation here.
    #
    # For example:
    #
    #   ("<P5p>(B0->K*mumu)", 4, 6)
    #
    # rather than P5p46.
    #

    observables=available_observables,

    measurements=None,

    #
    # Use our compatible copy in scratch.
    #
    # This means no linear coefficients are recalculated.
    #

    th_predictions=str(
        COMPATIBLE_COEFFICIENTS_PATH
    ),

    hyper_para={

        "n_batches": 300,

        "rollouts_per_batch": 25,

        "max_steps": 10,

        "seed": 42,
    },
)

# ============================================================
# SAVE TRAINING OUTPUTS
# ============================================================

RESULTS_PATH = OUTPUT_DIR / "results_seed42.json"
SUMMARY_PATH = OUTPUT_DIR / "summary_seed42.json"
SOLUTIONS_PATH = OUTPUT_DIR / "solutions_seed42.json"
METRICS_PATH = OUTPUT_DIR / "metrics_seed42.json"
RAW_SOLUTIONS_PATH = OUTPUT_DIR / "raw_solutions_seed42.json"


# ------------------------------------------------------------
# 1. Complete combined output
# ------------------------------------------------------------

aa.save(
    RESULTS_PATH,
    include_raw=True,
)


# ------------------------------------------------------------
# 2. Summary
# ------------------------------------------------------------

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        aa.summary(),
        handle,
        indent=2,
    )


# ------------------------------------------------------------
# 3. Distinct RL solutions
# ------------------------------------------------------------

with SOLUTIONS_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        aa.dict,
        handle,
        indent=2,
    )


# ------------------------------------------------------------
# 4. Training metrics
# ------------------------------------------------------------

with METRICS_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        aa.metrics,
        handle,
        indent=2,
    )


# ------------------------------------------------------------
# 5. Raw solutions / trajectories
# ------------------------------------------------------------

with RAW_SOLUTIONS_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        aa.raw_solutions,
        handle,
        indent=2,
    )


# ============================================================
# PRINT OUTPUT LOCATIONS
# ============================================================

print()
print("=" * 80)
print("TRAINING OUTPUT SAVED")
print("=" * 80)

print(f"Combined results : {RESULTS_PATH}")
print(f"Summary          : {SUMMARY_PATH}")
print(f"Solutions        : {SOLUTIONS_PATH}")
print(f"Metrics          : {METRICS_PATH}")
print(f"Raw solutions    : {RAW_SOLUTIONS_PATH}")
