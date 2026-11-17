from anomalies import anomalies

wc_vocab = ["lq1_1111", "lq3_1111", "phil3_11"]
observables = ["GammaZ", "m_W"]

# If th_predictions is omitted, linear coefficients are computed with
# flavio/wilson at NP_scale and saved automatically before RL training.
aa = anomalies(
    wc_list=wc_vocab,
    EFT="SMEFT",
    Basis="Warsaw",
    NP_scale=1000.0,
    observables=observables,
    measurements=None,  # use flavio's default/combined measurements
    th_predictions=None,
    hyper_para={
        "n_batches": 300,
        "rollouts_per_batch": 25,
        "max_steps": 10,
        "seed": 42,
        "min_step_delta_chi2": 2.0,
        "use_critic": False,
    },
)

print(aa.best)
print(aa.delta_chi2_gt(100.0))
