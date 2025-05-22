"""
%load_ext autoreload
%autoreload 2
"""
import pandas as pd
import plotnine as P
import requests

import matplotlib.pyplot as plt

from cachemir.main import SimpleLMDB
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Callable
from typing import Iterable

from imspy_wrapper.main import PredictorWrapper

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)

real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]

rt_inputs_df = (
    real_inputs[["peptide_sequences"]]
    .copy()
    .rename(columns={"peptide_sequences": "sequences"})
)

# TODO: generalize this to not use iRT in predict_compact
# https://github.com/wilhelm-lab/dlomix/blob/main/example_dataset/proteomTools_train.csv


def addcol(df, **kwargs):
    for col, val in kwargs.items():
        df[col] = val
    return df


irts = pd.concat(
    (
        addcol(pd.read_csv(p).rename(columns={"sequence": "sequences"}), path=p.stem)
        for p in Path(
            "/home/matteo/Projects/koina/dlomix-resources/example_datasets/RetentionTime"
        ).glob("*.csv")
    ),
    ignore_index=True,
)

# rm -rf /home/matteo/tmp/test_rts_1
deep_gru_rt_predictor = PredictorWrapper(
    db=SimpleLMDB(path="/home/matteo/tmp/test_rts_1"),
    server_url="http://localhost:5000/predict_iRTs",
    preprocessing=lambda x: x,
    postprocessing=lambda x: x,
    input_types=(("sequences", str),),
    columns_to_save=("iRT",),
    meta=(
        ("sofware", "imspy"),
        ("model", "Prosit_2023_intensity_timsTOF"),
    ),
)


# deep_gru_iim_predictor = PredictorWrapper(
#     db=SimpleLMDB(path="/home/matteo/tmp/test_ims_0"),
#     server_url="http://localhost:5000/predict_iims",
#     preprocessing=lambda x: x,
#     postprocessing=lambda x: x,
#     input_types=(("sequences", str),),
#     columns_to_save=("iRT",),
#     meta=(
#         ("sofware", "imspy"),
#         ("model", "Prosit_2023_intensity_timsTOF"),
#     ),
# )


xx = deep_gru_rt_predictor.predict(irts, return_inputs=True)
xx = deep_gru_rt_predictor.direct_predict(irts, return_inputs=True)

from imspy.simulation.timsim.jobs.simulate_ion_mobilities_and_variance import (
    simulate_ion_mobilities_and_variance,
)
from imspy.data.peptide import PeptideSequence
from imspy.chemistry.utility import calculate_mz


ions = real_inputs.rename(
    columns={"peptide_sequences": "sequence", "precursor_charges": "charge"}
)
ions["mz"] = [
    calculate_mz(PeptideSequence(sequence).mono_isotopic_mass, charge)
    for sequence, charge in zip(ions.sequence, ions.charge)
]
del ions["collision_energies"]

# carbamidomethylation is a common modification that is annotated in the UNIMOD database


# create a peptide sequence object, might contain modifications


deep_gru_iim_predictor = PredictorWrapper(
    db=SimpleLMDB(path="/home/matteo/tmp/test_ims_0"),
    server_url="http://localhost:5000/predict_iims",
    preprocessing=lambda x: x,
    postprocessing=lambda x: x,
    input_types=(
        ("sequence", str),
        ("charge", str),
    ),
    columns_to_save=(
        "inv_mobility_gru_predictor",
        "inv_mobility_gru_predictor_std",
    ),
    meta=(
        ("sofware", "imspy"),
        ("model", "deep_iim_mean_and_std_predictor"),
    ),
)
