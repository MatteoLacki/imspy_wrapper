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
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Callable
from typing import Iterable

from imspy_wrapper.main import DeepGruIimPredictor
from imspy_wrapper.main import DeepGruRtPredictor

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
deep_gru_rt_predictor = DeepGruRtPredictor()
deep_gru_rt_predictor.predict(irts, return_inputs=True)
deep_gru_rt_predictor.predict(irts, return_inputs=True)


# rm -rf /home/matteo/tmp/test_ims_0
deep_gru_iim_predictor = DeepGruIimPredictor()
deep_gru_iim_predictor.predict(ions)
deep_gru_iim_predictor.predict(ions)
