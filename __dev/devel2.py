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
    input_cols=("sequences",),
)

%%timeit
xx = deep_gru_rt_predictor.predict_compact(irts, return_inputs=True)


xx = list(deep_gru_rt_predictor.iter(irts.iloc[:1000]))
self = deep_gru_rt_predictor




xx = deep_gru_rt_predictor.predict_compact(irts.iloc[:1000])

predictions = deep_gru_rt_predictor.predict(irts.iloc[:1000])
missing_inputs_df = irts.iloc[:1000]

from cachemir.main import ITERTUPLES
from cachemir.serialization import derive_types
from cachemir.serialization import enforce_types
from functools import partial

output_types = tuple(derive_types(predictions).values())
def sanitize_outputs(outputs):
    x = enforce_types(outputs, types=output_types)
    return x if len(x) > 1 else x[0]

predictions = predictions[list(deep_gru_rt_predictor.columns_to_save)]
list(zip(ITERTUPLES(missing_inputs_df), map(sanitize_outputs, ITERTUPLES(predictions))))




xx = deep_gru_rt_predictor.predict_compact(irts)
with deep_gru_rt_predictor.db.open("r") as txn:
    print(len(txn))
# we should modify cachemir to have a d


%%time
xx = deep_gru_rt_predictor.predict(irts.iloc[:1000])

%%time
xx = list(deep_gru_rt_predictor.iter(irts))

inputs, outputs = xx[0]

%%timeit
xx = deep_gru_rt_predictor.predict_compact(irts)

%%timeit
deep_gru_rt_predictor.predict_compact(irts)


cols = [
    *deep_gru_rt_predictor.input_cols,
    *deep_gru_rt_predictor.columns_to_save,
]
with deep_gru_rt_predictor.db.open("r") as txn:
    print(len(txn))

len(irts.sequences.unique())



pd.DataFrame(x)


pd.set_option("display.max_rows", 20)
irts.groupby("path").irt.quantile([0, 0.5, 0.95, 1])
pd.set_option("display.max_rows", 5)


%%time
xx = deep_gru_rt_predictor.predict(irts)

%%time
xx = list(deep_gru_rt_predictor.iter(irts.iloc[:100]))



def scale(xx, new_min=0.0, new_max=60.0):
    return (xx - xx.min()) / (xx.max() - xx.min()) * (new_max - new_min) + new_min


irts["normed_pred_iRT"] = scale(
    irts.pred_iRT, new_min=irts.irt.min(), new_max=irts.irt.max()
)
irts["rt_diff"] = irts.normed_pred_iRT - irts.irt

plot = (
    P.ggplot(data=irts)
    + P.geom_freqpoly(P.aes(x="rt_diff", color="path"), size=1)
    + P.theme_minimal()
)
plot.show()


plot = (
    P.ggplot(data=irts)
    + P.geom_freqpoly(P.aes(x="pred_iRT-irt", color="path"), size=1)
    + P.theme_minimal()
)
plot.show()


plot = P.ggplot(data=irts) + P.geom_freqpoly(P.aes(x="irt", color="path"), size=2)
plot.show()

irts.to_parquet("/home/matteo/tmp/debugging_imspy_rt.parquet")
