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

pd.set_option("display.max_rows", 4)
pd.set_option("display.max_columns", None)


real_inputs = pd.read_csv("/tmp/real_inputs.csv")
real_inputs = real_inputs[
    ["peptide_sequences", "precursor_charges", "collision_energies"]
]


# sequences = ["PEPTIDE", "PEPTIDEC[UNIMOD:4]PEPTIDE"] * 2

# data = {"sequences": sequences, "gradient_length": 90}
# response = requests.post(
#     "http://localhost:5000/predict_retention_times", json=data, timeout=70
# )
# rts = pd.DataFrame(response.json())

# do we want a separate class for each model?
# likely no, so perhaps simply instantiate them all and instead sequences use inputs_df.

rt_inputs_df = (
    real_inputs[["peptide_sequences"]]
    .copy()
    .rename(columns={"peptide_sequences": "sequences"})
)

import numpy.typing as npt

@dataclass
class PredictorWrapper:
    db: SimpleLMDB
    server_url: str
    input_cols: tuple[str]
    timeout_in_seconds: int = 3_600
    # applied to every inputs_df and results_df
    preprocessing: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    postprocessing: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    meta: tuple[tuple[str, str | float | int]] = (
        ("sofware", "imspy"),
        ("model", "Prosit_2023_intensity_timsTOF"),
    )

    def sanitize_inputs(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.input_cols:
            assert col in inputs_df.columns
        inputs_df = inputs_df[list(self.input_cols)]
        inputs_df = self.preprocessing(inputs_df)
        return inputs_df

    def predict(
        self,
        inputs_df: pd.DataFrame,
        **kwargs,
    ):
        inputs_df = self.sanitize_inputs(inputs_df)
        response = requests.post(
            url=self.server_url,
            json=dict(
                inputs_df=inputs_df.to_dict(orient="records"),
                kwargs=kwargs,
            ),
            timeout=self.timeout_in_seconds,
        )
        results_df = pd.DataFrame(response.json())
        return self.postprocessing(results_df)

    def iter(self, inputs_df: pd.DataFrame, **kwargs):
        inputs_df = self.sanitize_inputs(inputs_df)
        # no need to sanitize it in iter_eval:
        # missing_inputs_df is a subset of inputs_df
        def iter_eval(missing_inputs_df):
            missing_results = self.predict(missing_inputs_df,**kwargs)
            assert len(missing_results) == len(missing_inputs_df)
            return (((missing_inputs_df.sequences.iloc[idx],), missing_results.iloc[[idx]] ) for idx in range(len(missing_results)))
        yield from self.db.iter_IO(
            iter_eval=iter_eval,
            inputs_df=inputs_df,
            meta=self.meta,
        )

    def predict_compact(self, inputs_df: pd.DataFrame) -> list[dict[str, npt.NDArray]]:
        return pd.DataFrame(dict(iRT=[data["iRT"][0] for seq, data in self.iter(inputs_df)]))





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

# rm -rf /home/matteo/tmp/test_rts_0
deep_gru_rt_predictor = PredictorWrapper(
    db=SimpleLMDB(path="/home/matteo/tmp/test_rts_0"),
    server_url="http://localhost:5000/predict_iRTs",
    input_cols=("sequences",),
)
%%timeit
deep_gru_rt_predictor.predict_compact(irts)

%%time
xx = deep_gru_rt_predictor.predict(irts)


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
