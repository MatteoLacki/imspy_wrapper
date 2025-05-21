import pandas as pd
import requests

from dataclasses import dataclass
from typing import Callable
from typing import Iterable

from cachemir.main import SimpleLMDB


@dataclass
class PredictorWrapper:
    """A (hopefully) common interface to David's models."""

    db: SimpleLMDB
    server_url: str
    input_cols: tuple[str]
    timeout_in_seconds: int = 3_600
    # applied to every inputs_df and results_df
    preprocessing: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    postprocessing: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    input_types: tuple[tuple[str, type]] = (("sequences", str),)
    columns_to_save: tuple[str, ...] = ("iRT",)
    meta: tuple[tuple[str, str | float | int]] = (
        ("sofware", "imspy"),
        ("model", "Prosit_2023_intensity_timsTOF"),
    )

    def __post_init__(self):
        self.input_cols = [col for col, _ in self.input_types]

    def sanitize_inputs(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(inputs_df, pd.DataFrame)
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
            json=dict(inputs_df=inputs_df.to_dict(orient="records")),
            timeout=self.timeout_in_seconds,
        )
        results_df = pd.DataFrame(response.json())
        return self.postprocessing(results_df)

    def iter(self, inputs_df: pd.DataFrame):
        inputs_df = self.sanitize_inputs(inputs_df)

        def iter_eval(missing_inputs_df):
            """Callback when missing values observed."""
            # NO NEED TO SANITIZE
            # missing_inputs_df is a subset of inputs_df
            predictions = self.predict(missing_inputs_df)
            assert len(predictions) == len(missing_inputs_df)
            return (
                (
                    tuple(missing_inputs_df[self.input_cols].iloc[idx]),
                    predictions.iloc[[idx]],
                )
                for idx in range(len(missing_inputs_df))
            )

        yield from self.db.iter_IO(
            iter_eval=iter_eval,
            inputs_df=inputs_df,
            meta=self.meta,
        )

    def predict_compact(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                **dict(zip(self.input_cols, inputs)),
                **{col: output[0] for col, output in outputs.items()},
            }
            for inputs, outputs in self.iter(inputs_df)
        )
