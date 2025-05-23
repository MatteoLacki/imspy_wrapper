import pandas as pd
import requests

from dataclasses import dataclass
from functools import partial
from typing import Callable
from typing import Iterable

from cachemir.main import ITERTUPLES
from cachemir.main import SimpleLMDB
from cachemir.serialization import derive_types
from cachemir.serialization import enforce_types


@dataclass
class PredictorWrapper:
    """A (hopefully) common interface to David's models.

    postprocessing should include some scaling.
    """

    db: SimpleLMDB
    server_url: str = "http://localhost:5000/predict_iRTs"
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

    def direct_predict(
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
            """Callback for non-cached values."""
            # NO NEED TO SANITIZE INPUTS: missing_inputs_df is a subset of inputs_df
            predictions = self.direct_predict(missing_inputs_df)
            assert len(predictions) == len(missing_inputs_df)
            output_types = tuple(derive_types(predictions).values())

            def sanitize_outputs(outputs):
                x = enforce_types(outputs, types=output_types)
                return x if len(x) > 1 else x[0]

            yield from zip(
                ITERTUPLES(missing_inputs_df),
                map(sanitize_outputs, ITERTUPLES(predictions)),
            )

        yield from self.db.iter_IO(
            iter_eval=iter_eval,
            inputs_df=inputs_df,
            meta=self.meta,
        )

    def predict(
        self,
        inputs_df: pd.DataFrame,
        return_inputs: bool = False,
    ) -> pd.DataFrame:
        """Predict"""
        if return_inputs:

            def iter_parsed_IO(inputs_df):
                for inputs, outputs in self.iter(inputs_df):
                    row = dict(zip(self.input_cols, inputs))
                    if isinstance(outputs, (float, int, str)):
                        row[self.columns_to_save[0]] = outputs
                    else:
                        for col, output in zip(self.columns_to_save, outputs):
                            row[col] = output
                    yield row

            return pd.DataFrame(iter_parsed_IO(inputs_df))
        else:
            return pd.DataFrame(
                [outputs for _, outputs in self.iter(inputs_df)],
                columns=self.columns_to_save,
            )


DeepGruRtPredictor = partial(
    PredictorWrapper,
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
DeepGruIimPredictor = partial(
    PredictorWrapper,
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
