"""
%load_ext autoreload
%autoreload 2
"""
import pandas as pd
import requests

from typing import Iterable
from typing import Iterator


from pathlib import Path

sequences = ["PEPTIDE", "PEPTIDEC[UNIMOD:4]PEPTIDE"] * 2
sequences = pd.Series(sequences)

list(sequences.to_numpy())


# data = {"sequences": sequences}
# response = requests.post(
#     "http://localhost:5000/predict_retention_times", json=data, timeout=70
# )
# results = response.json()


# OK, now we need a wrapper around it.
from cachemir.main import get_index_and_stats


from cachemir.main import MemoizedOutput
from dataclasses import dataclass


@dataclass
class DeepGruPredictorWrapper:
    model_name: str = "deep_gru_predictor"
    server_url: str = "http://localhost:5000/predict_retention_times"
    cache_path: Path | str | None = None
    input_colums: tuple[str] = ("sequences",)
    timeout_in_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)

    def predict(self, sequences: Iterable[str]) -> pd.DataFrame:
        """Predict Retention Time from a sequence.

        Arguments:
            sequences (Iterable[str]): Sequences for which to predict retention times.

        Returns:
            pd.DataFrame: A data frame with column corresponding to retention times.
        """
        results = requests.post(
            self.server_url,
            json={"sequences": list(sequences)},
            timeout=self.timeout_in_seconds,
        ).json()
        results_df = pd.DataFrame(results, copy=False)
        assert len(results_df) == len(sequences)
        return results_df

    __call__ = predict

    def iter_predict(self, sequences: Iterable[float]) -> Iterator[MemoizedOutput]:
        predicted_rts = self.predict(sequences)
        for sequence, rowNo in zip(sequences, range(len(predicted_rts))):
            yield MemoizedOutput(
                input=(sequence,),
                stats=(),
                data=predicted_rts.iloc[[rowNo]].reset_index(drop=True),
            )

    def get_index_and_stats(
        self,
        sequences: Iterable[str],
        cache_path: Path | str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if cache_path is None:
            assert self.cache_path is not None
            cache_path = self.cache_path
        cache_path = Path(cache_path)

        inputs_df = pd.DataFrame(dict(sequences=sequences), copy=False)
        index_and_stats, raw_data = get_index_and_stats(
            path=cache_path,
            inputs_df=inputs_df,
            results_iter=self.iter_predict,
            input_types=dict(sequences=str),
            stats_types={},
            verbose=verbose,
        )
        return index_and_stats, raw_data


deep_gru_predictor_wrapper = DeepGruPredictorWrapper(
    cache_path="/home/matteo/tmp/testrt1"
)
# predicted_rts = deep_gru_predictor_wrapper.predict(sequences=sequences)
deep_gru_predictor_wrapper.get_index_and_stats(sequences, verbose=True)

# HERE: use only LMDB and that's it.
# Cachemir: problem when inputs were not deduplicated.
