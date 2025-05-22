#!/usr/bin/env python3
import click
import pandas as pd

from flask import Flask
from flask import jsonify
from flask import request
from tqdm import tqdm

import functools


@functools.cache
def get_iRT_predictor():
    from imspy.algorithm import (
        DeepChromatographyApex,
        load_deep_retention_time_predictor,
    )
    from imspy.algorithm.utility import load_tokenizer_from_resources

    iRT_predictor = DeepChromatographyApex(
        model=load_deep_retention_time_predictor(),
        tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
        verbose=True,
    )
    return iRT_predictor


@functools.cache
def get_iim_predictor():
    from imspy.simulation.timsim.jobs.simulate_ion_mobilities_and_variance import (
        simulate_ion_mobilities_and_variance,
    )
    from imspy.data.peptide import PeptideSequence
    from imspy.chemistry.utility import calculate_mz

    def predict_iims(inputs_df):
        assert "sequence" in inputs_df
        assert "charge" in inputs_df
        inputs_df["mz"] = [
            calculate_mz(PeptideSequence(sequence).mono_isotopic_mass, charge)
            for sequence, charge in tqdm(
                zip(
                    inputs_df.sequence,
                    inputs_df.charge,
                ),
                total=len(inputs_df),
                desc="Getting monoisotopic m/z",
            )
        ]
        return simulate_ion_mobilities_and_variance(
            ions=inputs_df[["sequence", "charge", "mz"]],
            im_lower=9.0,
            im_upper=100000.0,
            remove_mods=True,
        )

    return predict_iims


@click.command(context_settings={"show_default": True})
@click.argument("host", type=str, default="0.0.0.0")
@click.argument("port", type=int, default=5000)
@click.option("--debug", help="Run in debug mode", is_flag=True)
def serve_david_teschner_models(host, port, debug):
    app = Flask(__name__)

    @app.route("/predict_iRTs", methods=["POST"])
    def predict_iRTs():
        data = request.get_json()
        inputs_df = pd.DataFrame(data["inputs_df"])
        assert "sequences" in inputs_df
        iRT_predictor = get_iRT_predictor()
        predicted_iRTs = dict(
            iRT=iRT_predictor.simulate_separation_times(sequences=inputs_df.sequences)
            .ravel()
            .tolist()
        )
        return jsonify(predicted_iRTs)

    @app.route("/predict_iims", methods=["POST"])
    def predict_iims():
        data = request.get_json()
        inputs_df = pd.DataFrame(data["inputs_df"])
        predict_iims = get_iim_predictor()
        predictions = predict_iims(inputs_df)
        return jsonify(predictions.to_dict(orient="records"))

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    serve_david_teschner_models()
