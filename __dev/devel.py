import pandas as pd

from flask import Flask
from flask import jsonify
from flask import request

import functools


@functools.cache
def get_rt_model():
    from imspy.algorithm import (
        DeepChromatographyApex,
        load_deep_retention_time_predictor,
    )
    from imspy.algorithm.utility import load_tokenizer_from_resources

    return DeepChromatographyApex(
        model=load_deep_retention_time_predictor(),
        tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
        verbose=True,
    )


app = Flask(__name__)


@app.route("/predict_retention_times", methods=["POST"])
def predict_retention_times():
    data = request.get_json()
    rt_model = get_rt_model()
    predicted_rt = rt_model.simulate_separation_times_pandas(
        data=pd.DataFrame(dict(sequence=data["sequences"])),
        batch_size=1024,
        gradient_length=data["gradient_length"],
        decoys_separate=False,
    )
    return jsonify(predicted_rt.to_dict(orient="records"))


if __name__ == "__main__":
    app.run()
