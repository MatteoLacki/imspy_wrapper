# If you get a ModuleNotFound error install koinapy with `pip install koinapy`.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotnine as P

from koinapy import Koina

# koinapy only takes the input it requires for the current model.
# if you want to compare multiple models you can use a dataframe wit all columns at the same time.
inputs = pd.DataFrame()
inputs["peptide_sequences"] = np.array(["AAAAAKAR[UNIMOD:7]K"])


# If you are unsure what inputs your model requires run `model.model_inputs`


model = Koina(
    model_name="Prosit_2024_irt_cit",
    server_url="192.168.1.73:8500",
    ssl=False,
)
predictions = model.predict(inputs)


X = pd.read_parquet("/home/matteo/tmp/debugging_imspy_rt.parquet")
inX = X[["sequences"]].rename(columns={"sequences": "peptide_sequences"})
res = model.predict(inX)
X["koina"] = res.irt

X["koina_diff"] = X.koina - X.irt

plot = (
    P.ggplot(data=X)
    + P.geom_freqpoly(P.aes(x="koina_diff", color="path"), size=1)
    + P.theme_minimal()
)
plot.show()

plt.hist()
X.rt_diff / 60
X.koina_diff / X.irt.max()
