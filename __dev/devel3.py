from imspy.algorithm import (
    DeepPeptideIonMobilityApex,
    DeepChromatographyApex,
    load_deep_ccs_predictor,
    load_deep_retention_time_predictor,
)
from imspy.algorithm.utility import load_tokenizer_from_resources
from imspy.chemistry.mobility import one_over_k0_to_ccs

# some example peptide sequences
sequences = ["PEPTIDE", "PEPTIDEC[UNIMOD:4]PEPTIDE"]
mz_values = [784.58, 1423.72]
charges = [1, 2]

# the retention time predictor model
rt_predictor = DeepChromatographyApex(
    load_deep_retention_time_predictor(),
    load_tokenizer_from_resources("tokenizer-ptm"),
    verbose=True,
)

# predict retention times for peptide sequences
predicted_rt = rt_predictor.simulate_separation_times(sequences=sequences)

# the ion mobility predictor model
im_predictor = DeepPeptideIonMobilityApex(
    load_deep_ccs_predictor(), load_tokenizer_from_resources("tokenizer-ptm")
)

# predict ion mobilities for peptide sequences and translate them to collision cross sections
predicted_inverse_mobility = im_predictor.simulate_ion_mobilities(
    sequences=sequences, charges=charges, mz=mz_values
)

# stds?
