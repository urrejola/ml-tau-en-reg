import awkward as ak
import glob

files = glob.glob("/data/user/urrejola/future_dataset/taus/*.parquet")

n_signal = 0
n_bg = 0
n_jets_signal = 0
n_jets_bg = 0

for f in files[:5]:  # start small
    data = ak.from_parquet(f)

    # signal definition used in your code
    sig = (data.gen_jet_tau_decaymode != -1)

    n_signal += ak.sum(sig)
    n_bg += ak.sum(~sig)

    n_jets_signal += ak.sum(ak.num(data.reco_cand_p4s[sig], axis=1))
    n_jets_bg += ak.sum(ak.num(data.reco_cand_p4s[~sig], axis=1))

print("events signal:", n_signal)
print("events bg:", n_bg)
print("jets signal:", n_jets_signal)
print("jets bg:", n_jets_bg)
