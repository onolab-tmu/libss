"""Example of blind source separation with auxiliary-function-based independent vector analysis."""
import os

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

import libss
from libss.separation.utils import callback_eval


# Read wav file
ref_mic = 0
n_src = 3
datadir = os.path.join(".", "examples", "input", "static")
fs = -1
wav_fmt = "src%d_mic%d.wav"

# (n_src, n_mic, n_samples)
premix = [[] for _ in range(n_src)]
for i in range(n_src):
    for j in range(n_src):
        fs, audio = wavfile.read(os.path.join(datadir, wav_fmt % (i, j)))
        premix[i].append(audio)
        if fs < 0:
            raise ValueError("Sampling frequency is not valid.")

premix = np.array(premix)
premix /= np.max(abs(premix))
mix = premix.sum(axis=0)
ref = premix[:, ref_mic, :]

# STFT
n_fft, hop = 4096, 2048
win_a = np.hamming(n_fft)
win_s = pra.transform.compute_synthesis_window(win_a, hop)
engine = pra.transform.STFT(
    n_fft,
    hop,
    analysis_window=win_a,
    synthesis_window=win_s,
    channels=n_src,
)

mix_tf = engine.analysis(mix.T)

# SI-SDRi before BSS
si_sdr, si_sir, si_sar = [], [], []
callback_eval(ref, mix, si_sdr, si_sir, si_sar)
print("SI-SDR:", si_sdr[-1])
print("SI-SIR:", si_sir[-1])

params = {
    "update_demix_filter": "IP1",
    "update_source_model": "Laplace",
    "n_blocks": 1,
    "forget_param": 0.99,
    "n_iter": 2,
    "ref_mic": ref_mic,
}
demix, sep = libss.separation.auxiva_online(mix_tf, **params)
# TODO: implement object-oriented version
# separator = libss.separation.OnlineAuxIVA(mix_tf, **params)
# separator.step()

# Inverse STFT
est = pra.transform.synthesis(sep, n_fft, hop, win=win_s)[n_fft - hop :, :].T
m = np.minimum(ref.shape[1], est.shape[1])

# Evaluate BSS performance
y = callback_eval(ref[:, :m], est[:, :m], si_sdr, si_sir, si_sar)
print("SI-SDR:", si_sdr[-1])
print("SI-SIR:", si_sir[-1])

for s in range(n_src):
    wavfile.write(f"./examples/output/est_online_{s}.wav", fs, y[s])
    wavfile.write(f"./examples/output/mix_online_{s}.wav", fs, mix[s])
