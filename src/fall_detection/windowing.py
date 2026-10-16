import numpy as np
from scipy.signal import resample_poly

fs = 50 
pre, post = fs* 1, fs * 2 
center = fs * 10
search = fs * 2

def extract_windows(sig):
    sig_ds = resample_poly(sig, up=25, down=119, axis=0)
    magnitude = np.linalg.norm(sig_ds, axis=1)
    begin, end = center - search, center + search
    peak = np.argmax(magnitude[begin:end]) + begin
    start, finish = peak - pre, peak + post
    if start < 0 or finish > len(sig_ds):
        return None
    return magnitude[start:finish]