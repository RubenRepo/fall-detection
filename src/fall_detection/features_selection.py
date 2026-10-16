import numpy as np

def feature_selection(x, fs):
    segments = {"all": x, "pre": x[:, :fs], "post": x[:, fs + 25:]}
    stats = {"max": np.max, "min": np.min, "mean": np.mean, "std": np.std}
    feats, names = [], []
    for seg_name, seg in segments.items():
        for stat_name, stat in stats.items():
            feats.append(stat(seg, axis=1))
            names.append(f"{seg_name}_{stat_name}")
    return np.column_stack(feats), names