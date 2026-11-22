"""Trial-local Figshare preprocessing and source-only training data."""
import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

FS = 500
EEG_CHANNELS = ["F3", "FZ", "FC1", "FCZ", "C1", "CZ", "CP1", "CPZ", "FC5", "FC3", "C5", "C3", "CP5",
                "CP3", "P3", "PZ", "F4", "FC2", "FC4", "FC6", "C2", "C4", "CP2", "CP4", "C6", "CP6", "P4", "P5"]


def labels_and_stability(codes, state, window):
    codes = np.asarray(codes).reshape(-1)
    if state not in {"static", "motion"} or window <= 0:
        raise ValueError("Invalid state/window")
    if not set(np.unique(codes)) <= {0, 400, 401, 402, 403, 404}:
        raise ValueError("Unknown task code")
    labels = np.full(codes.size, -1, dtype=np.int64)
    labels[np.isin(codes, [401, 402])] = 0 if state == "static" else 1
    labels[np.isin(codes, [403, 404])] = 3 if state == "static" else 2
    stable = np.zeros(codes.size, dtype=bool)
    # Require every sample of the current window to have the SAME stable raw code.
    for code in (402, 404):
        counts = np.r_[0, np.cumsum(codes == code)]
        if codes.size >= window:
            stable[window - 1:] |= (counts[window:] - counts[:-window]) == window
    return labels, stable


def causal_preprocess(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 28 or not np.isfinite(x).all():
        raise ValueError("Expected finite 28-channel EEG")
    b, a = iirnotch(60, 30, fs=FS)
    sos = np.vstack([tf2sos(b, a), butter(4, [1, 40], btype="bandpass", fs=FS, output="sos")])
    y = sosfilt(sos, x, axis=-1)
    y -= y.mean(axis=0, keepdims=True)
    return np.asarray(y, dtype=np.float32)


def source_trials(archive, subject):
    """Read only this subject's S1/S2 MAT members, in protocol order."""
    if subject not in {f"A{i:02d}" for i in range(1, 6)}:
        raise ValueError("Expected subject A01 through A05")
    records = {}
    with zipfile.ZipFile(archive) as z:
        files = [n for n in z.namelist() if n.endswith(".mat") and "__MACOSX" not in n
                 and not Path(n).name.startswith("._")]
        members = {Path(n).stem: n for n in files}
        if len(files) != len(members):
            raise ValueError("Duplicate trial identifiers")
        for session in (1, 2):
            last = 13 if session == 1 and subject in {"A01", "A04"} else 14
            for number in range(1, last + 1):
                state = "static" if number % 2 else "motion"
                trial = f"{subject}_session{session}_openloop_{number}_{state}"
                obj = loadmat(io.BytesIO(z.read(members[trial])), simplify_cells=True)["session"]
                raw = np.asarray(obj["data_EEG"])
                codes = np.asarray(obj["task_EEG"]).reshape(-1)
                if raw.shape != (32, 27750) or len(codes) != raw.shape[1]:
                    raise ValueError(f"Unexpected trial dimensions in {trial}")
                boundaries = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
                runs = list(zip(codes[boundaries[:-1]], np.diff(boundaries)))
                if runs != [(0,250), (400,7500), (401,2000), (402,3500),
                            (403,2000), (404,7000), (401,2000), (402,3500)]:
                    raise ValueError(f"Unexpected cue timeline in {trial}")
                labels, stable = labels_and_stability(codes, state, 1000)
                records[trial] = dict(x=causal_preprocess(raw[:28]), labels=labels, stable=stable)
    return records


def normalize_source(records):
    """Fit per-channel population statistics on exactly these source trials."""
    total = np.zeros(28)
    squares = np.zeros(28)
    count = 0
    for record in records.values():
        x = record["x"].astype(np.float64)
        total += x.sum(axis=1)
        squares += (x*x).sum(axis=1)
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-8))
    for record in records.values():
        record["x"] = (record["x"] - mean.astype("float32")[:, None]) / std.astype("float32")[:, None]
    return dict(mean=mean.tolist(), std=std.tolist(), fit_ids=list(records), count=count)
