"""Check the real notebook pipeline on synthetic 500 Hz WFDB records."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preprocessing_cli", ROOT/"preprocessing/run_preprocessing.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", *[f"V{i}" for i in range(1, 7)]]


def test_actual_notebook_50hz_build_from_raw_and_reuse(tmp_path):
    rawdir = tmp_path/"raw"
    records = rawdir/"records500"
    records.mkdir(parents=True)
    time = np.arange(5000)/500
    # Contains frequency content above the previous 40 Hz cut-off.
    signal = np.stack([.3*np.sin(2*np.pi*10*time+i/12)+.2*np.sin(2*np.pi*45*time)
                       for i in range(12)], axis=1)
    rows = []
    for i,fold in enumerate([1,9,10],1):
        name = f"record{i}"
        wfdb.wrsamp(name, fs=500, units=["mV"]*12, sig_name=LEADS,
                    p_signal=signal, fmt=["16"]*12, write_dir=str(records))
        rows.append(dict(ecg_id=i,patient_id=i,strat_fold=fold,scp_codes="{'NORM': 100.0}",
                         filename_hr=f"records500/{name}"))
    pd.DataFrame(rows).to_csv(rawdir/"ptbxl_database.csv",index=False)
    pd.DataFrame(dict(diagnostic=[1],diagnostic_class=["NORM"]),index=["NORM"]).to_csv(rawdir/"scp_statements.csv")
    output = tmp_path/"new_50hz"
    manifest = cli.run(data_dir=rawdir,output_dir=output,progress_every=1)
    assert manifest["config"]["bandpass_high"] == 50.
    assert manifest["split_policy"]["patient_overlap_checked"]
    actual = np.load(output/"signals_train.npy")[0]
    raw, _ = wfdb.rdsamp(str(records/"record1"))
    resampled = resample_poly(raw.T.astype(np.float32),1,2,axis=1,padtype="line").astype(np.float32)
    def filtered(high):
        x = sosfiltfilt(butter(4,[.67,high],btype="bandpass",fs=250,output="sos"),resampled,axis=1).astype(np.float32)
        return ((x-x.mean(1,keepdims=True,dtype=np.float64))/x.std(1,keepdims=True,dtype=np.float64)).astype(np.float32)
    assert actual.shape == (12,2500)
    assert np.allclose(actual,filtered(50),atol=1e-6)
    assert not np.allclose(actual,filtered(40),atol=1e-3)
    before = hashlib.sha256((output/"signals_train.npy").read_bytes()).hexdigest()
    reused = cli.run(data_dir=rawdir,output_dir=output,progress_every=1)
    assert reused["generated_at_utc"] == manifest["generated_at_utc"]
    assert before == hashlib.sha256((output/"signals_train.npy").read_bytes()).hexdigest()
    # Mismatched output metadata is rejected, not overwritten.
    manifest["config"]["bandpass_high"] = 40.
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Settings differ"):
        cli.run(data_dir=rawdir,output_dir=output,progress_every=1)
    assert before == hashlib.sha256((output/"signals_train.npy").read_bytes()).hexdigest()
