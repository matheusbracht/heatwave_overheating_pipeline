
# jos3_runner.py
import numpy as np
import pandas as pd
import copy
from typing import Optional, Dict, Any

try:
    from pythermalcomfort.models import JOS3
    from pythermalcomfort.classes_return import get_attribute_values
    from pythermalcomfort.jos3_functions.parameters import local_clo_typical_ensembles
except Exception as e:
    raise ImportError("pythermalcomfort (>=2.8) is required to run JOS-3. Please install it via pip.") from e

DEFAULT_CLO_KEY = "bra+panty, T-shirt, jeans, socks, sneakers"
DEFAULT_CLO = 0.5  

def build_clo_profile(key: str = DEFAULT_CLO_KEY):
    """Return a 17-segment clothing array for JOS3 given a textual ensemble key."""
    clo_profile = get_attribute_values(
        local_clo_typical_ensembles[key]["local_body_part"]
    )
    return clo_profile

def make_model(
    height=1.7,
    weight=60,
    fat=25,
    age=30,
    sex="male",
    bmr_equation="japanese",
    bsa_equation="fujimoto",
    clo=None,
    clo_key=None,
    **kwargs,
):
    """
    Cria o modelo JOS3 aceitando kwargs opcionais.
    - Se 'clo' for informado, usa diretamente (ex.: 0.5).
    - Se 'clo_key' for informado, você pode mapear para um valor clo (se tiver um dicionário próprio).
    - Quaisquer kwargs extras serão ignorados pelo JOS3 se não forem suportados.
    """
    model = JOS3(
        height=height,
        weight=weight,
        fat=fat,
        age=age,
        sex=sex,
        bmr_equation=bmr_equation,
        bsa_equation=bsa_equation,
        **{k: v for k, v in kwargs.items() if k not in ("clo", "clo_key")}
    )
    # Define clothing
    if clo is not None:
        model.clo = float(clo)
    else:
        # (opcional) se quiser mapear clo_key -> clo, faça aqui.
        # Ex.: if clo_key: model.clo = CLO_PRESETS[clo_key]
        model.clo = DEFAULT_CLO

    return model

def run_jos3_hourly(env: pd.DataFrame,
                    col_ta: str,
                    col_to: str,
                    col_rh: str,
                    ts_col: str = "timeset",
                    days_to_run: Optional[int] = None,
                    dt_seconds: int = 600,
                    model_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Fast hourly JOS-3 loop (clone trick) using arrays (no pandas in the inner loop).

    Parameters
    ----------
    env : DataFrame with at least [ts_col, col_ta, col_to, col_rh] at hourly step.
    col_ta, col_to, col_rh : column names for dry-bulb, operative (or mean radiant prox), and RH [%].
    ts_col : time column (datetime64 or str parseable).
    days_to_run : if provided, truncate to N = days_to_run*24 hours.
    dt_seconds : integration step (default 600s = 10min).
    model_kwargs : kwargs to make_model().

    Returns
    -------
    DataFrame with [ts_col, 't_skin_mean', 't_core', 'w_mean'] (hourly).
    """
    assert col_ta in env.columns and col_to in env.columns and col_rh in env.columns, "Missing env columns"
    df = env[[ts_col, col_ta, col_to, col_rh]].dropna().copy()
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col])
    if days_to_run is not None:
        df = df.iloc[:days_to_run*24].copy()

    ta = df[col_ta].to_numpy(dtype=float)
    to = df[col_to].to_numpy(dtype=float)
    rh = df[col_rh].to_numpy(dtype=float)
    ts = df[ts_col].to_numpy()

    N = len(df)
    if N == 0:
        return pd.DataFrame(columns=[ts_col, "t_skin_mean", "t_core", "w_mean"])

    STEPS_PER_H = 3600 // dt_seconds
    assert 3600 % dt_seconds == 0, "dt_seconds must divide 3600 exactly."

    # Model
    mk = model_kwargs or {}
    model = make_model(**mk)
    sim = model.simulate

    # Pre-alloc outputs
    t_skin = np.empty(N, dtype=np.float32)
    t_core = np.empty(N, dtype=np.float32)
    w_mean = np.empty(N, dtype=np.float32)

    def _grab_last_from_clone(mclone):
        # prefer results() if available
        if hasattr(mclone, "results"):
            r = mclone.results()
            return float(r.t_skin_mean[-1]), float(r.t_cb[-1]), float(r.w_mean[-1])
        d = mclone.dict_results()
        return float(d["t_skin_mean"][-1]), float(d["t_cb"][-1]), float(d["w_mean"][-1])

    for i in range(N):
        # set boundary conditions for this hour
        model.tdb = ta[i]
        model.to  = to[i]
        model.rh  = rh[i]

        # advance 50 minutes without recording (keeps physiology continuous)
        sim(times=STEPS_PER_H-1, dtime=dt_seconds, output=False)

        # take a cheap snapshot to record the final 10 minutes
        mclone = copy.deepcopy(model)
        mclone.simulate(times=1, dtime=dt_seconds, output=True)
        ts_, tc_, w_ = _grab_last_from_clone(mclone)
        # discard clone
        del mclone

        t_skin[i] = ts_
        t_core[i] = tc_
        w_mean[i] = w_

        # final 10 min on main model, without output, to keep continuity
        sim(times=1, dtime=dt_seconds, output=False)

    out = pd.DataFrame({
        ts_col: ts,
        "t_skin_mean": t_skin,
        "t_core": t_core,
        "w_mean": w_mean,
    })
    return out
