# src/events_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd

Method = Literal["INMET", "Ouzeau"]

@dataclass
class Thresholds:
    """
    Parâmetros dos limiares por método.
    - INMET: usar normals_m (cols: month, tmax_norm) e delta_c (default 5.0)
    - Ouzeau: usar sdeb_c (Sdeb = percentil 97.5 da T̄ diária na baseline)
    """
    method: Method
    normals_m: Optional[pd.DataFrame] = None  # INMET: ['month','tmax_norm']
    delta_c: float = 5.0                      # INMET: offset de 5°C
    sdeb_c: Optional[float] = None            # Ouzeau: Sdeb (P97.5) em °C

# ------------------------- helpers internos -------------------------

def _daily_excess_inmet(daily_tmax: pd.DataFrame,
                        normals_m: pd.DataFrame,
                        delta_c: float) -> pd.DataFrame:
    """
    Excedente diário para INMET:
      threshold = normal_mensal(Tmax) + delta_c
      excess    = max(Tmax - threshold, 0)
    Aceita normals_m com colunas 'month' e uma das:
      ['tmax_norm','normal_tmax_c','tmax_mean','tmax_clim','tmax']
    """
    dm = daily_tmax[["timeset", "tmax_c"]].dropna().copy()
    dm["month"] = dm["timeset"].dt.month

    # detectar coluna da normal
    candidates = ["tmax_norm", "normal_tmax_c", "tmax_mean", "tmax_clim", "tmax"]
    norm_col = next((c for c in candidates if c in normals_m.columns), None)
    if norm_col is None:
        raise ValueError(f"normals_m precisa ter 'month' e uma das {candidates}. Recebi: {list(normals_m.columns)}")

    nm = normals_m.rename(columns={norm_col: "norm"})[["month", "norm"]]
    dm = dm.merge(nm, on="month", how="left")
    dm["threshold"] = dm["norm"] + float(delta_c)
    dm["excess_c"]  = (dm["tmax_c"] - dm["threshold"]).clip(lower=0)
    return dm[["timeset", "excess_c", "threshold", "tmax_c"]]


def _daily_excess_ouzeau(daily_tmean: pd.DataFrame,
                         sdeb_c: float) -> pd.DataFrame:
    """
    Excedente diário para Ouzeau (Sdeb = P97.5 da T̄ diária):
      threshold = sdeb_c
      excess    = max(T̄ - sdeb_c, 0)
    Espera: daily_tmean[ 'timeset','tmean_c' ]
    Retorna: ['timeset','excess_c','threshold','tmean_c']
    """
    dm = daily_tmean[["timeset", "tmean_c"]].dropna().copy()
    dm["threshold"] = float(sdeb_c)
    dm["excess_c"] = (dm["tmean_c"] - dm["threshold"]).clip(lower=0)
    return dm[["timeset", "excess_c", "threshold", "tmean_c"]]

# ------------------------- API pública -------------------------

def compute_event_metrics(
    events: pd.DataFrame,
    daily_tmean: pd.DataFrame,
    daily_tmax: pd.DataFrame,
    thr: Thresholds,
) -> pd.DataFrame:
    """
    Adiciona métricas padronizadas aos eventos:
      - intensity_c  : máxima T diária (°C) durante a onda
      - duration_d   : duração (dias) [se ausente, é inferida de start/end]
      - severity_cday: soma dos excedentes (°C·dia) acima do limiar do método
                       (INMET: excedente de Tmax vs normal+5; Ouzeau: excedente de T̄ vs Sdeb)

    Parâmetros
    ----------
    events      : DataFrame com colunas ['start','end', ...]
    daily_tmean : DataFrame diário com ['timeset','tmean_c']
    daily_tmax  : DataFrame diário com ['timeset','tmax_c']
    thr         : Thresholds(method="INMET", normals_m=..., delta_c=5.0)
                  ou Thresholds(method="Ouzeau", sdeb_c=...)

    Retorno
    -------
    DataFrame de eventos com colunas adicionais:
      ['intensity_c','duration_d','severity_cday']
    """
    ev = events.copy()

    # 1) duration_d (se não existir)
    if "duration_d" not in ev.columns:
        ev["duration_d"] = (
            ev["end"].dt.normalize() - ev["start"].dt.normalize()
        ).dt.days + 1

    # 2) intensity_c = max(T diária) no período do evento
    tmax = daily_tmax.set_index("timeset").sort_index()
    if "tmax_c" not in tmax.columns:
        raise ValueError("daily_tmax deve conter a coluna 'tmax_c'.")
    ev["intensity_c"] = 0.0
    for i, r in ev.iterrows():
        days = pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D")
        vals = tmax.reindex(days)["tmax_c"]
        ev.at[i, "intensity_c"] = float(vals.max()) if vals.notna().any() else 0.0

    # 3) severity_cday = soma dos excedentes diários conforme método
    if thr.method == "INMET":
        if thr.normals_m is None:
            raise ValueError("Para INMET, forneça normals_m (cols: ['month','tmax_norm']).")
        dex = _daily_excess_inmet(daily_tmax, thr.normals_m, thr.delta_c).set_index("timeset")
    elif thr.method == "Ouzeau":
        if thr.sdeb_c is None:
            raise ValueError("Para Ouzeau, forneça sdeb_c (Sdeb = P97.5).")
        dex = _daily_excess_ouzeau(daily_tmean, thr.sdeb_c).set_index("timeset")
    else:
        raise ValueError(f"Método desconhecido: {thr.method}")

    ev["severity_cday"] = 0.0
    for i, r in ev.iterrows():
        days = pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D")
        vals = dex.reindex(days)["excess_c"].fillna(0.0)
        ev.at[i, "severity_cday"] = float(vals.sum())

    return ev

def expand_event_metrics_to_timeseries(
    events_with_metrics: pd.DataFrame,
    full_timeseries: pd.DataFrame,
    method_label: str,
) -> pd.DataFrame:
    """
    Propaga métricas (intensity/duration/severity) por dia para a série horária.
    Cria colunas prefixadas com o método (upper-case do label fornecido), por ex.:
      INMET_hw_id, INMET_intensity_c, INMET_duration_d, INMET_severity_cday
      OUZ_hw_id,   OUZ_intensity_c,   OUZ_duration_d,   OUZ_severity_cday

    Parâmetros
    ----------
    events_with_metrics : DataFrame de eventos já com ['hw_id','intensity_c','duration_d','severity_cday']
    full_timeseries     : DataFrame horário com coluna 'timeset'
    method_label        : str para prefixo (ex.: "INMET" ou "OUZ")

    Retorno
    -------
    DataFrame horário com as colunas adicionadas (por dia).
    """
    ev = events_with_metrics.copy()
    cols = ["hw_id", "duration_d", "intensity_c", "severity_cday", "method"]

    # Mapeia cada evento para os dias cobertos
    rows = []
    for _, r in ev.iterrows():
        days = pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D")
        tmp = pd.DataFrame({"date": days})
        for c in cols:
            tmp[c] = r[c]
        rows.append(tmp)

    mapper = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date"] + cols)

    out = full_timeseries.copy()
    out["date"] = out["timeset"].dt.normalize()

    pref = method_label.upper()
    m = mapper.add_prefix(f"{pref}_").rename(columns={f"{pref}_date": "date"})
    out = out.merge(m, on="date", how="left")

    return out
