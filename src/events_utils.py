import pandas as pd

def level_from_duration(duration_days: int) -> str:
    """Mapa padrão de nível por duração (regra INMET/WMO que você usa)."""
    if duration_days >= 5: 
        return "Red"
    if duration_days >= 3: 
        return "Orange"
    if duration_days >= 2: 
        return "Yellow"
    return ""

def standardize_events(
    events: pd.DataFrame,
    *,
    site_id: str,
    method: str,
    method_version: str,
    baseline: tuple[str, str] | None = None,
    threshold_info: dict | None = None,
    add_level_by_duration: bool = False,
) -> pd.DataFrame:
    """
    Garante um schema padronizado para eventos de qualquer método.
    Espera que `events` contenha ao menos: start, end, duration_d, peak_c.
    """
    required = {"start", "end", "duration_d", "peak_c"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Eventos do método {method} não possuem colunas: {missing}")

    df = events.copy()
    # Garantir tipos
    df["start"] = pd.to_datetime(df["start"], utc=False)
    df["end"] = pd.to_datetime(df["end"], utc=False)
    df["duration_d"] = df["duration_d"].astype(int)
    df["peak_c"] = pd.to_numeric(df["peak_c"], errors="coerce")

    # Campos fixos
    df["site_id"] = site_id
    df["method"] = method
    df["method_version"] = method_version

    if baseline is not None:
        df["baseline_start"] = pd.to_datetime(baseline[0]).date()
        df["baseline_end"] = pd.to_datetime(baseline[1]).date()
    else:
        df["baseline_start"] = pd.NaT
        df["baseline_end"] = pd.NaT

    if threshold_info is None:
        threshold_info = {}
    # Armazena como dict; se for salvar em parquet/csv, pode serializar para JSON depois
    df["threshold_info"] = [threshold_info] * len(df)

    # Level: para INMET é obrigatório (pela sua regra). Para Ouzeau, deixe optativo.
    if "level" not in df.columns:
        df["level"] = ""

    if add_level_by_duration:
        df["level"] = df["duration_d"].map(level_from_duration)

    # Ordenação determinística
    df = df.sort_values(["start", "end", "peak_c"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True)
    return df

def attach_event_id(events: pd.DataFrame, *, site_id: str, method: str) -> pd.DataFrame:
    """
    Cria IDs estáveis do tipo:
      {site_id}-{method}-{start_yyyymmdd}-{seq_ano}
    onde seq_ano é o índice do evento dentro do ano da data de início, após ordenação determinística.
    """
    df = events.copy()

    # Ordene antes de criar IDs (caso não esteja ordenado)
    df = df.sort_values(["start", "end", "peak_c"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True)

    df["year"] = df["start"].dt.year
    # contador por ano
    df["seq_ano"] = df.groupby("year").cumcount() + 1
    df["start_ymd"] = df["start"].dt.strftime("%Y%m%d")
    df["hw_id"] = (
        f"{site_id}-{method}-"
        + df["start_ymd"]
        + "-"
        + df["seq_ano"].map(lambda x: f"{x:03d}")
    )
    return df.drop(columns=["year","seq_ano","start_ymd"])

def flags_from_events(events: pd.DataFrame, timeline: pd.DataFrame, method: str, col_name: str) -> pd.DataFrame:
    tl = timeline[["timeset"]].copy()
    tl["timeset"] = pd.to_datetime(tl["timeset"], errors="coerce")
    if getattr(tl["timeset"].dt, "tz", None) is not None:
        tl["timeset"] = tl["timeset"].dt.tz_localize(None)

    evm = events.query("method == @method").copy()
    for c in ["start", "end"]:
        evm[c] = pd.to_datetime(evm[c], errors="coerce")
        if getattr(evm[c].dt, "tz", None) is not None:
            evm[c] = evm[c].dt.tz_localize(None)

    tl[col_name] = False
    for _, row in evm.iterrows():
        mask = (tl["timeset"] >= row["start"]) & (tl["timeset"] <= row["end"])
        tl.loc[mask, col_name] = True
    return tl