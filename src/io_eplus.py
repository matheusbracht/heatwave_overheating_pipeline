from __future__ import annotations
from pathlib import Path
import re
from typing import Optional, Literal, Dict, Tuple, List, Union
import pandas as pd
from dateutil import tz

# ------------------------------------------------------------
# Parsing helpers (mantém os seus)
# ------------------------------------------------------------

HDR_RE = re.compile(
    r"^(?P<zone>[^:]+):\s*(?P<var>.+?)\s*\[(?P<unit>[^\]]*)\]\((?P<freq>[^)]*)\)$"
)
HDR_SCHEDULE_RE = re.compile(
    r"^(?P<zone>[^:]+):\s*Schedule Value\s*\[\]\((?P<freq>[^)]*)\)$"
)

VAR_ALIASES: Dict[str, str] = {
    "Zone Mean Air Temperature": "Tair_C",
    "Zone Operative Temperature": "Top_C",
    "Zone Air Relative Humidity": "RH_pct",
    "Zone Mean Radiant Temperature": "MRT_C",
    "Zone Heat Index": "HeatIndex_C",
    "Zone Humidity Index": "HumidityIndex",
    "Schedule Value": "ScheduleValue",
    "Zone Ideal Loads Zone Total Heating Energy": "Heat_E_J",
    "Zone Ideal Loads Zone Total Cooling Energy": "Cool_E_J",
    "Zone Ideal Loads Zone Total Heating Rate":   "Heat_P_W",
    "Zone Ideal Loads Zone Total Cooling Rate":   "Cool_P_W",
}

def _clean_zone_name(zone: str, system: str) -> str:
    z = zone.strip()
    if system == "ac":
        z = re.sub(r"\s*IDEAL\s+LOADS\s+AIR\s+SYSTEM\s*$", "", z, flags=re.I)
    z = re.sub(r"[^\w]+", "_", z).strip("_")
    return z

def _parse_header(col: str) -> Optional[Tuple[str, str, str, str]]:
    col = col.strip()

    m = HDR_SCHEDULE_RE.match(col)
    if m:
        zone = m.group("zone").strip()
        var_name = "Schedule Value"
        unit = ""
        freq = m.group("freq").strip()
        var_key = VAR_ALIASES.get(var_name, var_name.replace(" ", "_"))
        return zone, var_key, unit, freq

    m = HDR_RE.match(col)
    if not m:
        return None
    zone = m.group("zone").strip()
    var_name = m.group("var").strip()
    unit = m.group("unit").strip()
    freq = m.group("freq").strip()
    var_key = VAR_ALIASES.get(var_name, var_name.replace(" ", "_"))
    return zone, var_key, unit, freq

def _infer_year_from_filename(path: Path) -> Optional[int]:
    # prioriza padrão MY.1991.csv / MY_1991.csv
    m = re.search(r"MY[._-]((19|20)\d{2})(?=\.csv$)", path.name, flags=re.I)
    if m:
        return int(m.group(1))
    # fallback: separador antes do ano no final
    m2 = re.search(r"[_.-]((19|20)\d{2})(?=\.csv$)", path.name)
    if m2:
        return int(m2.group(1))
    # último fallback: qualquer ano
    m3 = re.search(r"(19|20)\d{2}", path.name)
    return int(m3.group(0)) if m3 else None

def _infer_system_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "_vn_" in name:
        return "vn"
    if "_ac_" in name:
        return "ac"
    return "unknown"

def _resolve_tzinfo(tz_str: Optional[str]) -> Optional[tz.tzinfo]:
    return tz.gettz(tz_str) if tz_str else None

def _parse_eplus_datetime(dt_str: pd.Series, year: int) -> pd.Series:
    s = dt_str.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    mmdd = s.str.extract(r"(\d{1,2}/\d{1,2})", expand=False)
    hms  = s.str.extract(r"(\d{1,2}:\d{2}:\d{2})", expand=False)

    dt = pd.to_datetime(
        mmdd + f"/{year} " + hms,
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )

    is24 = hms.eq("24:00:00")
    if is24.any():
        dt24 = pd.to_datetime(
            mmdd + f"/{year} 00:00:00",
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce",
        ) + pd.Timedelta(days=1)
        dt.loc[is24] = dt24.loc[is24]

    return dt

# ------------------------------------------------------------
# Filename metadata: case, unit_id, seed
# ------------------------------------------------------------

CASE_RE = re.compile(r"(?:^|[_-])(Caso\d+)(?:[_-]|$)", re.IGNORECASE)
UNIT_RE = re.compile(r"^(U\d{3})(?:[_-]|$)", re.IGNORECASE)  # ex.: U001_
# assume que o seed é um bloco só de dígitos antes de _MY...
SEED_RE = re.compile(r"(?:[_-])(\d+)(?:[_-])MY[._-]((19|20)\d{2})", re.IGNORECASE)

def _infer_case_from_filename(path: Path) -> Optional[str]:
    m = CASE_RE.search(path.name)
    if not m:
        return None
    c = m.group(1)
    return c[0].upper() + c[1:]  # "Caso1"

def _infer_unit_id_from_filename(path: Path) -> Optional[str]:
    m = UNIT_RE.search(path.name)
    if not m:
        return None
    u = m.group(1)
    return u.upper()

def _infer_seed_from_filename(path: Path) -> Optional[str]:
    m = SEED_RE.search(path.name)
    if m:
        return m.group(1)
    # fallback: último bloco numérico longo (>=5) no nome (ex.: 833780)
    nums = re.findall(r"(?:^|[_-])(\d{5,})(?:[_-]|$)", path.stem)
    return nums[-1] if nums else None

# ------------------------------------------------------------
# Loader principal (versão "completa")
# ------------------------------------------------------------

def load_eplus_folder(
    folder: Path | str,
    site_id: str,
    tz_str: Optional[str] = None,
    wide: bool = True,
    keep_only_hourly: bool = True,
    system_filter: Optional[Literal["vn", "ac"]] = None,
    case_filter: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lê CSVs do EnergyPlus em `folder` (não recursivo), inferindo:
      - year: pelo nome do arquivo (prioriza MY.1991.csv / MY_1991.csv)
      - system: '_vn_' -> 'vn'; '_ac_' -> 'ac'
      - case: 'Caso1', 'Caso2', ... (por regex no filename)
      - unit_id: 'U001', 'U002', ... (prefixo do filename)
      - seed: bloco numérico antes de '_MY....' (ex.: 833780)
      - scenario: f"{case}_{system}"

    Parâmetros
    ----------
    case_filter : str ou lista de str (ex.: "Caso1" ou ["Caso1","Caso3"])
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"Nenhum CSV do EnergyPlus encontrado em {folder}")

    # normaliza case_filter
    if isinstance(case_filter, str):
        case_filter_set = {case_filter}
    elif isinstance(case_filter, list):
        case_filter_set = set(case_filter)
    else:
        case_filter_set = None

    tzinfo = _resolve_tzinfo(tz_str)
    chunks: List[pd.DataFrame] = []

    for f in files:
        system = _infer_system_from_filename(f)
        if system_filter and system != system_filter:
            continue

        year = _infer_year_from_filename(f)
        if year is None:
            if verbose:
                print(f"[WARN] Sem ano em {f.name}; pulando.")
            continue

        case = _infer_case_from_filename(f) or "CasoNA"
        if case_filter_set is not None and case not in case_filter_set:
            continue

        unit_id = _infer_unit_id_from_filename(f) or "U000"
        seed = _infer_seed_from_filename(f) or ""

        scenario = f"{case}_{system}"

        if verbose:
            print(f"Lendo {f.name} (ano {year}, unit={unit_id}, case={case}, system={system}, seed={seed})...")

        df = pd.read_csv(
            f,
            engine="python",
            skip_blank_lines=True,
            skipinitialspace=True,
        )
        df.columns = [c.strip() for c in df.columns]
        if "Date/Time" not in df.columns:
            raise ValueError(f"{f.name} sem coluna 'Date/Time'.")

        df["timeset"] = _parse_eplus_datetime(df["Date/Time"], year)

        # timezone (opcional)
        if tzinfo is not None:
            if str(df["timeset"].dtype).endswith("[tz]"):
                df["timeset"] = df["timeset"].dt.tz_convert(tzinfo)
            else:
                df["timeset"] = df["timeset"].dt.tz_localize(
                    tzinfo, ambiguous="NaT", nonexistent="shift_forward"
                )

        # Descobrir colunas válidas
        value_cols: List[str] = []
        meta_cols: Dict[str, Tuple[str, str, str]] = {}  # col -> (zone,var_key,unit)
        for c in df.columns:
            if c in ("Date/Time", "timeset"):
                continue
            parsed = _parse_header(c)
            if parsed is None:
                continue
            zone, var_key, unit, freq = parsed
            if keep_only_hourly and freq.lower() != "hourly":
                continue
            value_cols.append(c)
            meta_cols[c] = (zone, var_key, unit)

        if not value_cols:
            if verbose:
                print(f"[WARN] {f.name}: nenhuma coluna horária reconhecida.")
            continue

        base = df[["timeset"]].copy()
        base["site_id"] = site_id
        base["unit_id"] = unit_id
        base["case"] = case
        base["system"] = system
        base["scenario"] = scenario
        base["seed"] = seed

        base["year"] = base["timeset"].dt.year
        base["month"] = base["timeset"].dt.month
        base["day"] = base["timeset"].dt.day
        base["hour"] = base["timeset"].dt.hour

        if wide:
            rename_map = {}
            for c in value_cols:
                zone, var_key, unit = meta_cols[c]
                zone_clean = _clean_zone_name(zone, system)
                rename_map[c] = f"{zone_clean}_{var_key}"
            out = pd.concat([base, df[value_cols].rename(columns=rename_map)], axis=1)

        else:
            out = df[["timeset"] + value_cols].melt(
                id_vars="timeset", var_name="raw", value_name="value"
            )
            rows = []
            for raw in out["raw"]:
                parsed = _parse_header(raw)
                if parsed:
                    zone, var_key, unit, freq = parsed
                else:
                    zone, var_key, unit, freq = None, None, None, ""
                rows.append((zone, var_key, unit, freq))
            meta = pd.DataFrame(rows, columns=["zone", "variable", "unit", "freq"])
            out = pd.concat([out[["timeset", "raw", "value"]], meta], axis=1)
            out = out.dropna(subset=["zone", "variable"]).copy()

            out["site_id"] = site_id
            out["unit_id"] = unit_id
            out["case"] = case
            out["system"] = system
            out["scenario"] = scenario
            out["seed"] = seed

            out["year"] = out["timeset"].dt.year
            out["month"] = out["timeset"].dt.month
            out["day"] = out["timeset"].dt.day
            out["hour"] = out["timeset"].dt.hour

            out = out[
                ["site_id", "unit_id", "case", "system", "scenario", "seed",
                 "timeset", "year", "month", "day", "hour",
                 "zone", "variable", "unit", "value"]
            ]

        chunks.append(out)

    if not chunks:
        raise ValueError("Nenhum dado válido carregado dos CSVs do EnergyPlus.")

    out = (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["timeset", "unit_id", "case", "system"])
        .reset_index(drop=True)
    )

    # Conversão de energia (J → kWh)
    J_TO_KWH = 1.0 / 3.6e6

    if wide:
        energy_cols = [c for c in out.columns if c.endswith("_Heat_E_J") or c.endswith("_Cool_E_J")]
        for c in energy_cols:
            new_name = c.replace("_E_J", "_E_kWh")
            out[new_name] = out[c] * J_TO_KWH
            out.drop(columns=[c], inplace=True)
    else:
        # long: converte Heat_E_J / Cool_E_J
        mask = out["variable"].isin(["Heat_E_J", "Cool_E_J"])
        if mask.any():
            out.loc[mask, "value"] = out.loc[mask, "value"] * J_TO_KWH
            out.loc[mask, "unit"] = "kWh"
            out.loc[mask & out["variable"].eq("Heat_E_J"), "variable"] = "Heat_E_kWh"
            out.loc[mask & out["variable"].eq("Cool_E_J"), "variable"] = "Cool_E_kWh"

    return out
