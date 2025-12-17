from __future__ import annotations
from pathlib import Path
import re
from typing import Optional, Literal, Dict, Tuple, List
import pandas as pd
from dateutil import tz

# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------

# Ex.: "SALA:Zone Operative Temperature [C](Hourly)"
HDR_RE = re.compile(
    r"^(?P<zone>[^:]+):\s*(?P<var>.+?)\s*\[(?P<unit>[^\]]*)\]\((?P<freq>[^)]*)\)$"
)
# Ex.: "SCH_OCUP_SALA:Schedule Value [](Hourly)"
HDR_SCHEDULE_RE = re.compile(
    r"^(?P<zone>[^:]+):\s*Schedule Value\s*\[\]\((?P<freq>[^)]*)\)$"
)

# Mapear nomes “verbosos” -> chaves curtas e consistentes
VAR_ALIASES: Dict[str, str] = {
    # Condições ambientais internas
    "Zone Mean Air Temperature": "Tair_C",
    "Zone Operative Temperature": "Top_C",
    "Zone Air Relative Humidity": "RH_pct",
    "Zone Mean Radiant Temperature": "MRT_C",
    "Zone Heat Index": "HeatIndex_C",
    "Zone Humidity Index": "HumidityIndex",
    "Schedule Value": "ScheduleValue",

    # Ideal Loads (AC): Energia/Taxa
    "Zone Ideal Loads Zone Total Heating Energy": "Heat_E_J",
    "Zone Ideal Loads Zone Total Cooling Energy": "Cool_E_J",
    "Zone Ideal Loads Zone Total Heating Rate":   "Heat_P_W",
    "Zone Ideal Loads Zone Total Cooling Rate":   "Cool_P_W",
}

def _clean_zone_name(zone: str, system: str) -> str:
    z = zone.strip()
    # Para arquivos do AC, remova o sufixo "IDEAL LOADS AIR SYSTEM"
    if system == "ac":
        z = re.sub(r"\s*IDEAL\s+LOADS\s+AIR\s+SYSTEM\s*$", "", z, flags=re.I)
    # normaliza: troca qualquer coisa não alfanumérica por "_", colapsa múltiplos "_"
    z = re.sub(r"[^\w]+", "_", z).strip("_")
    return z

def _parse_header(col: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Retorna (zone, var_key, unit, freq) ou None.
    var_key é padronizada via VAR_ALIASES quando possível.
    """
    col = col.strip()
    # schedule
    m = HDR_SCHEDULE_RE.match(col)
    if m:
        zone = m.group("zone").strip()
        var_name = "Schedule Value"
        unit = ""
        freq = m.group("freq").strip()
        var_key = VAR_ALIASES.get(var_name, var_name.replace(" ", "_"))
        return zone, var_key, unit, freq

    # geral
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
    m = re.search(r"[_.-](19|20)\d{2}(?=\.csv$)", path.name)
    if m:
        return int(m.group(0).lstrip("._-"))
    m2 = re.search(r"(19|20)\d{2}", path.name)
    return int(m2.group(0)) if m2 else None

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
    """
    Parse robusto para a coluna 'Date/Time' do EnergyPlus no formato 'MM/DD  HH:MM:SS'
    com correção explícita de '24:00:00' -> '00:00:00' do dia seguinte.
    """
    s = dt_str.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # extrai MM/DD e HH:MM:SS
    mmdd = s.str.extract(r"(\d{1,2}/\d{1,2})", expand=False)
    hms  = s.str.extract(r"(\d{1,2}:\d{2}:\d{2})", expand=False)

    # parse padrão
    dt = pd.to_datetime(
        mmdd + f"/{year} " + hms,
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )

    # trata 24:00:00 -> 00:00:00 + 1 dia
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
# Loader principal
# ------------------------------------------------------------

def load_eplus_folder(
    folder: Path | str,
    site_id: str,
    tz_str: Optional[str] = None,
    wide: bool = True,
    keep_only_hourly: bool = True,
    system_filter: Optional[Literal["vn","ac"]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lê CSVs do EnergyPlus em `folder` (não recursivo), inferindo:
      - ano pelo nome do arquivo (ex.: *_MY.1991.csv)
      - sistema pelo nome do arquivo: '_vn_' -> 'vn'; '_ac_' -> 'ac'
      - variáveis (zona/var/unidade/freq) a partir do cabeçalho

    Parâmetros
    ----------
    folder : pasta com CSVs do E+ (um por ano)
    site_id : ex. 'BSB'
    tz_str : ex. 'America/Sao_Paulo' (opcional)
    wide : True -> colunas por zona/variável; False -> formato longo (tidy)
    keep_only_hourly : mantém apenas '(Hourly)'
    system_filter : 'vn', 'ac' ou None para carregar todos
    verbose : prints auxiliares

    Retorno
    -------
    DataFrame com:
      - chaves: site_id, system, timeset, year, month, day, hour
      - (wide) ex.: 'SALA_Top_C', 'DORM1_RH_pct', 'DORM1_Cool_E_J', 'SALA_Cool_P_W', ...
      - (long) colunas: zone, variable, unit, value
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"Nenhum CSV do EnergyPlus encontrado em {folder}")

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
        if verbose:
            print(f"Lendo {f.name} (ano {year}, system={system})...")

        df = pd.read_csv(
            f,
            engine="python",
            skip_blank_lines=True,
            skipinitialspace=True,
        )
        df.columns = [c.strip() for c in df.columns]
        if "Date/Time" not in df.columns:
            raise ValueError(f"{f.name} sem coluna 'Date/Time'.")

        # 'MM/DD  HH:MM:SS' -> combinar com ano + correção 24:00:00
        df["timeset"] = _parse_eplus_datetime(df["Date/Time"], year)


        # timezone (opcional)
        if tzinfo is not None:
            if str(df["timeset"].dtype).endswith("[tz]"):
                df["timeset"] = df["timeset"].dt.tz_convert(tzinfo)
            else:
                df["timeset"] = df["timeset"].dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="shift_forward")

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

        base = df[["timeset"]].copy()
        base["site_id"] = site_id
        base["system"]  = system
        base["year"]  = base["timeset"].dt.year
        base["month"] = base["timeset"].dt.month
        base["day"]   = base["timeset"].dt.day
        base["hour"]  = base["timeset"].dt.hour

        if not value_cols:
            if verbose:
                print(f"[WARN] {f.name}: nenhuma coluna horária reconhecida.")
            continue

        if wide:
            # Zone_VarKey (ex.: SALA_Top_C, DORM1_Cool_E_J)
            rename_map = {}
            for c in value_cols:
                zone, var_key, unit = meta_cols[c]
                zone_clean = _clean_zone_name(zone, system)
                new_name = f"{zone_clean}_{var_key}"
                rename_map[c] = new_name
            out = pd.concat([base, df[value_cols].rename(columns=rename_map)], axis=1)

        else:
            # formato longo/tidy
            out = df[["timeset"] + value_cols].melt(id_vars="timeset", var_name="raw", value_name="value")
            rows = []
            for raw in out["raw"]:
                parsed = _parse_header(raw)
                if parsed:
                    zone, var_key, unit, freq = parsed
                else:
                    zone, var_key, unit, freq = None, None, None, ""
                rows.append((zone, var_key, unit, freq))
            meta = pd.DataFrame(rows, columns=["zone","variable","unit","freq"])
            out = pd.concat([out[["timeset","raw","value"]], meta], axis=1)
            out = out.dropna(subset=["zone","variable"]).copy()
            out["site_id"] = site_id
            out["system"]  = system
            out["year"]  = out["timeset"].dt.year
            out["month"] = out["timeset"].dt.month
            out["day"]   = out["timeset"].dt.day
            out["hour"]  = out["timeset"].dt.hour
            out = out[["site_id","system","timeset","year","month","day","hour","zone","variable","unit","value"]]

        chunks.append(out)

    if not chunks:
        raise ValueError("Nenhum dado válido carregado dos CSVs do EnergyPlus.")
    out = pd.concat(chunks, ignore_index=True).sort_values(["timeset","system"]).reset_index(drop=True)

    # Conversão de energia (J → kWh) para colunas do sistema ideal (AC)
    J_TO_KWH = 1.0 / 3.6e6

    energy_cols = [c for c in out.columns if c.endswith("_Heat_E_J") or c.endswith("_Cool_E_J")]
    if energy_cols:
        for c in energy_cols:
            new_name = c.replace("_E_J", "_E_kWh")  # renomeia a unidade no nome da coluna
            out[new_name] = out[c] * J_TO_KWH
            out.drop(columns=[c], inplace=True)
            
    return out
