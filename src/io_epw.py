from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

EPW_COLS = [
    "Year","Month","Day","Hour","Minute",
    "DataSourceAndUncertaintyFlags",
    "DryBulb","DewPoint","RelHum","AtmosPressure",
    "ExtHorzRad","ExtDirNormRad","HorzIRSky","GloHorzRad","DirNormRad","DifHorzRad",
    "GloHorzIllum","DirNormIllum","DifHorzIllum","ZenLum",
    "WindDir","WindSpd","TotSkyCvr","OpaqSkyCvr","Visibility","CeilHgt",
    "PresWeathObs","PresWeathCodes","PrecipWtr","AerosolOptDepth","SnowDepth","DaysSinceLastSnow",
    "Albedo","LiquidPrecipDepth","LiquidPrecipRate"
]

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_epw_header(epw_file: Path) -> Dict[str, Any]:
    with epw_file.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    parts = [p.strip() for p in first.split(",")]
    meta = {"raw_header": first}
    if len(parts) >= 9 and parts[0].upper().startswith("LOCATION"):
        # Padrão do seu EPW:
        # LOCATION,City,StateProv,Country,DataSource,WMO,Latitude,Longitude,TimeZone,Elevation
        meta.update({
            "city": parts[1] or None,
            "state_prov": parts[2] or None,
            "country": parts[3] or None,
            "data_source": parts[4] or None,
            "wmo": parts[5] or None,
            "latitude": _to_float(parts[6]),
            "longitude": _to_float(parts[7]),
            "timezone_gmt_offset": _to_float(parts[8]),
            "elevation_m": _to_float(parts[9]) if len(parts) > 9 else None,
        })
    return meta

def load_epw_folder(
    folder: Path | str,
    site_id: str,
    scenario: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Lê todos os .epw em `folder` (sem recursão) e retorna DF:
      - timeset NAÏVE (sem timezone)
      - epw_tz_offset_h / epw_tz_label como metadados
      - colunas de cenário: scenario_category, scenario_horizon, scenario_rcp, period_start, period_end
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.epw"))
    if not files:
        raise ValueError(f"Nenhum .epw encontrado em {folder}.")

    rows: List[pd.DataFrame] = []
    for f in files:
        meta = parse_epw_header(f)
        df = pd.read_csv(f, skiprows=8, header=None, names=EPW_COLS)

        y = df["Year"].astype(int) if "Year" in df.columns else df["year"].astype(int)
        m = df["Month"].astype(int) if "Month" in df.columns else df["month"].astype(int)
        d = df["Day"].astype(int) if "Day" in df.columns else df["day"].astype(int)
        h = df["Hour"].astype(int) if "Hour" in df.columns else df["hour"].astype(int)

        base = pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce")
        df["timeset"] = base + pd.to_timedelta(h, unit="h")

        df = df.rename(columns={
            "Year":"year","Month":"month","Day":"day","Hour":"hour",
            "DryBulb":"ta_c","DewPoint":"tdp_c","RelHum":"rh_pct","AtmosPressure":"p_atm_pa",
            "GloHorzRad":"ghi_Whm2","DirNormRad":"dni_Whm2","DifHorzRad":"dhi_Whm2",
            "ExtHorzRad":"ext_ghi_Whm2","ExtDirNormRad":"ext_dni_Whm2",
            "HorzIRSky":"ir_horiz_Wm2",
            "GloHorzIllum":"ghi_illum_lux","DirNormIllum":"dni_illum_lux","DifHorzIllum":"dhi_illum_lux",
            "ZenLum":"zen_lum_cd_m2",
            "WindDir":"wind_dir_deg","WindSpd":"wind_spd_ms",
            "TotSkyCvr":"tot_sky_cover_tenths","OpaqSkyCvr":"opaq_sky_cover_tenths",
            "Visibility":"visibility_km","CeilHgt":"ceil_hgt_m",
            "PresWeathObs":"pres_weather_obs","PresWeathCodes":"pres_weather_codes",
            "PrecipWtr":"precip_wtr_cm","AerosolOptDepth":"aod_thousandths",
            "SnowDepth":"snow_depth_cm","DaysSinceLastSnow":"days_since_last_snow",
            "Albedo":"albedo","LiquidPrecipDepth":"liquid_precip_depth_mm","LiquidPrecipRate":"liquid_precip_rate_mmph",
            "DataSourceAndUncertaintyFlags":"data_source",
        })

        # timeset NAÏVE
        dt = pd.to_datetime({
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].astype(int),
        })
        df["timeset"] = dt

        # metadados fixos (do cabeçalho do EPW)
        tz_off = meta.get("timezone_gmt_offset")
        tz_label = f"UTC{tz_off:+.0f}" if isinstance(tz_off, (int, float)) else None

        df["site_id"]   = site_id
        df["city"]      = meta.get("city")
        df["state_prov"]= meta.get("state_prov")
        df["country"]   = meta.get("country")
        df["data_source"]= meta.get("data_source")
        df["wmo"]       = meta.get("wmo")
        df["latitude"]  = meta.get("latitude")
        df["longitude"] = meta.get("longitude")
        df["elevation_m"]= meta.get("elevation_m")
        df["epw_tz_offset_h"] = tz_off
        df["epw_tz_label"]    = tz_label

        # metadados de cenário (aplicados a TODAS as linhas)
        cat = hor = rcp = None
        p0 = p1 = None
        if scenario:
            cat = scenario.get("category")
            hor = scenario.get("horizon")
            rcp = scenario.get("rcp")
            period = scenario.get("period")
            if isinstance(period, (tuple, list)) and len(period) == 2:
                p0, p1 = period[0], period[1]

        df["scenario_category"] = cat
        df["scenario_horizon"]  = hor
        df["scenario_rcp"]      = rcp
        df["period_start"]      = p0
        df["period_end"]        = p1

        rows.append(df[[
            "site_id","timeset","year","month","day","hour",
            "ta_c","tdp_c","rh_pct","p_atm_pa",
            "ghi_Whm2","dni_Whm2","dhi_Whm2","ext_ghi_Whm2","ext_dni_Whm2","ir_horiz_Wm2",
            "ghi_illum_lux","dni_illum_lux","dhi_illum_lux","zen_lum_cd_m2",
            "wind_dir_deg","wind_spd_ms","tot_sky_cover_tenths","opaq_sky_cover_tenths",
            "visibility_km","ceil_hgt_m","pres_weather_obs","pres_weather_codes",
            "precip_wtr_cm","aod_thousandths","snow_depth_cm","days_since_last_snow",
            "albedo","liquid_precip_depth_mm","liquid_precip_rate_mmph",
            "city","state_prov","country","data_source","wmo","latitude","longitude","elevation_m",
            "scenario_category","scenario_horizon","scenario_rcp","period_start","period_end",
            "epw_tz_offset_h","epw_tz_label"
        ]])

    out = pd.concat(rows, ignore_index=True).sort_values("timeset").reset_index(drop=True)
    return out
