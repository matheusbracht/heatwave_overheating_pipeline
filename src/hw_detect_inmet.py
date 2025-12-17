import pandas as pd
from .events_utils import level_from_duration

def detect_inmet_events(daily_max_df, normals_m, site_id, delta=5.0):
    df = daily_max_df.copy()
    df["month"] = df["timeset"].dt.month
    df = df.merge(normals_m, on="month", how="left")
    df["thr_c"] = df["normal_tmax_c"] + delta
    df["hit"] = df["tmax_c"] >= df["thr_c"]

    # grupos de True consecutivos
    grp = (df["hit"] != df["hit"].shift()).cumsum()
    df["run_id"] = grp.where(df["hit"])
    events = (df.dropna(subset=["run_id"])
                .groupby("run_id")
                .agg(start=("timeset","min"),
                     end=("timeset","max"),
                     duration_d=("timeset","nunique"),
                     peak_c=("tmax_c","max"))
                .reset_index(drop=True))
    # regra INMET: >= 2 dias
    events = events.query("duration_d >= 2").copy()
    events["method"] = "INMET"
    events["site_id"] = site_id
    events["level"] = events["duration_d"].map(level_from_duration)
    return events, df[["timeset","hit"]].rename(columns={"hit":"HW_INMET_bool"})