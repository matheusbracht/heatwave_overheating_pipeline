import pandas as pd

def detect_wetbulb_p90_events(
    daily_tw_df: pd.DataFrame,
    tw_p90: float,
    site_id: str,
    n_consecutive: int = 3,
    value_col: str = "twmean_c",
):
    d = daily_tw_df.copy().reset_index(drop=True)
    d["timeset"] = pd.to_datetime(d["timeset"])

    d["above_thr"] = d[value_col] >= tw_p90

    events = []
    i = 0
    while i < len(d):
        if d.iloc[i]["above_thr"]:
            start = d.iloc[i]["timeset"]
            max_val = float(d.iloc[i][value_col])
            j = i

            while j + 1 < len(d) and d.iloc[j + 1]["above_thr"]:
                j += 1
                max_val = max(max_val, float(d.iloc[j][value_col]))

            end = d.iloc[j]["timeset"]
            duration = (end.date() - start.date()).days + 1

            if duration >= n_consecutive:
                events.append({
                    "start": start,
                    "end": end,
                    "duration_d": duration,
                    "peak_c": max_val,          # ✅ padrão esperado pelo standardize_events
                })

            i = j + 1
        else:
            i += 1

    ev = pd.DataFrame(events)
    if not ev.empty:
        ev["method"] = "TW_P90"
        ev["site_id"] = site_id

    flags = d[["timeset"]].copy()
    flags["HW_TW_bool"] = False
    if not ev.empty:
        for _, row in ev.iterrows():
            mask = (d["timeset"] >= row["start"]) & (d["timeset"] <= row["end"])
            flags.loc[mask, "HW_TW_bool"] = True

    return ev, flags
