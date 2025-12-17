import pandas as pd

def detect_ouzeau_events(daily_mean_df, thresholds, site_id, n_consecutive=3):
    d = daily_mean_df.copy().reset_index(drop=True)

    d["above_sdeb"] = d["tmean_c"] > thresholds["sdeb"]
    d["below_sint"] = d["tmean_c"] < thresholds["sint"]

    events = []
    i = 0
    while i < len(d):
        if d.iloc[i]["above_sdeb"]:              # ✅ iloc
            start = d.iloc[i]["timeset"]         # ✅ iloc
            max_t = d.iloc[i]["tmean_c"]         # ✅ iloc
            j = i
            consec_below_sdeb = 0
            ended = False

            while j + 1 < len(d):
                j += 1
                max_t = max(max_t, d.iloc[j]["tmean_c"])     # ✅ iloc

                if d.iloc[j]["below_sint"]:                  # ✅ iloc
                    ended = True
                    break

                if d.iloc[j]["tmean_c"] < thresholds["sdeb"]:  # ✅ iloc
                    consec_below_sdeb += 1
                    if consec_below_sdeb >= n_consecutive:
                        ended = True
                        break
                else:
                    consec_below_sdeb = 0

            end = d.iloc[j]["timeset"] if ended else d.iloc[len(d) - 1]["timeset"]  # ✅ iloc
            duration = (end.date() - start.date()).days + 1

            if max_t >= thresholds["spic"]:
                events.append({"start": start, "end": end, "duration_d": duration, "peak_c": max_t})

            i = j + 1
        else:
            i += 1

    ev = pd.DataFrame(events)
    if not ev.empty:
        ev["method"] = "Ouzeau"
        ev["site_id"] = site_id

    flags = d[["timeset"]].copy()
    flags["HW_OU_bool"] = False
    for _, row in ev.iterrows():
        mask = (d["timeset"] >= row["start"]) & (d["timeset"] <= row["end"])
        flags.loc[mask, "HW_OU_bool"] = True

    return ev, flags
