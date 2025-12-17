import pandas as pd

def daily_max(df, col="tmax_c"):
    return (df.set_index("timeset")
              .groupby(pd.Grouper(freq="D"))[col].max()
              .reset_index())

def daily_mean(df, col="ta_c"):
    return (df.set_index("timeset")
              .groupby(pd.Grouper(freq="D"))[col].mean()
              .reset_index()
              .rename(columns={col:"tmean_c"}))

def monthly_normals_tmax(daily_max_df, baseline):
    d0, d1 = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
    m = (daily_max_df.query("@d0 <= timeset <= @d1")
           .assign(month=lambda d: d["timeset"].dt.month)
           .groupby("month", as_index=False)["tmax_c"].mean()
           .rename(columns={"tmax_c":"normal_tmax_c"}))
    return m

def ouzeau_thresholds_tmean(daily_mean_df, baseline):
    d0, d1 = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
    ref = daily_mean_df.query("@d0 <= timeset <= @d1")["tmean_c"].dropna()
    return {
        "spic": ref.quantile(0.995),
        "sdeb": ref.quantile(0.975),
        "sint": ref.quantile(0.95),
    }