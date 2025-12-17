from pathlib import Path
TZ = "America/Sao_Paulo"
BASELINE = ("1991-01-01","2020-12-31")
DATA = {
    "RAW": Path("data/raw"),
    "INTERIM": Path("data/interim"),
    "PROC": Path("data/processed"),
}