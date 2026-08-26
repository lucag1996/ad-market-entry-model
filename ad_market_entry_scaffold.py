"""
A&D international market entry — two-layer country attractiveness model
======================================================================

Empirical model for the MBA final dissertation.

Layer 1  = market attractiveness (continuous 0-100 composite score,
           two pillars: size and quality)
Layer 2  = geopolitical accessibility (embargo gate x access enabler x offset)
Final    = Layer 1 x Layer 2 accessibility
Validation = Spearman rank correlation between the final ranking and
             realised Italian defence export flows (SIPRI TIV 2015-2024).

HOW TO USE
----------
1. Place the source data files in ./data/ (see README for the list of
   sources and the expected file names).
2. Run: python ad_market_entry_scaffold.py
   Outputs (ranking tables and charts) are written to ./outputs/.

The model uses only public data; no values are fabricated. Countries with
missing variables are scored on the variables available, without imputation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Country set (25), ISO3 codes. Built to generate rank_shift:
# each group is attractive on Layer 1 but differs on Layer 2 accessibility,
# seen from the perspective of an Italy-based (EU/NATO-aligned) A&D exporter.
COUNTRIES = [
    # Group A — attractive & accessible (NATO/EU allies)
    "USA", "GBR", "FRA", "DEU", "POL", "AUS", "JPN", "KOR", "CAN", "ESP", "NLD",
    # Group B — attractive, conditional access (case-by-case licensing)
    "IND", "SAU", "ARE", "QAT", "ISR", "TUR", "BRA", "EGY", "IDN",
    # Group C — attractive but restricted (EU embargo / sanctions)
    "CHN", "RUS", "PAK", "VNM", "DZA",
]
# Conversion ISO3 codes to country names (for charts and tables).
COUNTRY_NAMES = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "FRA": "France",
    "DEU": "Germany",
    "POL": "Poland",
    "AUS": "Australia",
    "JPN": "Japan",
    "KOR": "South Korea",
    "CAN": "Canada",
    "ESP": "Spain",
    "NLD": "Netherlands",
    "IND": "India",
    "SAU": "Saudi Arabia",
    "ARE": "United Arab Emirates",
    "QAT": "Qatar",
    "ISR": "Israel",
    "TUR": "Turkey",
    "BRA": "Brazil",
    "EGY": "Egypt",
    "IDN": "Indonesia",
    "CHN": "China",
    "RUS": "Russia",
    "PAK": "Pakistan",
    "VNM": "Vietnam",
    "DZA": "Algeria"
}
# Safety check: the two lists must stay perfectly in sync.
assert set(COUNTRY_NAMES) == set(COUNTRIES), "COUNTRIES e COUNTRY_NAMES non coincidono"

# Layer-1 weights. Must sum to 1. Keep them explicit — you will justify them
# in the methodology chapter and stress-test them in run_sensitivity().
L1_WEIGHTS = {
    "military_expenditure": 0.20,
    "aero_imports_hs88": 0.15,
    "air_traffic": 0.05,
    "gdp": 0.10,
    "gdp_per_capita": 0.10,
    "lpi": 0.10,
    "industrial_base": 0.10,
    "business_ready": 0.10,
    "country_risk": 0.10,  # already inverted at load time (high = good)
}
# Safety check: weights must sum to 1.
assert abs(sum(L1_WEIGHTS.values()) - 1.0) < 1e-9, "Weights for Layer 1 do not sum to 1"

PILLAR_SIZE = ["military_expenditure", "aero_imports_hs88", "gdp",
               "industrial_base", "air_traffic"]
PILLAR_QUALITY = ["gdp_per_capita", "lpi", "business_ready", "country_risk"]
PILLAR_WEIGHTS = {"size": 0.5, "quality": 0.5}

# ---------- LAYER 2: operational feasibility ----------

# 1. Gate embargo (EU Sanctions Map, agg. 02/07/2026): 0 = closed, 1 = no embargo
EMBARGO_GATE = {"RUS": 0.0, "CHN": 0.1}   # all the others = 1.0 (default)

# 2. Access Enabler: relationship with EU/NATO (NATO/UE 1.0, MNNA 0.85, partner 0.6, sensitive 0.4)
ACCESS_ENABLER = {
    "USA": 1.0, "GBR": 1.0, "FRA": 1.0, "DEU": 1.0, "POL": 1.0,
    "CAN": 1.0, "ESP": 1.0, "NLD": 1.0, "TUR": 1.0,
    "JPN": 0.85, "KOR": 0.85, "AUS": 0.85, "ISR": 0.85,
    "IND": 0.6, "SAU": 0.6, "ARE": 0.6, "QAT": 0.6, "BRA": 0.6, "IDN": 0.6,
    "EGY": 0.4, "DZA": 0.4, "VNM": 0.4, "PAK": 0.4,
    "CHN": 0.6, "RUS": 0.4,   # irrelevant: already zeroed out by the gate
}

# 3. Requirements for offset/localization (draft from sources, tiers verified with Luca)
OFFSET = {
    "USA": 1.0, "GBR": 1.0, "FRA": 1.0, "DEU": 1.0, "ESP": 1.0,
    "NLD": 1.0, "CAN": 1.0, "JPN": 1.0, "EGY": 1.0, "DZA": 1.0,
    "VNM": 1.0, "PAK": 1.0,
    "AUS": 0.9, "ISR": 0.9, "QAT": 0.9, "POL": 0.9, "KOR": 0.9,
    "IND": 0.8, "ARE": 0.8, "SAU": 0.8, "TUR": 0.8, "IDN": 0.8, "BRA": 0.8,
    "CHN": 1.0, "RUS": 1.0,   # irrelevant: already zeroed out by the gate
}

N_CLUSTERS = 4


# ---------------------------------------------------------------------------
# DATA LOADING — each function returns a Series/DataFrame indexed by ISO3
# ---------------------------------------------------------------------------
def load_sipri_milex(ref_year: int = 2024) -> pd.Series:
    """
    Upload the SIPRI military expenditure data (constant 2024 US$ mln) for the 25 countries,
    indexed by ISO3 code.


    Methodology rule: the value is used ONLY if it exists for ref_year.
    Countries without data in the reference year (e.g., ARE, QAT) remain NaN
    and are handled downstream in the scoring, not filled upstream.
    """
    sipri_file = DATA_DIR / "sipri_milex_2025.xlsx"
    sheet = "Constant (2024) US$"
    header_row = 5  # row with "Country", "Notes" and the years

    # map ISO3 -> Exact spelling used by SIPRI (verified in the file)
    iso3_to_sipri = {
        "USA": "United States of America", "GBR": "United Kingdom",
        "FRA": "France", "DEU": "Germany", "POL": "Poland", "AUS": "Australia",
        "JPN": "Japan", "KOR": "Korea, South", "CAN": "Canada", "ESP": "Spain",
        "NLD": "Netherlands", "IND": "India", "SAU": "Saudi Arabia",
        "ARE": "United Arab Emirates", "QAT": "Qatar", "ISR": "Israel",
        "TUR": "Türkiye", "BRA": "Brazil", "EGY": "Egypt", "IDN": "Indonesia",
        "CHN": "China", "RUS": "Russia", "PAK": "Pakistan", "VNM": "Viet Nam",
        "DZA": "Algeria",
    }

    # 1. read the sheet with the actual header row
    df = pd.read_excel(sipri_file, sheet_name=sheet, header=header_row)
    df = df.rename(columns={df.columns[0]: "country"})
    df["country"] = df["country"].astype(str).str.strip()

    # 2. normalize the year column names to integers (1949.0 -> 1949)
    def to_year(c):
        try:
            return int(float(c))
        except (ValueError, TypeError):
            return c
    df.columns = ["country"] + [to_year(c) for c in df.columns[1:]]

    if ref_year not in df.columns:
        raise ValueError(f"Anno {ref_year} non presente nel file SIPRI.")

    # 3. map the country names to their values for the reference year (xxx/... -> NaN)
    name_to_value = dict(zip(df["country"],
                             pd.to_numeric(df[ref_year], errors="coerce")))

    # 4. construct the series indexed by ISO3, only for our 25 countries
    values = {}
    for iso, sipri_name in iso3_to_sipri.items():
        values[iso] = name_to_value.get(sipri_name)  # None if the name is missing

    serie = pd.Series(values, name="military_expenditure")

    # 5. log of control: how many countries have a valid value
    n_ok = serie.notna().sum()
    missing = sorted(serie[serie.isna()].index)
    print(f"[SIPRI milex {ref_year}] {n_ok}/25 countries with data. Missing: {missing}")

    return serie


def load_sipri_milex_avg(start_year: int = 2022, end_year: int = 2024) -> pd.Series:
    """Average SIPRI military expenditure (constant 2024 US$) over the indicated years,
    for temporal robustness check."""
    sipri_file = DATA_DIR / "sipri_milex_2025.xlsx"
    df = pd.read_excel(sipri_file, sheet_name="Constant (2024) US$", header=5)
    df = df.rename(columns={df.columns[0]: "country"})
    df["country"] = df["country"].astype(str).str.strip()

    def to_year(c):
        try:
            return int(float(c))
        except (ValueError, TypeError):
            return c
    df.columns = ["country"] + [to_year(c) for c in df.columns[1:]]
    year_cols = [y for y in range(start_year, end_year + 1) if y in df.columns]

    iso3_to_sipri = {
        "USA": "United States of America", "GBR": "United Kingdom", "FRA": "France",
        "DEU": "Germany", "POL": "Poland", "AUS": "Australia", "JPN": "Japan",
        "KOR": "Korea, South", "CAN": "Canada", "ESP": "Spain", "NLD": "Netherlands",
        "IND": "India", "SAU": "Saudi Arabia", "ARE": "United Arab Emirates",
        "QAT": "Qatar", "ISR": "Israel", "TUR": "Türkiye", "BRA": "Brazil",
        "EGY": "Egypt", "IDN": "Indonesia", "CHN": "China", "RUS": "Russia",
        "PAK": "Pakistan", "VNM": "Viet Nam", "DZA": "Algeria",
    }
    vals = {}
    for iso, name in iso3_to_sipri.items():
        row = df[df["country"] == name]
        if row.empty:
            vals[iso] = None
            continue
        s = pd.to_numeric(row[year_cols].iloc[0], errors="coerce")
        vals[iso] = s.mean() if s.notna().any() else None
    serie = pd.Series(vals, name="military_expenditure")
    print(f"[SIPRI milex media {start_year}-{end_year}] {serie.notna().sum()}/25 con dato.")
    return serie


def load_comtrade_aero_imports(ref_year: int = 2024) -> pd.Series:
    """
    Upload aerospace imports (HS chapter 88, partner World, flow Import)
    from UN Comtrade, in current US$, indexed by ISO3 code.

    Filters on the numeric code 88 (not the textual description, which varies
    across versions of the HS classification). Returns NaN for the 25 countries
    that do not have a row in the file (missing declaration for ref_year).
    """
    csv_file = DATA_DIR / "comtrade_hs88_imports_2024.csv"

    df = pd.read_csv(csv_file, encoding="latin-1", index_col=False)

    # keep only the relevant rows: import, chapter 88, partner World
    df = df[(df["flowCode"] == "M")
            & (df["cmdCode"].astype(str) == "88")
            & (df["partnerCode"] == 0)].copy()

    # value in dollars and ISO3 code of the reporter
    df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce")

    # check for duplicates: a country should not appear more than once
    dupes = df["reporterISO"].value_counts()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        print(f"[Comtrade] ATTENZIONE, dichiaranti duplicati (sommo i valori): "
              f"{list(dupes.index)}")
    per_country = df.groupby("reporterISO")["primaryValue"].sum()

    # construct the series only for our 25 countries
    values = {iso: per_country.get(iso) for iso in COUNTRIES}
    serie = pd.Series(values, name="aero_imports_hs88")

    n_ok = serie.notna().sum()
    missing = sorted(serie[serie.isna()].index)
    print(f"[Comtrade aero {ref_year}] {n_ok}/25 paesi con dato. Mancanti: {missing}")

    return serie


def load_comtrade_aero_avg(years=(2022, 2023, 2024)) -> pd.Series:
    """Average aerospace imports (HS88) over the indicated years, for temporal
    robustness check. Average over the available years for each country."""
    per_year = []
    for y in years:
        f = DATA_DIR / f"comtrade_hs88_imports_{y}.csv"
        df = pd.read_csv(f, encoding="latin-1", index_col=False)
        df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce")
        s = df.groupby("reporterISO")["primaryValue"].sum(min_count=1)
        per_year.append(s.reindex(COUNTRIES))
    serie = pd.concat(per_year, axis=1).mean(axis=1)   # media sugli anni con dato
    serie.name = "aero_imports_hs88"
    print(f"[Comtrade aero media {years[0]}-{years[-1]}] {serie.notna().sum()}/25 con dato.")
    return serie


def fetch_worldbank_indicator(indicator: str, date_range: str = "2016:2024") -> pd.Series:
    """
    Generic Helper: interrogates the World Bank API for an indicator and returns
    the most recent available value for each of the 25 countries, indexed by ISO3.

    Reusable for any WDI indicator by changing only 'indicator'.
    No API key required. Internet connection needed.
    """
    codes = ";".join(COUNTRIES)
    url = f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}"
    params = {"date": date_range, "format": "json", "per_page": 500}

    import time
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  tentativo {attempt+1} fallito ({type(e).__name__}), riprovo...")
            time.sleep(3)
    else:
        raise RuntimeError(f"World Bank API non raggiungibile dopo 3 tentativi: {last_error}")
    payload = r.json()
    if len(payload) < 2 or payload[1] is None:
        raise ValueError(f"No data returned for {indicator}")

    df = pd.DataFrame(payload[1])
    df["iso3"] = df["countryiso3code"]
    df["year"] = pd.to_numeric(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # for each country, the value of the most recent year with a valid value
    valid = df.dropna(subset=["value"]).sort_values("year")
    latest = valid.groupby("iso3").tail(1).set_index("iso3")["value"]

    values = {iso: latest.get(iso) for iso in COUNTRIES}
    return pd.Series(values)


def load_worldbank_lpi() -> pd.Series:
    """LPI overall (1-5), edition 2023 archived by the World Bank as 2022."""
    serie = fetch_worldbank_indicator("LP.LPI.OVRL.XQ", date_range="2016:2023")
    serie.name = "lpi"
    n_ok = serie.notna().sum()
    missing = sorted(serie[serie.isna()].index)
    print(f"[World Bank LPI] {n_ok}/25 countries with data. Missing: {missing}")
    return serie


def load_worldbank_gdp() -> pd.Series:
    """GDP in US$ current (market size)."""
    serie = fetch_worldbank_indicator("NY.GDP.MKTP.CD", date_range="2020:2024")
    serie.name = "gdp"
    n_ok = serie.notna().sum()
    print(f"[World Bank GDP] {n_ok}/25 countries with data.")
    return serie


def load_worldbank_gdp_per_capita() -> pd.Series:
    """GDP per capita in US$ current (wealth / spending capacity)."""
    serie = fetch_worldbank_indicator("NY.GDP.PCAP.CD", date_range="2020:2024")
    serie.name = "gdp_per_capita"
    n_ok = serie.notna().sum()
    print(f"[World Bank GDP pc] {n_ok}/25 countries with data.")
    return serie


def load_worldbank_industrial_base() -> pd.Series:
    """
    Industrial base = Manufacturing value added, current US$ (NV.IND.MANF.CD).
    Proxy of the manufacturing capacity of the country, relevant for offset,
    local production and industrial partnerships in A&D.
    Note: correlates with gdp, to be verified in the correlation matrix.
    """
    serie = fetch_worldbank_indicator("NV.IND.MANF.CD", date_range="2019:2024")
    serie.name = "industrial_base"
    n_ok = serie.notna().sum()
    print(f"[World Bank industrial base] {n_ok}/25 countries with data.")
    return serie


def load_worldbank_air_traffic() -> pd.Series:
    """
    Air traffic = air passengers (IS.AIR.PSGR).
    Proxy of the vitality of the civilian aviation of the country (demand for aircraft
    and services). Measures the civilian component, not the military one: to be declared
    in the notes as an indirect proxy.
    """
    serie = fetch_worldbank_indicator("IS.AIR.PSGR", date_range="2019:2024")
    serie.name = "air_traffic"
    n_ok = serie.notna().sum()
    print(f"[World Bank air traffic] {n_ok}/25 countries with data.")
    return serie


def load_wgi_regulatory_quality(ref_year: int = 2024) -> pd.Series:
    """
    business_ready = Regulatory Quality (WGI), Governance estimate (-2.5..+2.5).
    Source: excel file downloaded manually, sheet 'rq' (long format:
    one row per country-year). Proxy of the regulatory quality / business
    environment, in substitution of the B-READY (incomplete coverage).
    """
    wgi_file = DATA_DIR / "wgi.xlsx"
    col_code = "Economy (code)"
    col_year = "Year"
    col_est = "Governance estimate (approx. -2.5 to +2.5)"

    df = pd.read_excel(wgi_file, sheet_name="rq")
    df = df[[col_code, col_year, col_est]].copy()
    df[col_est] = pd.to_numeric(df[col_est], errors="coerce")  # '..' -> NaN

    # for each country, the value of the most recent available year up to ref_year
    df = df[df[col_year] <= ref_year].dropna(subset=[col_est])
    df = df.sort_values(col_year)
    latest = df.groupby(col_code).tail(1).set_index(col_code)[col_est]

    values = {iso: latest.get(iso) for iso in COUNTRIES}
    serie = pd.Series(values, name="business_ready")

    n_ok = serie.notna().sum()
    missing = sorted(serie[serie.isna()].index)
    print(f"[WGI Regulatory Quality] {n_ok}/25 paesi con dato. Mancanti: {missing}")
    return serie


def load_oecd_country_risk() -> pd.Series:
    """
    country_risk = OECD Country Risk Classification (0=minimo, 7=massimo),
    applicabile dal 26/06/2026. It measures commercial/payment risk
    (failure to reimburse foreign debt), distinct from the geopolitical gates of
    Layer 2. Inverted to 'high = good' like the other attractiveness variables:
    score = 7 - category, so risk 0 -> score 7 (best).
    """
    oecd_risk = {
        "USA": 0, "GBR": 0, "FRA": 0, "DEU": 0, "JPN": 0, "KOR": 0,
        "CAN": 0, "ESP": 0, "NLD": 0, "AUS": 0, "ISR": 0, "POL": 0,
        "IND": 3, "SAU": 2, "ARE": 2, "QAT": 2, "TUR": 5, "BRA": 4,
        "EGY": 6, "IDN": 3, "CHN": 2, "RUS": 7, "PAK": 7, "VNM": 3,
        "DZA": 5,
    }
    values = {iso: (7 - oecd_risk[iso]) if iso in oecd_risk else None
              for iso in COUNTRIES}
    serie = pd.Series(values, name="country_risk")

    n_ok = serie.notna().sum()
    print(f"[OECD country risk] {n_ok}/25 countries with data (inverted, high=good).")
    return serie


def load_italy_arms_exports() -> pd.Series:
    """
    TIV cumulato 2015-2024 dell'export d'armi italiano verso i 25 paesi
    (SIPRI Arms Transfers Database). Paesi non presenti = nessuna consegna = 0.
    """
    csv = DATA_DIR / "sipri_italy_exports.csv"
    df = pd.read_csv(csv, skiprows=9, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["Recipient"] = df["Recipient"].astype(str).str.strip()
    tiv = pd.to_numeric(df["2015-2024"].astype(str).str.strip(),
                        errors="coerce").fillna(0)
    name_to_tiv = dict(zip(df["Recipient"], tiv))

    iso3_to_sipri = {
        "USA": "United States", "GBR": "United Kingdom", "FRA": "France",
        "DEU": "Germany", "POL": "Poland", "AUS": "Australia", "JPN": "Japan",
        "KOR": "South Korea", "CAN": "Canada", "ESP": "Spain", "NLD": "Netherlands",
        "IND": "India", "SAU": "Saudi Arabia", "ARE": "United Arab Emirates",
        "QAT": "Qatar", "ISR": "Israel", "TUR": "Turkiye", "BRA": "Brazil",
        "EGY": "Egypt", "IDN": "Indonesia", "CHN": "China", "RUS": "Russia",
        "PAK": "Pakistan", "VNM": "Viet Nam", "DZA": "Algeria",
    }
    values = {iso: name_to_tiv.get(name, 0.0) for iso, name in iso3_to_sipri.items()}
    serie = pd.Series(values, name="italy_arms_tiv")
    print(f"[SIPRI Italy exports] paesi con export > 0 = {(serie > 0).sum()}/25")
    return serie


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

# dimension variables on which to apply log (span across orders of magnitude)
LOG_VARS = ["military_expenditure", "aero_imports_hs88", "gdp",
            "gdp_per_capita", "industrial_base", "air_traffic"]


def assemble_layer1() -> pd.DataFrame:
    """unify the 9 series into a single DataFrame: rows = countries (ISO3), columns = variables."""
    series = {
        "military_expenditure": load_sipri_milex(2024),
        "aero_imports_hs88": load_comtrade_aero_imports(2024),
        "lpi": load_worldbank_lpi(),
        "industrial_base": load_worldbank_industrial_base(),
        "gdp": load_worldbank_gdp(),
        "gdp_per_capita": load_worldbank_gdp_per_capita(),
        "air_traffic": load_worldbank_air_traffic(),
        "business_ready": load_wgi_regulatory_quality(2024),
        "country_risk": load_oecd_country_risk(),
    }
    df = pd.DataFrame(series)
    df = df.reindex(COUNTRIES)   # fixed order of the 25 countries
    return df


def normalize_layer1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log to size variables, then min-max 0-1 to all.
    NaN values preserved (missing data handling in subsequent step).
    """
    out = df.copy()

    # 1. log on size variables (all positive where present)
    for col in LOG_VARS:
        out[col] = np.log(out[col])

    # 2. min-max 0-1 column by column, ignoring NaN
    for col in out.columns:
        lo, hi = out[col].min(), out[col].max()   # min/max skip NaN
        out[col] = (out[col] - lo) / (hi - lo)

    return out


def compute_layer1(norm: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1 score with two pillars (0-100).
    Each pillar = simple average of its normalized variables; missing
    values are skipped, so a country is evaluated on the variables it has (no invented data, no penalty). Final score = 50/50 combination.
    """
    size = norm[PILLAR_SIZE].mean(axis=1)         # average, skips NaN
    quality = norm[PILLAR_QUALITY].mean(axis=1)

    score = (PILLAR_WEIGHTS["size"] * size
             + PILLAR_WEIGHTS["quality"] * quality) * 100

    out = pd.DataFrame({
        "pillar_size": (size * 100).round(1),
        "pillar_quality": (quality * 100).round(1),
        "layer1_score": score.round(1),
    }).sort_values("layer1_score", ascending=False)

    # transparency: countries with the size pillar based on partial data
    partial = norm[PILLAR_SIZE].isna().any(axis=1)
    if partial.any():
        print("[Layer 1] countries evaluated on partial data:")
        for iso in norm.index[partial]:
            n = int(norm.loc[iso, PILLAR_SIZE].notna().sum())
            miss = list(norm.loc[iso, PILLAR_SIZE].index[norm.loc[iso, PILLAR_SIZE].isna()])
            print(f"  {iso}: {n}/5 variables, missing {miss}")

    return out


def save_layer1(raw: pd.DataFrame, norm: pd.DataFrame, result: pd.Series) -> None:
    """Save Layer 1 results in output/ as CSV."""
    OUT_DIR.mkdir(exist_ok=True)   # create the folder if it doesn't exist

    # 1. the attractiveness ranking, with the full country name
    ranking = result.copy()
    ranking.insert(0, "country", ranking.index.map(COUNTRY_NAMES))
    ranking.to_csv(OUT_DIR / "layer1_ranking.csv", index_label="iso3")

    # 2. the complete normalized table (useful for charts and verification)
    norm.to_csv(OUT_DIR / "layer1_normalized.csv", index_label="iso3")

    # 3. the assembled raw data (for replicability and inspection)
    raw.to_csv(OUT_DIR / "layer1_raw.csv", index_label="iso3")

    print(f"Saved in {OUT_DIR}/: layer1_ranking.csv (with pillars), layer1_normalized.csv, layer1_raw.csv")


def compute_layer2() -> pd.DataFrame:
    """
    Layer 2 accessibility (operational feasibility) for the 25 countries, in 0-1.
    accessibility = gate_embargo × enabler_access × offset.
    Each component is a multiplier; the gate can zero out everything.
    """
    rows = {}
    for iso in COUNTRIES:
        gate = EMBARGO_GATE.get(iso, 1.0)      # default 1 = no embargo
        enabler = ACCESS_ENABLER[iso]
        offset = OFFSET[iso]
        rows[iso] = {
            "gate": gate,
            "enabler": enabler,
            "offset": offset,
            "accessibility": gate * enabler * offset,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def apply_layer2(layer1: pd.DataFrame, layer2: pd.DataFrame) -> pd.DataFrame:
    """
    Combines attractiveness (Layer 1) and accessibility (Layer 2) into a final score,
    and calculates the rank_shift: how much a country drops (or rises) when moving
    from raw attractiveness to actual reachability.
    """
    df = pd.DataFrame(index=COUNTRIES)
    df["layer1"] = layer1["layer1_score"]
    df["accessibility"] = layer2["accessibility"]
    df["final_score"] = (df["layer1"] * df["accessibility"]).round(1)

    # rank_shift = attractiveness rank - final rank (positive = dropped)
    df["rank_l1"] = df["layer1"].rank(ascending=False).astype(int)
    df["rank_final"] = df["final_score"].rank(ascending=False).astype(int)
    df["rank_shift"] = df["rank_l1"] - df["rank_final"]

    return df.sort_values("final_score", ascending=False)


def save_final(final: pd.DataFrame) -> None:
    """Salva la tabella finale (Layer 1 + Layer 2 + rank_shift) in output/."""
    OUT_DIR.mkdir(exist_ok=True)
    out = final.copy()
    out.insert(0, "country", out.index.map(COUNTRY_NAMES))
    out.to_csv(OUT_DIR / "final_ranking.csv", index_label="iso3")
    print(f"Salvato in {OUT_DIR}/final_ranking.csv")

# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def correlation_layer1(norm: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix (Spearman) between the normalized variables of Layer 1.
    Spearman is used because we are interested in the rank agreement between variables,
    not linearity, and it is robust to outliers. NaN values are handled pairwise.
    Highlights pairs with |correlation| >= 0.8 (possible redundancy).
    """
    cols = list(L1_WEIGHTS)
    corr = norm[cols].corr(method="spearman")

    print("=== CORRELATION MATRIX (Spearman) ===")
    print(corr.round(2).to_string())

    print("\n=== STRONGLY CORRELATED PAIRS (|rho| >= 0.8) ===")
    found = False
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= 0.8:
                print(f"  {a} ~ {b}: {rho:.2f}")
                found = True
    if not found:
        print("  no pairs above 0.8")

    return corr


def validate_against_flows(final: pd.DataFrame, flows: pd.Series) -> None:
    """
    Compare two Spearman correlations against real flows:
    layer1_score vs flows and final_score vs flows. If the final correlates
    better, the Layer 2 brings the model closer to reality.
    """
    df = pd.DataFrame({"layer1": final["layer1"],
                       "final": final["final_score"],
                       "flows": flows}).dropna()
    rho_l1, p_l1 = spearmanr(df["layer1"], df["flows"])
    rho_fin, p_fin = spearmanr(df["final"], df["flows"])
    print(f"\n[validation] n = {len(df)} countries")
    print(f"  Layer 1 vs real flows : rho = {rho_l1:.3f} (p = {p_l1:.3f})")
    print(f"  Final  vs real flows : rho = {rho_fin:.3f} (p = {p_fin:.3f})")
    delta = rho_fin - rho_l1
    print(f"  difference = {delta:+.3f}")


def run_sensitivity(norm: pd.DataFrame, l2_base: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """
    Robustness of the final ranking against judgmental choices. Perturbs the
    allocation between the pillars and the soft multipliers of Layer 2
    (enabler, offset); keeps the embargo gate fixed (legal fact, not judgment).
    Reports mean and standard deviation of each country's rank across n simulations.
    """
    size = norm[PILLAR_SIZE].mean(axis=1)
    quality = norm[PILLAR_QUALITY].mean(axis=1)
    gate, enabler, offset = l2_base["gate"], l2_base["enabler"], l2_base["offset"]

    ranks = []
    for _ in range(n):
        w = np.random.uniform(0.35, 0.65)                      # pillar split
        l1 = w * size + (1 - w) * quality
        enab = (enabler + np.random.uniform(-0.10, 0.10, len(enabler))).clip(0, 1)
        off = (offset + np.random.uniform(-0.05, 0.05, len(offset))).clip(0, 1)
        final = l1 * (gate * enab * off)
        ranks.append(final.rank(ascending=False))

    r = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank_mean": r.mean(axis=1).round(1),
                         "rank_std": r.std(axis=1).round(2)}).sort_values("rank_mean")


def cluster_markets(final: pd.DataFrame, k: int = N_CLUSTERS) -> pd.DataFrame:
    """
    Segment the 25 markets along the two decision-making dimensions: attractiveness
    (Layer 1) and accessibility (Layer 2), both scaled 0-1 for KMeans.
    Returns 'final' with the 'cluster' column and prints the profile of each group.
    """
    X = pd.DataFrame({
        "attractiveness": final["layer1"] / 100.0,   # 0-1
        "accessibility": final["accessibility"],       # già 0-1
    })
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    out = final.copy()
    out["cluster"] = labels

    print("=== CLUSTER PROFILE ===")
    for c in sorted(set(labels)):
        members = out.index[out["cluster"] == c].tolist()
        a = X.loc[members, "attractiveness"].mean() * 100
        acc = X.loc[members, "accessibility"].mean()
        print(f"\nCluster {c}: average attractiveness {a:.0f}/100, "
              f"average accessibility {acc:.2f}, {len(members)} countries")
        print(f"  {members}")
    return out


def run_temporal_check(raw: pd.DataFrame, final_base: pd.DataFrame,
                       l2: pd.DataFrame) -> pd.DataFrame:
    """Run the model with the average military expenditure 2022-2024 instead of just 2024 and the average aerospace imports 2022-2024 instead of just 2024,
    and compare the final ranking with the base one."""
    raw_alt = raw.copy()
    raw_alt["military_expenditure"] = load_sipri_milex_avg(2022, 2024)
    raw_alt["aero_imports_hs88"] = load_comtrade_aero_avg((2022, 2023, 2024))
    final_alt = apply_layer2(compute_layer1(normalize_layer1(raw_alt)), l2)

    cmp = pd.DataFrame({
        "score_2024": final_base["final_score"],
        "score_avg": final_alt["final_score"],
        "rank_2024": final_base["rank_final"],
        "rank_avg": final_alt["rank_final"],
    })
    cmp["rank_change"] = cmp["rank_2024"] - cmp["rank_avg"]

    rho, p = spearmanr(cmp["score_2024"], cmp["score_avg"])
    print(f"\n[temporal robustness] Spearman rank 2024 vs average 2022-24 "
          f"= {rho:.3f} (p = {p:.4f})")
    moved = cmp[cmp["rank_change"] != 0].sort_values("rank_change", key=abs, ascending=False)
    print("\nCountries that change position:")
    print(moved[["rank_2024", "rank_avg", "rank_change"]].to_string() if not moved.empty
          else "  none, identical ranking")
    return cmp


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def plot_pillars(result: pd.DataFrame) -> None:
    """
    Positioning scatter plot: pillar size (x) vs pillar quality (y).
    Each country is a point labeled with its ISO3 code; the medians divide
    the plot into four quadrants. Saves the plot in output/.
    """
    OUT_DIR.mkdir(exist_ok=True)
    x, y = result["pillar_size"], result["pillar_quality"]

    fig, ax = plt.subplots(figsize=(9, 8))
    groups = {
        "Accessible": (["USA","GBR","FRA","DEU","POL","AUS","JPN","KOR","CAN","ESP","NLD"], "#2ca02c"),
        "Conditional": (["IND","SAU","ARE","QAT","ISR","TUR","BRA","EGY","IDN"], "#ff9900"),
        "Restricted": (["CHN","RUS","PAK","VNM","DZA"], "#d62728"),
    }
    for label, (isos, color) in groups.items():
        sub = [i for i in isos if i in result.index]
        ax.scatter(x[sub], y[sub], s=70, color=color, edgecolor="white",
                   label=label, zorder=3)
    ax.legend(title="Geopolitical group", loc="center left", fontsize=8)

    # personalized shifts for overlapping points; default (4, 4)
    label_offsets = {
        "FRA": (-26, 0),
        "GBR": (7, 9),
        "JPN": (9, -10),
        "KOR": (7, -11),
        "DEU": (7, 6),
        "ESP": (-26, 0),
    }
    for iso in result.index:
        dx, dy = label_offsets.get(iso, (4, 4))
        ax.annotate(iso, (x[iso], y[iso]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8)

    # median --> quadrants
    mx, my = x.median(), y.median()
    ax.axvline(mx, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(my, color="grey", linestyle="--", linewidth=0.8, zorder=1)

    # quadrant labels
    ax.text(0.98, 0.98, "Large and Quality", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="grey")
    ax.text(0.02, 0.98, "Small but Quality", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="grey")
    ax.text(0.98, 0.02, "Large but Difficult", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="grey")
    ax.text(0.02, 0.02, "Small and Difficult", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="grey")

    ax.set_xlabel("Pillar Size (0-100)")
    ax.set_ylabel("Pillar Quality (0-100)")
    ax.set_title("Layer 1: Positioning of Markets by Size and Quality")

    fig.savefig(OUT_DIR / "layer1_pillars_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved in {OUT_DIR}/layer1_pillars_scatter.png")


def plot_validation(final: pd.DataFrame, flows: pd.Series) -> None:
    """
    Compare two Spearman correlations against real flows:
    layer1_score vs flows and final_score vs flows. Save in output/.
    Per-panel manual label offsets reduce overlaps.
    """
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame({"layer1": final["layer1"],
                       "final": final["final_score"],
                       "tiv": flows}).dropna()

    # scostamenti manuali (dx, dy) in punti, per pannello; default (3, 3)
    off_l1 = {
        "ISR": (-15, -2), "POL": (-10, 6), "SAU": (7, 1),
        "CAN": (-16, 4), "GBR": (5, -9), "DEU": (6, 2), "JPN": (-16, -12),
        "KOR": (-16, 2), "ESP": (-16, 1), "FRA": (6, 1), "AUS": (-2, 7),
        "ARE": (6, -9), "IND": (-16, 1), "RUS": (-16, 2), "VNM": (-2, -10),
        "CHN": (3, 6),
    }
    off_fin = {
        "POL": (-2, 7), "AUS": (7, -3), "ISR": (5, 3),
        "CAN": (-3, -10), "GBR": (-6, 5), "DEU": (6, 1), "NLD": (1, 3),
        "JPN": (-16, 3), "ESP": (6, -9), "FRA": (6, 2), "KOR": (5, -10),
        "IND": (6, -9), "ARE": (-16, 4), "RUS": (-16, 3), "CHN": (5, -10),
        "VNM": (-2, 7),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panels = [(axes[0], "layer1", "Layer 1 score (attractiveness only)", off_l1),
              (axes[1], "final", "Final score (attractiveness x accessibility)", off_fin)]
    for ax, col, label, offs in panels:
        rho, p = spearmanr(df[col], df["tiv"])
        ax.scatter(df[col], df["tiv"], s=60, color="#2c6fbb",
                   edgecolor="white", zorder=3)
        for iso in df.index:
            dx, dy = offs.get(iso, (3, 3))
            ax.annotate(iso, (df.loc[iso, col], df.loc[iso, "tiv"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=7)
        ax.set_xlabel(label)
        ax.set_title(f"Spearman rho = {rho:.2f} (p = {p:.2f})")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("Actual Italian arms exports, TIV 2015-2024")
    fig.suptitle("Model score vs actual Italian arms exports", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {OUT_DIR}/validation_scatter.png")


def name_clusters(clustered: pd.DataFrame) -> dict:
    """Assign managerial names to each cluster based on its centroid, so that the names
    do not depend on the arbitrary numerical label of the KMeans."""
    prof = clustered.groupby("cluster").agg(attr=("layer1", "mean"),
                                            acc=("accessibility", "mean"))
    names = {}
    blocked = prof["acc"].idxmin()                 # less accessible
    names[blocked] = "Blocked opportunities"
    rest = prof.drop(blocked)
    priority = rest["acc"].idxmax()                # more accessible among the remaining
    names[priority] = "Priority markets"
    rest = rest.drop(priority)
    marginal = rest["attr"].idxmin()               # less attractive of the two
    names[marginal] = "Marginal markets"
    names[rest["attr"].idxmax()] = "Conditional markets"
    return names


def plot_clusters(clustered: pd.DataFrame) -> None:
    """
    Summary plot: attractiveness (x) vs accessibility (y), colored points
    by cluster with managerial names. Saves in output/.
    """
    OUT_DIR.mkdir(exist_ok=True)
    names = name_clusters(clustered)
    colors = {"Priority markets": "#2ca02c", "Conditional markets": "#ff9900",
              "Blocked opportunities": "#d62728", "Marginal markets": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == c]
        ax.scatter(sub["layer1"], sub["accessibility"], s=90,
                   color=colors[names[c]], edgecolor="white",
                   label=names[c], zorder=3)
    label_offsets = {
        "FRA": (-15, 7),    # separates from CAN
        "CAN": (-9, -12),    # pushes down
        "NLD": (-19, 3),    # to the left
        "ESP": (-17, 3),    # to the left
        "GBR": (-2, 7),     # up
        "DEU": (5, 5),
        "AUS": (5, -11),    # down
        "KOR": (5, 6),      # up, separates from AUS
        "JPN": (6, -3),
        "ARE": (0, -12),     # down
    }
    for iso in clustered.index:
        dx, dy = label_offsets.get(iso, (4, 4))
        ax.annotate(iso, (clustered.loc[iso, "layer1"], clustered.loc[iso, "accessibility"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7)

    ax.set_xlabel("Market attractiveness (Layer 1, 0-100)")
    ax.set_ylabel("Operational accessibility (Layer 2, 0-1)")
    ax.set_title("A&D market segmentation: attractiveness vs accessibility")
    ax.legend(title="Cluster", loc="upper left", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(OUT_DIR / "cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {OUT_DIR}/cluster_scatter.png")


def plot_rank_shift(final: pd.DataFrame) -> None:
    """
    Slope chart: position just before (left) vs position after (right)
    with accessibility applied. Downward lines show markets that
    decline once operational feasibility is considered. Saves in output/.
    """
    OUT_DIR.mkdir(exist_ok=True)
    df = final.copy()
    df["disp_l1"] = df["layer1"].rank(ascending=False, method="first").astype(int)
    df["disp_final"] = df["final_score"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("disp_final")
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, 11))
    for iso in df.index:
        r1, r2 = df.loc[iso, "disp_l1"], df.loc[iso, "disp_final"]
        drop = r2 - r1
        color = "#d62728" if drop >= 3 else ("#2ca02c" if drop <= -3 else "#999999")
        ax.plot([0, 1], [r1, r2], color=color, linewidth=1.5, zorder=2)
        ax.text(-0.02, r1, f"{iso} {int(r1)}", ha="right", va="center", fontsize=8)
        ax.text(1.02, r2, f"{int(r2)} {iso}", ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(n + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Attractiveness rank\n(Layer 1)", "Final rank\n(with accessibility)"])
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("How market priorities shift once feasibility is applied")
    fig.savefig(OUT_DIR / "rank_shift_slope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {OUT_DIR}/rank_shift_slope.png")


def plot_sensitivity(sens: pd.DataFrame) -> None:
    """
    Horizontal bar chart: average rank of each country with error bar
    (standard deviation of rank across simulations). Short bars = robust position.
    Saves in output/.
    """
    OUT_DIR.mkdir(exist_ok=True)
    df = sens.sort_values("rank_mean")

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.errorbar(df["rank_mean"], df.index, xerr=df["rank_std"],
                fmt="o", color="#2c6fbb", ecolor="#d62728",
                elinewidth=1.5, capsize=3, zorder=3)
    ax.set_xlabel("Average rank across simulations (error bar = std. dev.)")
    ax.set_title("Ranking robustness: how much each country's position varies")
    ax.invert_yaxis()                       # rango 1 in alto
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.5)
    fig.savefig(OUT_DIR / "sensitivity_ranks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {OUT_DIR}/sensitivity_ranks.png")


def plot_correlation_heatmap(norm: pd.DataFrame) -> None:
    """Heatmap of the correlation matrix (Spearman) between Layer 1 variables,
    ordered by pillar so the two blocks are visually evident."""
    OUT_DIR.mkdir(exist_ok=True)
    cols = PILLAR_SIZE + PILLAR_QUALITY
    corr = norm[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title("Correlation matrix of Layer 1 variables (ordered by pillar)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved in {OUT_DIR}/correlation_heatmap.png")
    
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # --- 1. Data, Layer 1, Layer 2, final score ---
    raw = assemble_layer1()
    norm = normalize_layer1(raw)
    result = compute_layer1(norm)
    l2 = compute_layer2()
    final = apply_layer2(result, l2)
    print("\n=== FINAL RANKING ===")
    print(final[["layer1", "accessibility", "final_score",
                 "rank_l1", "rank_final", "rank_shift"]].to_string())

    # --- 2. Saving tables ---
    save_layer1(raw, norm, result)
    save_final(final)

    # --- 3. Analysis and robustness checks ---
    correlation_layer1(norm)
    clustered = cluster_markets(final)
    sens = run_sensitivity(norm, l2)
    sens_out = sens.copy()
    sens_out.insert(0, "country", sens_out.index.map(COUNTRY_NAMES))
    sens_out.to_csv(OUT_DIR / "sensitivity.csv", index_label="iso3")
    print("\n=== SENSITIVITY (mean rank & std) ===")
    print(sens.to_string())
    flows = load_italy_arms_exports()
    validate_against_flows(final, flows)
    run_temporal_check(raw, final, l2)

    # --- 4. Charts ---
    plot_pillars(result)
    plot_correlation_heatmap(norm)
    plot_clusters(clustered)
    plot_rank_shift(final)
    plot_sensitivity(sens)
    plot_validation(final, flows)

    print("\nDone. See ./outputs/")


if __name__ == "__main__":
    main()