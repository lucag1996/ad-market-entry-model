# A&D International Market Entry — Two-Layer Attractiveness Model

Empirical model developed for an MBA final dissertation (MIB Trieste School of Management).
It prioritises international markets for an Italian Aerospace & Defence (A&D) exporter by
combining two layers:

- **Layer 1 — Market attractiveness**: a 0–100 composite score built on two pillars
  (market *size* and market *quality*).
- **Layer 2 — Geopolitical accessibility**: a 0–1 multiplier combining an embargo gate,
  an alliance-based access enabler, and an offset/localisation factor.
- **Final score** = Layer 1 × Layer 2 accessibility.
- **Validation**: Spearman rank correlation between the model ranking and realised Italian
  arms-export flows (SIPRI TIV, 2015–2024).

The model covers 25 country-markets and produces ranking tables and charts in `./outputs/`.

## Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `requests`

Install them with:

```
pip install pandas numpy scipy scikit-learn matplotlib requests
```

## How to run

1. Download the source data files and place them in `./data/` (see **Data sources** below).
2. Ensure an internet connection is available (World Bank indicators are fetched live via API).
3. Run:

```
python ad_market_entry_scaffold.py
```

Outputs (ranking tables and charts) are written to `./outputs/`.

## Data sources

The underlying datasets are **not redistributed** in this repository, in compliance with the
licences of the original providers. Download each file from its official source and save it in
`./data/` with the exact file name below.

| File name (in `./data/`)             | Source | What to download |
|--------------------------------------|--------|------------------|
| `sipri_milex_2025.xlsx`              | SIPRI Military Expenditure Database (2025 edition), milex.sipri.org | The full Excel workbook; the code reads the sheet **"Constant (2024) US$"**. |
| `sipri_italy_exports.csv`            | SIPRI Arms Transfers Database, armstransfers.sipri.org | Trade-register / TIV export of **Italy's arms exports, 2015–2024**, by recipient. |
| `comtrade_hs88_imports_2022.csv`     | UN Comtrade, comtradeplus.un.org | Imports, HS chapter **88**, reporter = all 25 countries, partner = World, year **2022**. |
| `comtrade_hs88_imports_2023.csv`     | UN Comtrade, comtradeplus.un.org | Same query, year **2023**. |
| `comtrade_hs88_imports_2024.csv`     | UN Comtrade, comtradeplus.un.org | Same query, year **2024**. |
| `wgi.xlsx`                           | World Bank Worldwide Governance Indicators, govindicators.org | The WGI workbook; the code reads the sheet **"rq"** (Regulatory Quality). |

The following indicators are fetched **automatically at run time** via the World Bank API
(no manual download needed): GDP (`NY.GDP.MKTP.CD`), GDP per capita (`NY.GDP.PCAP.CD`),
LPI (`LP.LPI.OVRL.XQ`), manufacturing value added (`NV.IND.MANF.CD`), air passengers
(`IS.AIR.PSGR`).

The **OECD Country Risk Classification** values are embedded directly in the code
(`load_oecd_country_risk`), based on the classification published at oecd.org.

## Licences and attribution

Data are used for academic, non-commercial research and remain the property of their
respective providers (SIPRI, UN Comtrade, World Bank, OECD). Please refer to each provider's
terms of use. This repository contains only the analysis code and the derived outputs, not the
source data.

## Repository structure

```
.
├── ad_market_entry_scaffold.py   # main model (run this)
├── data/                         # source data files (not included — see Data sources)
├── outputs/                      # generated tables and charts
└── README.md
```
