# pv-curtailment-brazil

Code and result tables for the manuscript *Grid-driven versus energy-surplus curtailment of utility-scale photovoltaics in Brazil: plant-level evidence and day-ahead forecasting with public operator data, 2024–2026* (submitted to *Solar Energy*, 2026).

Half-hourly constrained-off records of the Brazilian system operator (ONS) for every centrally dispatched photovoltaic plant are aggregated to a plant-hour panel and split by reason (energy surplus ENE, reliability REL, transmission unavailability CNF) and origin (system SIS vs local LOC). The study quantifies how much curtailment is grid-driven, associates daily curtailment with net load and month in regressions with HAC errors, and forecasts next-day plant-level curtailment with LightGBM against persistence, 7-day mean and a linear GLM baseline, with sensitivity analyses for the unbalanced panel, recalibration and a 2-day lag.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Panel construction (`painel.py`), descriptive statistics and daily regressions (`descritiva.py`, `analises_complementares.py`), day-ahead forecasts and sensitivity runs (`previsao.py`, `previsao_detalhe.py`, `previsao_extra.py`), figures and the manuscript builder |
| `dados/` | ONS data dictionary of the constrained-off dataset. Raw ONS parquet files go here but are not redistributed |
| `resultados/` | All tables behind the paper: reason × origin, monthly/annual/hourly/state/plant summaries, regressions, forecast skill, bootstrap CIs, balanced-panel, recalibration, D−2, linear-baseline and permutation-importance results |
| `figuras/` | Figs. 1–4 of the manuscript (PNG; the submission uses 300 dpi TIFF) |

## Pipeline

1. Download the ONS open data (see **Data**) into `dados/ons_pv/`, `dados/ons_eol/` and `dados/carga/`.
2. `scripts/painel.py` — builds the plant-hour panel and the system-hour table → `dados/painel_pv_horario.parquet`, `dados/sistema_horario.parquet`, `resultados/usinas_pv.csv`.
3. `scripts/descritiva.py` — monthly, annual, hourly, state and plant summaries and daily regressions → `resultados/mensal.csv`, `anual.csv`, `por_uf.csv`, `por_usina.csv`, `reg_diaria_*.csv`.
4. `scripts/previsao.py`, `previsao_detalhe.py` — day-ahead LightGBM forecasts with monthly expanding-window evaluation → `resultados/previsao_*.csv`, `dados/painel_modelo.parquet`, `dados/previsoes_detalhe.parquet`.
5. `scripts/analises_complementares.py` — reason × origin table, REL hours above the 30.5-h threshold, description classes, balanced panel, fractional logit, concentration, bootstrap of skill, AUC thresholds and Fig. 4.
6. `scripts/previsao_extra.py` — revision sensitivity runs: balanced panel, past-only cap, recalibration, D−2 lag, linear GLM baseline, LOC/SIS grid targets, permutation importance.
7. `scripts/figuras.py` — Figs. 1–3; `scripts/build_docx.py` — manuscript in the journal format (source not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Constrained-off records of photovoltaic plants (`restricao_coff_fotovoltaica`), wind plants and load: Operador Nacional do Sistema Elétrico (ONS), Dados Abertos (https://dados.ons.org.br; open data, CC-BY 4.0), April 2024 – 2026. The data dictionary is included in `dados/`.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.
