# FP&A Variance Analysis Tool

An end-to-end Python solution for automated FP&A variance analysis. The project ingests budget and actual files from CSV or Excel, cleans and reconciles the data, identifies the main variance drivers, generates commentary, and publishes both a dashboard and printable reports.

## What It Does

- Loads budget and actual financial data from CSV or Excel files.
- Standardizes dates, categories, and numeric values.
- Handles duplicates, missing data, and outlier-style anomalies.
- Calculates absolute and percentage variance.
- Identifies top variance drivers with contribution and robust z-score logic.
- Produces executive commentary using template-based NLP.
- Builds HTML and PDF reports with charts.
- Serves an interactive Streamlit dashboard with drill-down filters.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the demo data:

```bash
python scripts/generate_sample_data.py
```

3. Run the batch pipeline:

```bash
python run_pipeline.py --budget sample_data/budget_demo.csv --actuals sample_data/actuals_demo.csv --output outputs/demo_run
```

4. Launch the dashboard:

```bash
streamlit run app.py
```

Optional: validate a dataset schema before analysis:

```bash
python validate_dataset.py --file templates/validation_valid.csv
```

The dashboard supports three input modes:

- Upload budget and actual files directly in the sidebar.
- Use the bundled sample data.
- Enter local file paths manually.

It also supports direct exports from the filtered dashboard view:

- Filtered variance CSV
- Filtered driver CSV
- Drill-down CSV
- ZIP pack with commentary, quality outputs, anomalies, and filtered tables
- Management pack HTML for the current filtered view
- Management pack PDF for the current filtered view
- Management pack ZIP with reports, charts, commentary, and supporting tables

Optional AI assistant:

- Add `OPENAI_API_KEY` in your environment or paste it into the sidebar
- Ask follow-up questions about the current filtered FP&A view directly in the dashboard

## Expected Input Schema

Each budget or actual file should contain:

- `date`
- `department`
- `region`
- `product_line`
- `account`
- `amount`

Dates are normalized to month-start. The sample dataset uses positive revenue and negative costs.

Common header aliases are also accepted. For example:

- `dept` or `cost_center` -> `department`
- `market` or `country` -> `region`
- `product` or `lob` -> `product_line`
- `gl_account`, `acct`, or `category` -> `account`
- `value`, `balance`, `actual_amount`, or `budget_amount` -> `amount`

## Outputs

The pipeline writes the following to the chosen output folder:

- `reconciled_variance.csv`
- `top_drivers.csv`
- `driver_scores.csv`
- `anomalies.csv`
- `data_quality_summary.csv`
- `system_validation_report.csv`
- `full_fpa_report.txt`
- `report.html`
- `report.pdf`
- `charts/`

## Business Impact

This workflow reduces manual monthly variance analysis effort, improves consistency of management commentary, and helps finance teams move faster from reconciliation to decision support. The driver scoring surfaces high-impact variances so analysts can focus on issues that are both material and statistically unusual.
