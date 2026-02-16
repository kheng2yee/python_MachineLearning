import pandas as pd
import numpy as np

# =========================================================
# Task 1 — Data Quality Assessment
# =========================================================
def data_quality_report(df, numeric_cols=None):
    numeric_cols = set(numeric_cols or [])
    duplicate_count = df.duplicated().sum()

    report = []
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100

        invalid_count = None
        if col in numeric_cols:
            invalid_count = (
                pd.to_numeric(df[col], errors="coerce")
                .le(0)
                .sum()
            )

        report.append({
            "column": col,
            "missing_pct": round(missing_pct, 2),
            "invalid_count": invalid_count,
            "duplicate_rows": duplicate_count
        })

    return pd.DataFrame(report)


# =========================================================
# Task 2 — Standardisation & Cleaning
# =========================================================
def clean_text(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )
    return df


LINE_MAP = {
    "L1": "LINE01", "LINE-1": "LINE01", "LINE1": "LINE01",
    "L2": "LINE02", "LINE-2": "LINE02", "LINE2": "LINE02",
    "L3": "LINE03", "LINE-3": "LINE03", "LINE3": "LINE03",
}

def normalize_production_lines(df):
    df = df.copy()
    df["production_line"] = df["production_line"].map(
        lambda x: LINE_MAP.get(x, x)
    )
    return df


# =========================================================
# Task 3 — Business Rule Validation
# =========================================================
def validate_production(df):
    return df[
        (df["units_produced"] > 0) &
        (df["planned_units"].notna()) &
        (df["planned_units"] >= df["units_produced"])
    ]


def validate_quality(df):
    return df[
        (df["units_inspected"].notna()) &
        (df["units_inspected"] > 0) &
        (df["defect_units"].notna()) &
        (df["defect_units"] >= 0) &
        (df["defect_units"] <= df["units_inspected"])
    ]


# =========================================================
# Task 4 — Product Master Mapping
# =========================================================
def map_product_master(df, product_master):
    df = df.copy()
    product_master = product_master.copy()

    df = clean_text(df, ["product_code"])
    product_master = clean_text(
        product_master,
        ["product_code_raw", "product_code_std", "product_family"]
    )

    merged = df.merge(
        product_master,
        left_on="product_code",
        right_on="product_code_raw",
        how="left"
    )

    merged = merged[merged["product_code_std"].notna()]
    merged.drop(columns=["product_code_raw"], inplace=True)

    return merged


# =========================================================
# Task 5 — Monthly Aggregation & Alignment
# =========================================================
def add_month(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    return df


def aggregate_production(df):
    return (
        df
        .groupby(
            ["plant_id", "production_line", "product_code_std", "product_family", "month"],
            as_index=False
        )
        .agg(
            units_produced=("units_produced", "sum"),
            planned_units=("planned_units", "sum"),
            downtime_minutes=("downtime_minutes", "sum")
        )
    )


def aggregate_quality(df):
    return (
        df
        .groupby(
            ["plant_id", "product_code_std", "product_family", "month"],
            as_index=False
        )
        .agg(
            units_inspected=("units_inspected", "sum"),
            defect_units=("defect_units", "sum")
        )
    )


# =========================================================
# Task 6 — KPI Engineering
# =========================================================
def calculate_kpis(df):
    df = df.copy()

    df["production_efficiency"] = df["units_produced"] / df["planned_units"]
    df["defect_rate"] = df["defect_units"] / df["units_inspected"]
    df["downtime_per_unit"] = df["downtime_minutes"] / df["units_produced"]

    df["performance_rank"] = (
        df
        .groupby(["plant_id", "month"])["production_efficiency"]
        .rank(ascending=False, method="dense")
    )

    return df


# =========================================================
# Task 7 — Risk Scoring
# =========================================================
def risk_scoring(df):
    df = df.copy()

    efficiency = df["production_efficiency"].fillna(0)
    defect_rate = df["defect_rate"].fillna(0)
    downtime = df["downtime_per_unit"].fillna(0)

    df["risk_score"] = (
        (1 - efficiency) * 50 +
        defect_rate * 30 +
        downtime * 20
    ).clip(0, 100)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[0, 30, 60, 100],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return df


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    print("Loading data...")
    production_df = pd.read_csv("production_data.csv")
    quality_df = pd.read_csv("quality_data.csv")
    product_master = pd.read_csv("product_master.csv")

    print("Running data quality checks...")
    prod_dq = data_quality_report(
        production_df,
        ["units_produced", "planned_units", "downtime_minutes"]
    )
    qual_dq = data_quality_report(
        quality_df,
        ["units_inspected", "defect_units"]
    )

    print("Cleaning and standardising data...")
    production_df = clean_text(
        production_df,
        ["plant_id", "production_line", "product_code"]
    )
    production_df = normalize_production_lines(production_df)

    quality_df = clean_text(
        quality_df,
        ["plant_id", "product_code"]
    )

    production_df["units_produced"] = pd.to_numeric(production_df["units_produced"], errors="coerce")
    production_df["planned_units"] = pd.to_numeric(production_df["planned_units"], errors="coerce")
    production_df["downtime_minutes"] = pd.to_numeric(
        production_df["downtime_minutes"], errors="coerce"
    ).fillna(0)

    quality_df["units_inspected"] = pd.to_numeric(quality_df["units_inspected"], errors="coerce")
    quality_df["defect_units"] = pd.to_numeric(quality_df["defect_units"], errors="coerce")

    print("Applying business rules...")
    production_df = validate_production(production_df)
    quality_df = validate_quality(quality_df)

    print("Mapping product master...")
    production_df = map_product_master(production_df, product_master)
    quality_df = map_product_master(quality_df, product_master)

    print("Aggregating monthly data...")
    production_df = add_month(production_df, "production_date")
    quality_df = add_month(quality_df, "inspection_date")

    prod_monthly = aggregate_production(production_df)
    qual_monthly = aggregate_quality(quality_df)

    print("Merging datasets...")
    final_df = prod_monthly.merge(
        qual_monthly,
        on=["plant_id", "product_code_std", "product_family", "month"],
        how="left"
    )

    print("Calculating KPIs and risk...")
    final_df = calculate_kpis(final_df)
    final_df = risk_scoring(final_df)

    print("Saving outputs...")
    final_df.to_csv("final_monthly_dataset.csv", index=False)
    prod_dq.to_csv("production_data_quality_report.csv", index=False)
    qual_dq.to_csv("quality_data_quality_report.csv", index=False)

    print("Pipeline completed successfully!")
    print(f"Final dataset rows: {len(final_df)}")


if __name__ == "__main__":
    main()
