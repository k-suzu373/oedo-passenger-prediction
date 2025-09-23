"""build_metro_master.py

Create metro passenger + land price master datasets according to 2025‑06‑25 spec.

Outputs
-------
metro_2011-2021.csv
metro_2022.csv
landprice_2017_2021.csv
landprice_zero_stations.txt
oedo_master_2011_2021.csv
"""

import argparse
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def debug(msg: str) -> None:
    if os.getenv("DEBUG"):
        print(f"[DEBUG] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# XML helpers
# ----------------------------------------------------------------------
def get_text(elem: ET.Element, tag_suffix: str) -> str | None:
    """Return text for first descendant whose local-name matches *tag_suffix*."""
    for child in elem.iter():
        if child.tag.split("}")[-1] == tag_suffix:
            return child.text
    return None


def norm(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


# ----------------------------------------------------------------------
# Passenger extraction
# ----------------------------------------------------------------------
_ALLOWED_TOEI_LINES = {
    "12号線大江戸線",
}

_LINE_RE = re.compile(r"^(\d+)号線")


def _smallest_route(routes: list[str]) -> str:
    """Return routeName with smallest numeric prefix (if any)."""

    def key(name: str):
        m = _LINE_RE.match(name)
        return (int(m.group(1)) if m else 999, name)

    return sorted(routes, key=key)[0]


def _extract_year_value(elem: ET.Element, year: int) -> int | None:
    txt = get_text(elem, f"passengers{year}")
    try:
        return int(txt) if txt is not None else None
    except ValueError:
        return None


def build_passenger_frames(xml_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_17_21, df_2022)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: list[dict] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "TheNumberofTheStationPassengersGettingonandoff":
            continue

        station = norm(get_text(el, "stationName"))
        company = norm(get_text(el, "administrationCompany"))
        route = norm(get_text(el, "routeName"))

        if company not in {"東京都"}:
            continue
        if company == "東京都" and route not in _ALLOWED_TOEI_LINES:
            continue

        code = norm(get_text(el, "stationCode"))
        yearly = {y: _extract_year_value(el, y) for y in range(2011, 2023)}

        rec = {
            "stationName": station,
            "administrationCompany": company,
            "routeName": route,
            "stationCode": code,
        }
        rec.update(yearly)
        records.append(rec)

    if not records:
        raise RuntimeError("No passenger records after filtering.")

    df_raw = pd.DataFrame(records)

    # Aggregate duplicates
    agg_dict: dict = {y: "sum" for y in range(2011, 2023)}
    agg_dict.update(
        {
            "stationCode": "first",
            "routeName": lambda s: _smallest_route(list(s)),
        }
    )
    g = df_raw.groupby(["stationName", "administrationCompany"], as_index=False).agg(
        agg_dict
    )

    rows_17_21: list[dict] = []
    rows_22: list[dict] = []

    for _, row in g.iterrows():
        for year in range(2011, 2022):
            pax = row[year]
            if pd.notna(pax):
                rows_17_21.append(
                    {
                        "stationName": row["stationName"],
                        "administrationCompany": row["administrationCompany"],
                        "routeName": row["routeName"],
                        "stationCode": row["stationCode"],
                        "year": year,
                        "passengers": int(pax),
                    }
                )
        pax22 = row[2022]
        if pd.notna(pax22):
            rows_22.append(
                {
                    "stationName": row["stationName"],
                    "administrationCompany": row["administrationCompany"],
                    "routeName": row["routeName"],
                    "stationCode": row["stationCode"],
                    "year": 2022,
                    "passengers": int(pax22),
                }
            )

    df_17_21 = pd.DataFrame(rows_17_21).sort_values(["stationCode", "year"])
    df_22 = pd.DataFrame(rows_22).sort_values(["stationCode"])
    return df_17_21, df_22


# ----------------------------------------------------------------------
# Land price extraction
# ----------------------------------------------------------------------
_YEAR_TAG_MAP = {
    2011: "postedLandPriceOfH23",
    2012: "postedLandPriceOfH24",
    2013: "postedLandPriceOfH25",
    2014: "postedLandPriceOfH26",
    2015: "postedLandPriceOfH27",
    2016: "postedLandPriceOfH28",
    2017: "postedLandPriceOfH29",
    2018: "postedLandPriceOfH30",
    2019: "postedLandPriceOfR01",
    2020: "postedLandPriceOfR02",
    2021: "postedLandPriceOfR03",
}


def build_landprice_frame(xml_path: str) -> tuple[pd.DataFrame, list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: list[dict] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "LandPrice":
            continue

        station = norm(get_text(el, "nameOfNearestStation"))
        location = norm(get_text(el, "location"))

        for year, tag in _YEAR_TAG_MAP.items():
            txt = get_text(el, tag)
            if txt is None:
                continue
            try:
                price_val = int(txt.replace(",", ""))
            except ValueError:
                continue
            records.append(
                {
                    "nameOfNearestStation": station,
                    "location": location,
                    "year": year,
                    "price": price_val,
                }
            )

    if not records:
        raise RuntimeError("No land‑price records.")

    df_raw = pd.DataFrame(records)

    # 数値化と除外
    df_raw["price"] = pd.to_numeric(df_raw["price"], errors="coerce")
    df_raw = df_raw.dropna(subset=["price"])
    df_raw["price"] = df_raw["price"].astype(float)

    df_avg = (
        df_raw.groupby(["nameOfNearestStation", "year"], as_index=False)
        .agg({"location": "first", "price": "mean"})
        .sort_values(["nameOfNearestStation", "year"])
    )
    # ★ここで再度数値化しつつ、失敗したら落とす
    df_avg["price"] = pd.to_numeric(df_avg["price"], errors="coerce")
    df_avg = df_avg.dropna(subset=["price"])
    df_avg["price"] = df_avg["price"].astype(int)

    zero_flag = (
        df_avg.groupby("nameOfNearestStation")["price"]
        .apply(lambda s: (s == 0).any())
        .reset_index(name="has_zero")
    )
    dropped = zero_flag.loc[zero_flag["has_zero"], "nameOfNearestStation"].tolist()

    df_filtered = df_avg[~df_avg["nameOfNearestStation"].isin(dropped)].copy()
    df_filtered.sort_values(["nameOfNearestStation", "year"], inplace=True)
    return df_filtered, dropped


# ----------------------------------------------------------------------
# Master merge
# ----------------------------------------------------------------------
def build_master(df_pax: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
    merged = df_pax.merge(
        df_price[["nameOfNearestStation", "year", "price"]],
        left_on=["stationName", "year"],
        right_on=["nameOfNearestStation", "year"],
        how="left",
    )
    merged.drop(columns=["nameOfNearestStation"], inplace=True)
    merged.rename(columns={"price": "land_price"}, inplace=True)
    merged.sort_values(["stationCode", "year"], inplace=True)
    return merged


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build metro master dataset (2017‑2021)"
    )
    parser.add_argument("--passenger", default="S12-23.xml", help="Passenger XML path")
    parser.add_argument(
        "--landprice", default="L01-22_13.xml", help="LandPrice XML path"
    )
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Passenger
    df_pax_17_21, df_pax_22 = build_passenger_frames(args.passenger)
    df_pax_17_21.to_csv(
        out_dir / "metro_2017-2021.csv", index=False, encoding="utf_8_sig"
    )
    df_pax_22.to_csv(out_dir / "metro_2022.csv", index=False, encoding="utf_8_sig")
    print(f"✓ metro_2017-2021.csv  ({len(df_pax_17_21)})")
    print(f"✓ metro_2022.csv       ({len(df_pax_22)})")

    # Land price
    df_price, dropped = build_landprice_frame(args.landprice)
    df_price.to_csv(
        out_dir / "landprice_2017_2021.csv", index=False, encoding="utf_8_sig"
    )
    (out_dir / "landprice_zero_stations.txt").write_text(
        "\n".join(dropped), encoding="utf-8"
    )
    print(f"✓ landprice_2017_2021.csv     ({len(df_price)})")
    print(f"✓ landprice_zero_stations.txt ({len(dropped)})")

    # Master
    df_master = build_master(df_pax_17_21, df_price)
    df_master.to_csv(
        out_dir / "oedo_master_2011_2021.csv", index=False, encoding="utf_8_sig"
    )
    print(f"✓ oedo_master_2011_2021.csv  ({len(df_master)})")

    # ---- land_price==0または空白の駅の全年度を削除 ----
    df_master = pd.read_csv(out_dir / "oedo_master_2011_2021.csv")
    zero_stations = df_master.loc[
        (df_master["land_price"].isna()) | (df_master["land_price"] == 0),
        "stationName"
    ].unique().tolist()
    df_master_clean = df_master[~df_master["stationName"].isin(zero_stations)]
    df_master_clean.to_csv(out_dir / "oedo_master_2011_2021_clean.csv", index=False, encoding="utf_8_sig")
    (out_dir / "metro_master_zero_stations.txt").write_text("\\n".join(zero_stations), encoding="utf-8")
    print(f"✓ oedo_master_2011_2021_clean.csv  ({len(df_master_clean)})")
    print(f"✓ oedo_master_zero_stations.txt    ({len(zero_stations)})")


if __name__ == "__main__":
    main()
