"""Turn the raw Olist files into the processed tables: prefixes, sites and folded demand."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from irp2e.config import Params

EARTH_RADIUS_KM = 6371.0  # mean Earth radius (IUGG value 6371.0088 km, rounded)
DAY_COLUMNS = [f"d{day}" for day in range(7)]


def haversine_km(lat: np.ndarray, lng: np.ndarray, to_lat: float, to_lng: float) -> np.ndarray:
    """Straight-line (great-circle) km from each (lat, lng) to one point."""
    phi, to_phi = np.radians(lat), np.radians(to_lat)
    half_dlat = np.square(np.sin((to_phi - phi) / 2))
    half_dlng = np.square(np.sin(np.radians(to_lng - lng) / 2))
    a = half_dlat + np.multiply(np.multiply(np.cos(phi), np.cos(to_phi)), half_dlng)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def read_items(raw: Path) -> pd.DataFrame:
    """Order items with their product weight: order_id, product_id, price, product_weight_g."""
    items = pd.read_csv(
        raw / "olist_order_items_dataset.csv", usecols=["order_id", "product_id", "price"]
    )
    products = pd.read_csv(
        raw / "olist_products_dataset.csv", usecols=["product_id", "product_weight_g"]
    )
    return items.merge(products, on="product_id", how="left")


def fold_window(params: Params) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First day and the day after the last day of the fold window."""
    start = pd.Timestamp(params.fold_start)
    return start, start + pd.Timedelta(days=7 * params.fold_weeks)


def load_sp_orders(raw: Path, items: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Orders of the state with the kept status: order_id, prefix, city, day, kg, n_missing."""
    orders = pd.read_csv(
        raw / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    )
    customers = pd.read_csv(
        raw / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
        dtype={"customer_zip_code_prefix": str},
    )
    orders = orders.merge(customers, on="customer_id")
    orders = orders[
        (orders["customer_state"] == params.state) & (orders["order_status"] == params.order_status)
    ]
    weights = items.assign(missing=items["product_weight_g"].isna())
    per_order = weights.groupby("order_id").agg(
        grams=("product_weight_g", "sum"), n_missing_weight=("missing", "sum")
    )
    # the inner merge drops orders without items
    orders = orders.merge(per_order, left_on="order_id", right_index=True)
    return pd.DataFrame(
        {
            "order_id": orders["order_id"],
            "prefix": orders["customer_zip_code_prefix"],
            "city": orders["customer_city"],
            "day": pd.to_datetime(orders["order_purchase_timestamp"]).dt.normalize(),
            "kg": orders["grams"] / 1000.0,
            "n_missing_weight": orders["n_missing_weight"],
        }
    ).reset_index(drop=True)


def drop_missing_weight(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep only orders whose every item has a weight."""
    return orders[orders["n_missing_weight"] == 0].drop(columns="n_missing_weight")


def prefix_coordinates(raw: Path, params: Params) -> pd.DataFrame:
    """Median lat and lng per zip prefix, from geolocation rows of the state inside its box."""
    geo = pd.read_csv(
        raw / "olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": str}
    )
    inside = geo[
        (geo["geolocation_state"] == params.state)
        & geo["geolocation_lat"].between(params.sp_box_lat[0], params.sp_box_lat[1])
        & geo["geolocation_lng"].between(params.sp_box_lng[0], params.sp_box_lng[1])
    ]
    medians = inside.groupby("geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]
    ].median()
    return pd.DataFrame(
        {
            "prefix": medians.index,
            "lat": medians["geolocation_lat"].to_numpy(),
            "lng": medians["geolocation_lng"].to_numpy(),
        }
    )


def prefix_table(orders: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """One row per prefix with a coordinate: prefix, city, lat, lng, n_orders, kg."""
    per_prefix = orders.groupby("prefix").agg(
        city=("city", lambda cities: cities.mode().iloc[0]),  # most common city name
        n_orders=("order_id", "count"),
        kg=("kg", "sum"),
    )
    table = coords.merge(per_prefix, left_on="prefix", right_index=True)
    return table[["prefix", "city", "lat", "lng", "n_orders", "kg"]].sort_values("prefix")


def choose_warehouse(raw: Path) -> dict:
    """The seller zip prefix with the most sellers (ties: lower prefix)."""
    sellers = pd.read_csv(raw / "olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    counts = sellers.groupby("seller_zip_code_prefix").agg(
        city=("seller_city", lambda cities: cities.mode().iloc[0]),
        n_sellers=("seller_id", "count"),
    )
    counts = counts.reset_index().sort_values(
        ["n_sellers", "seller_zip_code_prefix"], ascending=[False, True]
    )
    best = counts.iloc[0]
    return {
        "prefix": best["seller_zip_code_prefix"],
        "city": best["city"],
        "n_sellers": int(best["n_sellers"]),
    }


def rank_hubs(orders: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Cities ranked by kg, each at its prefix with the most orders: city, prefix, lat, lng, kg."""
    city_kg = orders.groupby("city")["kg"].sum()
    located = orders[orders["prefix"].isin(coords["prefix"])]
    counts = located.groupby(["city", "prefix"]).size().reset_index(name="n_orders")
    counts = counts.sort_values(["city", "n_orders", "prefix"], ascending=[True, False, True])
    hubs = counts.drop_duplicates("city").merge(coords, on="prefix")
    hubs["kg"] = hubs["city"].map(city_kg)
    hubs = hubs.sort_values(["kg", "city"], ascending=[False, True])
    return hubs[["city", "prefix", "lat", "lng", "kg"]].reset_index(drop=True)


def fold_demand(orders: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Rule D1: kg per prefix and weekday, summed over the fold window: prefix, d0 .. d6."""
    start, end = fold_window(params)
    window = orders[(orders["day"] >= start) & (orders["day"] < end)]
    weekday = (window["day"] - start).dt.days % 7
    table = window.assign(weekday=weekday).pivot_table(
        index="prefix", columns="weekday", values="kg", aggfunc="sum", fill_value=0.0
    )
    table = table.reindex(index=sorted(orders["prefix"].unique()), columns=range(7), fill_value=0.0)
    table.columns = DAY_COLUMNS
    return table.rename_axis("prefix").reset_index()


def goods_value_per_kg(items: pd.DataFrame, orders: pd.DataFrame, params: Params) -> float:
    """Sum of item prices over sum of item kg, for the kept orders in the fold window."""
    start, end = fold_window(params)
    window_ids = orders.loc[(orders["day"] >= start) & (orders["day"] < end), "order_id"]
    window = items[items["order_id"].isin(window_ids)]
    return float(np.sum(window["price"]) / (np.sum(window["product_weight_g"]) / 1000.0))


def build_pool(
    prefixes: pd.DataFrame, hubs: pd.DataFrame, n_hubs: int, params: Params
) -> pd.DataFrame:
    """Rule P1: busiest prefixes within the radius of their nearest of the first n_hubs hubs."""
    first = hubs.iloc[:n_hubs]
    km = np.column_stack(
        [
            haversine_km(prefixes["lat"].to_numpy(), prefixes["lng"].to_numpy(), lat, lng)
            for lat, lng in zip(first["lat"], first["lng"], strict=True)
        ]
    )
    pool = prefixes.assign(hub=np.argmin(km, axis=1), hub_km=np.min(km, axis=1))
    pool = pool[(pool["hub_km"] <= params.pool_radius_km) & ~pool["prefix"].isin(first["prefix"])]
    pool = pool.sort_values(["n_orders", "kg", "prefix"], ascending=[False, False, True])
    return pool.head(params.pool_size).reset_index(drop=True)


def check_value_per_kg(computed: float, params: Params) -> None:
    """Stop when the data gives another goods value than the config holds."""
    if abs(computed - params.goods_value_per_kg) > 0.001:
        raise ValueError(
            f"goods value per kg from the data is {computed:.4f} BRL, config holds "
            f"{params.goods_value_per_kg}; update config/params.toml"
        )


def read_prefixes(path: Path) -> pd.DataFrame:
    """Read prefixes.csv; zip prefixes stay strings, so 01026 keeps its leading zero."""
    return pd.read_csv(path, dtype={"prefix": str, "city": str})


def read_demand_week(path: Path) -> pd.DataFrame:
    """Read demand_week.csv indexed by prefix (a string)."""
    return pd.read_csv(path, dtype={"prefix": str}).set_index("prefix")


def read_sites(path: Path) -> dict:
    """Read sites.json: warehouse, ranked hubs, goods value, filters."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run_preprocess(raw: Path, out: Path, params: Params) -> None:
    """Write prefixes.csv, demand_week.csv and sites.json to the processed folder."""
    items = read_items(raw)
    all_orders = load_sp_orders(raw, items, params)
    orders = drop_missing_weight(all_orders)
    coords = prefix_coordinates(raw, params)
    prefixes = prefix_table(orders, coords)
    warehouse = choose_warehouse(raw)
    if warehouse["prefix"] not in set(coords["prefix"]):
        raise ValueError(f"warehouse prefix {warehouse['prefix']} has no coordinate")
    warehouse_row = coords[coords["prefix"] == warehouse["prefix"]].iloc[0]
    n_hubs = max(max(spec.hubs) for spec in params.instance_sets.values())
    hubs = rank_hubs(orders, coords).head(n_hubs)
    demand = fold_demand(orders, params)
    demand = demand[demand["prefix"].isin(prefixes["prefix"])]
    value_per_kg = goods_value_per_kg(items, orders, params)
    check_value_per_kg(value_per_kg, params)

    out.mkdir(parents=True, exist_ok=True)
    prefixes.round({"lat": 6, "lng": 6, "kg": 3}).to_csv(out / "prefixes.csv", index=False)
    demand.round(3).to_csv(out / "demand_week.csv", index=False)
    located = orders["prefix"].isin(prefixes["prefix"])
    sites = {
        "warehouse": {
            "prefix": warehouse["prefix"],
            "city": warehouse["city"],
            "n_sellers": warehouse["n_sellers"],
            "lat": round(float(warehouse_row["lat"]), 6),
            "lng": round(float(warehouse_row["lng"]), 6),
        },
        "hubs": [
            {
                "city": row.city,
                "prefix": row.prefix,
                "lat": round(float(row.lat), 6),
                "lng": round(float(row.lng), 6),
                "kg": round(float(row.kg), 3),
            }
            for row in hubs.itertuples()
        ],
        "goods_value_per_kg": value_per_kg,
        "filters": {
            "state": params.state,
            "order_status": params.order_status,
            "sp_box_lat": list(params.sp_box_lat),
            "sp_box_lng": list(params.sp_box_lng),
            "fold_start": params.fold_start.isoformat(),
            "fold_weeks": params.fold_weeks,
            "orders_with_items": len(all_orders),
            "orders_dropped_missing_weight": len(all_orders) - len(orders),
            "orders_dropped_no_coordinate": int(np.sum(~located)),
            "prefixes_dropped_no_coordinate": int(orders.loc[~located, "prefix"].nunique()),
            "prefixes_kept": len(prefixes),
        },
    }
    with open(out / "sites.json", "w", encoding="utf-8") as file:
        json.dump(sites, file, indent=2)
