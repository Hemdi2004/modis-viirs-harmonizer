import os
import io
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.cluster import DBSCAN

app = FastAPI(title="MODIS & VIIRS Thermal Anomaly Harmonizer")
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Bounding box for Türkiye in WGS 84 (min_lon, min_lat, max_lon, max_lat)
TURKEY_BBOX = "25.5,35.8,44.8,42.1"
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")


def fetch_firms_data(days: int = 5):
    """Fetches MODIS (1km) and VIIRS (375m) NRT data from NASA FIRMS API with local CSV fallback."""
    modis_df, viirs_df = None, None

    if FIRMS_MAP_KEY:
        try:
            modis_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/MODIS_NRT/{TURKEY_BBOX}/{days}"
            viirs_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_NOAA20_NRT/{TURKEY_BBOX}/{days}"
            m_res = requests.get(modis_url, timeout=10)
            v_res = requests.get(viirs_url, timeout=10)
            if m_res.ok and v_res.ok and "latitude" in m_res.text and "latitude" in v_res.text:
                modis_df = pd.read_csv(io.StringIO(m_res.text))
                viirs_df = pd.read_csv(io.StringIO(v_res.text))
        except Exception:
            modis_df, viirs_df = None, None

    if modis_df is None or viirs_df is None:
        modis_df = pd.read_csv("data/modis.csv")
        viirs_df = pd.read_csv("data/viirs.csv")

    for df in (modis_df, viirs_df):
        if not df.empty:
            df.query("25.5 <= longitude <= 44.8 and 35.8 <= latitude <= 42.1", inplace=True)

    return modis_df, viirs_df


def normalize_confidence(val, sensor: str) -> float:
    if sensor == "MODIS":
        try:
            return round(float(val) / 100.0, 2)
        except ValueError:
            return 0.50
    mapping = {"l": 0.35, "low": 0.35, "n": 0.70, "nominal": 0.70, "h": 0.95, "high": 0.95}
    return mapping.get(str(val).strip().lower(), 0.60)


def harmonize_sensors(modis_df: pd.DataFrame, viirs_df: pd.DataFrame, eps_km: float = 1.0):
    cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]
    m = modis_df[cols].copy() if not modis_df.empty else pd.DataFrame(columns=cols)
    m["sensor"] = "MODIS"
    m["norm_conf"] = m["confidence"].apply(lambda x: normalize_confidence(x, "MODIS"))

    v = viirs_df[cols].copy() if not viirs_df.empty else pd.DataFrame(columns=cols)
    v["sensor"] = "VIIRS"
    v["norm_conf"] = v["confidence"].apply(lambda x: normalize_confidence(x, "VIIRS"))

    combined = pd.concat([m, v], ignore_index=True)
    if combined.empty:
        return {"stats": {"raw_modis_count": 0, "raw_viirs_count": 0, "raw_total": 0, "harmonized_count": 0, "deduplication_rate_pct": 0.0, "eps_km": eps_km}, "hotspots": [], "calendar": []}

    combined["frp"] = pd.to_numeric(combined["frp"], errors="coerce").fillna(1.0).clip(lower=0.1)

    harmonized_records = []
    eps_rad = eps_km / 6371.0

    for acq_date, group in combined.groupby("acq_date"):
        coords_rad = np.radians(group[["latitude", "longitude"]].values)
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords_rad)
        group = group.copy()
        group["cluster"] = db.labels_

        for _, cluster_df in group.groupby("cluster"):
            weights = cluster_df["frp"].values
            w_lat = float(np.average(cluster_df["latitude"], weights=weights))
            w_lon = float(np.average(cluster_df["longitude"], weights=weights))
            total_frp = float(np.max(weights)) if len(set(cluster_df["sensor"])) > 1 else float(np.sum(weights))
            max_conf = float(cluster_df["norm_conf"].max())
            sensors_present = sorted(cluster_df["sensor"].unique().tolist())
            sensor_label = "MODIS+VIIRS" if len(sensors_present) > 1 else sensors_present[0]

            harmonized_records.append({
                "latitude": round(w_lat, 5),
                "longitude": round(w_lon, 5),
                "acq_date": str(acq_date),
                "frp": round(total_frp, 2),
                "confidence": round(max_conf, 2),
                "sensor_source": sensor_label,
                "merged_detections": int(len(cluster_df))
            })

    harm_df = pd.DataFrame(harmonized_records)

    calendar_data = []
    if not harm_df.empty:
        daily = harm_df.groupby("acq_date").agg(
            total_frp=("frp", "sum"),
            event_count=("frp", "count"),
            avg_conf=("confidence", "mean")
        ).reset_index().sort_values("acq_date")

        mean_frp = daily["total_frp"].mean()
        std_frp = daily["total_frp"].std(ddof=0) if len(daily) > 1 else 0.0
        spike_threshold = mean_frp + 1.5 * std_frp

        for _, row in daily.iterrows():
            calendar_data.append({
                "date": row["acq_date"],
                "total_frp": round(float(row["total_frp"]), 2),
                "event_count": int(row["event_count"]),
                "avg_conf": round(float(row["avg_conf"]), 2),
                "is_spike": bool(row["total_frp"] > spike_threshold and std_frp > 0),
                "threshold": round(float(spike_threshold), 2)
            })

    return {
        "stats": {
            "raw_modis_count": int(len(m)),
            "raw_viirs_count": int(len(v)),
            "raw_total": int(len(combined)),
            "harmonized_count": int(len(harm_df)),
            "deduplication_rate_pct": round((1 - len(harm_df) / max(len(combined), 1)) * 100, 1),
            "eps_km": eps_km
        },
        "hotspots": harmonized_records,
        "calendar": calendar_data
    }


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/api/harmonize")
def get_harmonized_data(
    eps_km: float = Query(1.0, ge=0.2, le=5.0),
    days: int = Query(5, ge=1, le=5)
):
    modis_df, viirs_df = fetch_firms_data(days=days)
    return harmonize_sensors(modis_df, viirs_df, eps_km=eps_km)