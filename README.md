# 🛰️ MODIS & VIIRS Thermal Anomaly Harmonizer
**2026 NASA Space Apps Challenge** | Challenge: *Harmonization of MODIS and VIIRS Hot Spots*

🔗 **Live Application:** https://modis-viirs-harmonizer.onrender.com/

## Overview
A full-stack geospatial analytics dashboard that unifies thermal anomaly detections from NASA's **MODIS (1 km)** and **VIIRS NOAA-20 (375 m)** instruments in **WGS 84 (EPSG:4326)**. Built to resolve cross-sensor spatial resolution mismatches, normalize heterogeneous confidence schemas, and detect anomalous wildfire spikes across Türkiye.

## Core Architecture
- **Ingestion Pipeline (`main.py`):** Queries NASA FIRMS Near-Real-Time (NRT) REST APIs (`MODIS_NRT` & `VIIRS_NOAA20_NRT`) within Türkiye's WGS 84 bounding box (`25.5, 35.8, 44.8, 42.1`) with automatic local CSV fallback resilience.
- **Schema Normalization:** Maps MODIS numeric confidence (`0–100%`) and VIIRS categorical confidence (`l`, `n`, `h`) onto a unified `0.0–1.0` reliability scale alongside Fire Radiative Power (`FRP` in MW).
- **Haversine DBSCAN Clustering:** Converts WGS 84 coordinates to radians (`eps_rad = eps_km / 6371.0`) per acquisition date to deduplicate overlapping cross-sensor detections into FRP-weighted centroids while preserving VIIRS-only micro-hotspots.
- **Burning Activity Calendar:** Aggregates daily harmonized FRP and flags anomalous fire spikes exceeding `mean + 1.5 * std`.

## Local Setup
```bash
pip install -r requirements.txt
export FIRMS_MAP_KEY="your_nasa_firms_key"
uvicorn main:app --reload --port 8000
