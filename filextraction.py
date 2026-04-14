"""
F1 Data Collection Script
DSA 210 Project — Formula 1 Race Winner Prediction

Sources:
  1. Kaggle F1 World Championship dataset (primary)
  2. Jolpica/Ergast API (circuit details, qualifying enrichment)
  3. OpenWeatherMap API (optional — weather per race weekend)

Run: python f1_data_collection.py
"""

import os
import time
import json
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KAGGLE_DATASET = "rohanrao/formula-1-world-championship-1950-2020"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ERGAST_BASE = "https://api.jolpi.ca/ergast/f1"  # Jolpica mirror of Ergast API

# Optional: add your OpenWeatherMap key here for weather data
OPENWEATHER_API_KEY = ""  # e.g. "abc123def456"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ─────────────────────────────────────────────
# STEP 1: KAGGLE DOWNLOAD
# ─────────────────────────────────────────────

def download_kaggle_dataset():
    log("\n=== STEP 1: Downloading Kaggle F1 Dataset ===")

    # Check kaggle is configured
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log("ERROR: kaggle.json not found at ~/.kaggle/kaggle.json")
        log("Please follow SETUP_GUIDE.md Step 2 to get your API key.")
        return False

    zip_path = RAW_DIR / "f1_dataset.zip"

    if not zip_path.exists():
        log(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")
        log("Download complete.")
    else:
        log("Kaggle dataset already downloaded. Skipping.")

    # Verify key files exist
    expected_files = [
        "races.csv", "results.csv", "drivers.csv",
        "constructors.csv", "qualifying.csv", "circuits.csv",
        "driver_standings.csv", "constructor_standings.csv",
        "lap_times.csv", "pit_stops.csv"
    ]
    missing = [f for f in expected_files if not (RAW_DIR / f).exists()]
    if missing:
        log(f"Warning: missing files after download: {missing}")
    else:
        log(f"All {len(expected_files)} expected CSV files found.")
    return True


# ─────────────────────────────────────────────
# STEP 2: LOAD & PREVIEW KAGGLE DATA
# ─────────────────────────────────────────────

def load_kaggle_data():
    log("\n=== STEP 2: Loading Kaggle CSVs ===")
    dfs = {}
    files = {
        "races":                  "races.csv",
        "results":                "results.csv",
        "drivers":                "drivers.csv",
        "constructors":           "constructors.csv",
        "qualifying":             "qualifying.csv",
        "circuits":               "circuits.csv",
        "driver_standings":       "driver_standings.csv",
        "constructor_standings":  "constructor_standings.csv",
        "lap_times":              "lap_times.csv",
    }
    for name, filename in files.items():
        path = RAW_DIR / filename
        if path.exists():
            dfs[name] = pd.read_csv(path, na_values=["\\N", "N/A", ""])
            log(f"  Loaded {filename}: {dfs[name].shape}")
        else:
            log(f"  MISSING: {filename}")
    return dfs


# ─────────────────────────────────────────────
# STEP 3: ERGAST API — CIRCUIT CHARACTERISTICS
# ─────────────────────────────────────────────

def fetch_ergast_circuits():
    """
    Fetch circuit details from the Jolpica/Ergast API.
    Returns a DataFrame with circuitId, country, lat, long, url.
    """
    log("\n=== STEP 3: Fetching Circuit Data from Ergast API ===")
    url = f"{ERGAST_BASE}/circuits.json?limit=100"
    cache_path = RAW_DIR / "ergast_circuits.json"

    if cache_path.exists():
        log("  Using cached ergast_circuits.json")
        with open(cache_path) as f:
            data = json.load(f)
    else:
        log(f"  GET {url}")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)

    circuits_raw = data["MRData"]["CircuitTable"]["Circuits"]
    rows = []
    for c in circuits_raw:
        rows.append({
            "circuitId":     c["circuitId"],
            "circuitName":   c["circuitName"],
            "country":       c["Location"]["country"],
            "locality":      c["Location"]["locality"],
            "lat":           float(c["Location"]["lat"]),
            "long":          float(c["Location"]["long"]),
            "wikipedia_url": c["url"],
        })
    df = pd.DataFrame(rows)
    log(f"  Fetched {len(df)} circuits from Ergast.")
    return df


# ─────────────────────────────────────────────
# STEP 4: ERGAST API — RECENT QUALIFYING (2021–2024)
# ─────────────────────────────────────────────

def fetch_ergast_qualifying(years=range(2021, 2025)):
    """
    Kaggle dataset may be missing recent years.
    This fills gaps by pulling qualifying from Ergast API.
    """
    log("\n=== STEP 4: Fetching Recent Qualifying Data (2021–2024) ===")
    all_rows = []
    cache_path = RAW_DIR / "ergast_qualifying_recent.json"

    if cache_path.exists():
        log("  Using cached ergast_qualifying_recent.json")
        with open(cache_path) as f:
            all_rows = json.load(f)
    else:
        for year in years:
            url = f"{ERGAST_BASE}/{year}/qualifying.json?limit=500"
            log(f"  Fetching {year} qualifying...")
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                races = r.json()["MRData"]["RaceTable"]["Races"]
                for race in races:
                    for q in race.get("QualifyingResults", []):
                        all_rows.append({
                            "year":          year,
                            "round":         race["round"],
                            "raceName":      race["raceName"],
                            "circuitId":     race["Circuit"]["circuitId"],
                            "driverId":      q["Driver"]["driverId"],
                            "constructorId": q["Constructor"]["constructorId"],
                            "number":        q["number"],
                            "position":      int(q["position"]),
                            "Q1":            q.get("Q1", None),
                            "Q2":            q.get("Q2", None),
                            "Q3":            q.get("Q3", None),
                        })
                time.sleep(0.3)  # be polite to the API
            except Exception as e:
                log(f"  Warning: failed to fetch {year}: {e}")

        with open(cache_path, "w") as f:
            json.dump(all_rows, f)

    df = pd.DataFrame(all_rows)
    log(f"  Fetched {len(df)} qualifying rows (2021–2024).")
    return df


# ─────────────────────────────────────────────
# STEP 5: BUILD MASTER DATASET
# ─────────────────────────────────────────────

def build_master_dataset(dfs, ergast_circuits):
    log("\n=== STEP 5: Building Master Dataset ===")

    races = dfs["races"].copy()
    results = dfs["results"].copy()
    drivers = dfs["drivers"].copy()
    constructors = dfs["constructors"].copy()
    qualifying = dfs["qualifying"].copy()
    circuits = dfs["circuits"].copy()
    driver_standings = dfs["driver_standings"].copy()
    constructor_standings = dfs["constructor_standings"].copy()

    # ── Clean up results ──
    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["points"] = pd.to_numeric(results["points"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["laps"] = pd.to_numeric(results["laps"], errors="coerce")
    results["milliseconds"] = pd.to_numeric(results["milliseconds"], errors="coerce")
    results["is_winner"] = (results["positionOrder"] == 1).astype(int)

    # ── Clean up races ──
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    races["year"] = races["date"].dt.year

    # ── Qualifying: keep only pole position (position == 1) for merge ──
    if "position" in qualifying.columns:
        qualifying["position"] = pd.to_numeric(qualifying["position"], errors="coerce")
        pole = qualifying[qualifying["position"] == 1][["raceId", "driverId", "q1", "q2", "q3"]].copy()
        pole.columns = ["raceId", "pole_driverId", "pole_q1", "pole_q2", "pole_q3"]
    else:
        pole = pd.DataFrame()

    # ── Driver info: name, nationality ──
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    driver_info = drivers[["driverId", "driver_name", "nationality", "driverRef"]].copy()

    # ── Constructor info ──
    constructor_info = constructors[["constructorId", "name", "nationality"]].copy()
    constructor_info.columns = ["constructorId", "constructor_name", "constructor_nationality"]

    # ── Pre-race standings (championship points entering the race) ──
    # We want the standings BEFORE each race, so we use standings from the previous round
    driver_standings["points"] = pd.to_numeric(driver_standings["points"], errors="coerce")
    driver_standings["position"] = pd.to_numeric(driver_standings["position"], errors="coerce")
    driver_standings_pre = driver_standings[["raceId", "driverId", "points", "position", "wins"]].copy()
    driver_standings_pre.columns = ["raceId", "driverId", "champ_points_pre", "champ_pos_pre", "champ_wins_pre"]

    constructor_standings["points"] = pd.to_numeric(constructor_standings["points"], errors="coerce")
    constructor_standings_pre = constructor_standings[["raceId", "constructorId", "points", "position"]].copy()
    constructor_standings_pre.columns = ["raceId", "constructorId", "constructor_points_pre", "constructor_pos_pre"]

    # ── Circuits: merge Kaggle + Ergast ──
    circuits["circuitId"] = circuits["circuitId"].astype(str)
    ergast_circuits["circuitId"] = ergast_circuits["circuitId"].astype(str)

    circuit_merged = circuits.merge(
        ergast_circuits[["circuitId", "lat", "long", "country"]],
        on="circuitId", how="left", suffixes=("_kaggle", "_ergast")
    )
    # Prefer Kaggle lat/long if available, else Ergast
    for col in ["lat", "long"]:
        k, e = f"{col}_kaggle", f"{col}_ergast"
        if k in circuit_merged.columns and e in circuit_merged.columns:
            circuit_merged[col] = circuit_merged[k].combine_first(circuit_merged[e])

    # ── Main merge ──
    log("  Merging results → races → drivers → constructors → circuits...")
    master = results.merge(races[["raceId", "year", "round", "circuitId", "name", "date"]], on="raceId", how="left")
    master = master.merge(driver_info, on="driverId", how="left")
    master = master.merge(constructor_info, on="constructorId", how="left")
    master["circuitId"] = master["circuitId"].astype(str)

    master = master.merge(
        circuit_merged[["circuitId", "circuitRef", "country_kaggle", "location", "lat", "long"]].rename(columns={"country_kaggle": "country"}),
        on="circuitId", how="left"
    )
    master = master.merge(driver_standings_pre, on=["raceId", "driverId"], how="left")
    master = master.merge(constructor_standings_pre, on=["raceId", "constructorId"], how="left")

    if not pole.empty:
        master = master.merge(pole, on="raceId", how="left")
        master["started_from_pole"] = (master["driverId"] == master["pole_driverId"]).astype(int)

    # ── Final column selection ──
    keep_cols = [
        "raceId", "year", "round", "date", "name",
        "circuitId", "circuitRef", "country", "location", "lat", "long",
        "driverId", "driver_name", "nationality",
        "constructorId", "constructor_name", "constructor_nationality",
        "grid", "positionOrder", "points", "laps", "milliseconds",
        "is_winner",
        "champ_points_pre", "champ_pos_pre", "champ_wins_pre",
        "constructor_points_pre", "constructor_pos_pre",
    ]
    if "started_from_pole" in master.columns:
        keep_cols.append("started_from_pole")

    master = master[[c for c in keep_cols if c in master.columns]]

    log(f"  Master dataset shape: {master.shape}")
    log(f"  Years covered: {master['year'].min()} – {master['year'].max()}")
    log(f"  Races: {master['raceId'].nunique()}")
    log(f"  Drivers: {master['driverId'].nunique()}")
    log(f"  Winners (is_winner=1): {master['is_winner'].sum()}")

    return master


# ─────────────────────────────────────────────
# STEP 6: SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(master, ergast_circuits, ergast_qualifying):
    log("\n=== STEP 6: Saving Processed Files ===")

    master_path = PROCESSED_DIR / "master_dataset.csv"
    master.to_csv(master_path, index=False)
    log(f"  Saved: {master_path}  ({master_path.stat().st_size // 1024} KB)")

    circuits_path = PROCESSED_DIR / "circuits_enriched.csv"
    ergast_circuits.to_csv(circuits_path, index=False)
    log(f"  Saved: {circuits_path}")

    if not ergast_qualifying.empty:
        qual_path = PROCESSED_DIR / "qualifying_recent.csv"
        ergast_qualifying.to_csv(qual_path, index=False)
        log(f"  Saved: {qual_path}")

    log_path = PROCESSED_DIR / "collection_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"  Saved: {log_path}")


# ─────────────────────────────────────────────
# OPTIONAL: WEATHER DATA
# ─────────────────────────────────────────────

def fetch_weather_for_races(master, api_key, max_races=50):
    """
    Fetches historical weather for each race location + date using OpenWeatherMap.
    Free tier: limited to ~1000 calls/day. We fetch only post-1979 races.
    Set api_key in CONFIG section above.
    """
    if not api_key:
        log("\nSkipping weather (no API key set). Add OPENWEATHER_API_KEY to config.")
        return pd.DataFrame()

    log("\n=== OPTIONAL: Fetching Weather Data ===")
    # OpenWeatherMap Historical API: https://openweathermap.org/history
    WEATHER_URL = "https://history.openweathermap.org/data/2.5/history/city"

    races_with_coords = master.dropna(subset=["lat", "long", "date"]).drop_duplicates("raceId")
    races_with_coords = races_with_coords[races_with_coords["year"] >= 1979]  # OWM limit
    races_with_coords = races_with_coords.sort_values("date", ascending=False).head(max_races)

    cache_path = RAW_DIR / "weather_cache.json"
    weather_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            weather_cache = json.load(f)

    weather_rows = []
    for _, row in tqdm(races_with_coords.iterrows(), total=len(races_with_coords), desc="Weather"):
        race_id = str(row["raceId"])
        if race_id in weather_cache:
            weather_rows.append(weather_cache[race_id])
            continue
        try:
            ts = int(pd.Timestamp(row["date"]).timestamp())
            params = {
                "lat":   row["lat"],
                "lon":   row["long"],
                "type":  "hour",
                "start": ts,
                "end":   ts + 3600 * 24,  # race day window
                "appid": api_key,
                "units": "metric",
            }
            resp = requests.get(WEATHER_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("list"):
                w = data["list"][0]["main"]
                wind = data["list"][0].get("wind", {})
                rain = data["list"][0].get("rain", {})
                rec = {
                    "raceId":       int(race_id),
                    "temp_c":       w.get("temp"),
                    "humidity_pct": w.get("humidity"),
                    "wind_ms":      wind.get("speed"),
                    "rain_mm":      rain.get("1h", 0.0),
                }
                weather_rows.append(rec)
                weather_cache[race_id] = rec
            time.sleep(1.2)  # stay under rate limit
        except Exception as e:
            log(f"  Weather fetch failed for raceId {race_id}: {e}")

    with open(cache_path, "w") as f:
        json.dump(weather_cache, f)

    df = pd.DataFrame(weather_rows)
    if not df.empty:
        path = PROCESSED_DIR / "weather.csv"
        df.to_csv(path, index=False)
        log(f"  Saved weather data: {path}  ({len(df)} races)")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 55)
    log("F1 Data Collection — DSA 210 Project")
    log("=" * 55)

    # 1. Download from Kaggle
    ok = download_kaggle_dataset()
    if not ok:
        print("\nFix Kaggle setup first, then re-run. Exiting.")
        exit(1)

    # 2. Load CSVs
    dfs = load_kaggle_data()
    if "results" not in dfs or "races" not in dfs:
        print("Core files missing. Check download. Exiting.")
        exit(1)

    # 3. Ergast circuits
    ergast_circuits = fetch_ergast_circuits()

    # 4. Ergast qualifying (recent years to fill gaps)
    ergast_qualifying = fetch_ergast_qualifying(years=range(2021, 2025))

    # 5. Build master dataset
    master = build_master_dataset(dfs, ergast_circuits)

    # 6. Save
    save_outputs(master, ergast_circuits, ergast_qualifying)

    # 7. Weather (optional)
    fetch_weather_for_races(master, OPENWEATHER_API_KEY)

    log("\n✅ Data collection complete!")
    log(f"   Main file: data/processed/master_dataset.csv")
    log(f"   Shape: {master.shape}")
    log("\nNext step: open master_dataset.csv for EDA.")