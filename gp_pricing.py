"""
Gaussian Process pricing engine.

Flow:
    1. Load comps.csv at startup → build feature matrix X, price vector y
    2. Fit a GP with a Matern kernel over the feature space
    3. Given extracted visual features for a new truck, build its feature
       vector and call gp.predict() → mean price + std deviation
    4. Find the K nearest neighbours by Euclidean distance in feature space
       and return them as comparable listings
    5. Compute a confidence score from identity certainty, coverage, and
       the GP posterior std relative to the mean

The LLM never sees this file. No prices come from the LLM.
"""

import csv
import math
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# Feature schema
# -------------------------------------------------------------------------

TRUCK_CLASSES = [
    "pickup", "semi", "day_cab", "box_truck", "flatbed",
    "dump_truck", "tanker", "tow_truck", "utility", "reefer", "other_commercial",
]

CAB_TYPES = [
    "sleeper", "day_cab", "crew_cab", "extended_cab",
    "regular_cab", "cab_over", "unknown",
]

CONDITIONS = ["excellent", "good", "fair", "poor"]

CURRENT_YEAR = 2025


def _one_hot(value, categories, unknown_label="unknown"):
    value = str(value or unknown_label).strip().lower()
    vec = []
    for cat in categories:
        vec.append(1.0 if value == cat else 0.0)
    return vec


def _norm_year(year):
    try:
        y = int(year)
        # Age in years, normalised so 0 = new, 1 = 25 years old
        return max(0.0, min(1.0, (CURRENT_YEAR - y) / 25.0))
    except (TypeError, ValueError):
        return 0.5          # unknown → middle of range


def _norm_mileage(mileage, truck_class="pickup"):
    try:
        m = float(str(mileage).replace(",", "").strip())
        # Semis and heavy trucks commonly run 1M+ miles
        ceiling = 1_200_000 if truck_class in ("semi", "day_cab") else 400_000
        return max(0.0, min(1.0, m / ceiling))
    except (TypeError, ValueError):
        return 0.5


def _condition_score(condition):
    mapping = {"excellent": 1.0, "good": 0.72, "fair": 0.42, "poor": 0.15}
    return mapping.get(str(condition or "good").strip().lower(), 0.5)


def _severity_norm(value):
    try:
        return max(0.0, min(1.0, float(value or 0) / 5.0))
    except (TypeError, ValueError):
        return 0.0


def build_feature_vector(features):
    """
    Convert a dict of extracted features into a 1-D numpy array.
    Missing values get a neutral default so the GP can still produce an
    estimate; the missing fields widen the uncertainty.

    Feature layout (27 dimensions):
        [0]     normalised age
        [1]     normalised mileage
        [2]     condition score
        [3]     rust_severity / 5
        [4]     body_damage_severity / 5
        [5]     tire_wear_severity / 5
        [6]     interior_wear_severity / 5
        [7]     frame_concern (0/1)
        [8]     mechanical_warning (0/1)
        [9]     transmission (0=manual, 1=auto, 0.5=unknown)
        [10-20] truck_class one-hot (11 classes)
        [21-27] cab_type one-hot (7 types)
    """
    tc    = str(features.get("truck_class") or "pickup").lower()
    year  = features.get("year_guess") or features.get("year")
    mi    = features.get("mileage")
    cond  = features.get("overall_condition") or features.get("condition") or "good"
    trans = str(features.get("transmission_guess") or features.get("transmission") or "").lower()
    cab   = str(features.get("cab_type") or "unknown").lower()

    trans_val = 1.0 if "auto" in trans else (0.0 if "man" in trans else 0.5)

    vec = [
        _norm_year(year),
        _norm_mileage(mi, tc),
        _condition_score(cond),
        _severity_norm(features.get("rust_severity")),
        _severity_norm(features.get("body_damage_severity")),
        _severity_norm(features.get("tire_wear_severity")),
        _severity_norm(features.get("interior_wear_severity")),
        1.0 if features.get("frame_concern") else 0.0,
        1.0 if features.get("mechanical_warning") else 0.0,
        trans_val,
    ]
    vec += _one_hot(tc, TRUCK_CLASSES)
    vec += _one_hot(cab, CAB_TYPES)

    return np.array(vec, dtype=np.float64)


# -------------------------------------------------------------------------
# Comp dataset loading
# -------------------------------------------------------------------------

def _find_comps_csv():
    candidates = [
        Path(__file__).parent / "comps.csv",
        Path.cwd() / "comps.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_comps(csv_path=None):
    path = Path(csv_path) if csv_path else _find_comps_csv()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "comps.csv not found. Place your curated listings file next to server.py "
            "and restart. See README for the required columns."
        )

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            price_raw = str(row.get("price") or "").replace(",", "").replace("$", "").strip()
            try:
                price = float(price_raw)
            except ValueError:
                continue        # skip rows with no usable price
            if price <= 0 or price > 1_000_000:
                continue

            # Build a feature dict from the CSV row
            feat = {
                "truck_class":             row.get("truck_class") or "pickup",
                "year_guess":              row.get("year"),
                "mileage":                 row.get("mileage"),
                "overall_condition":       row.get("condition") or "good",
                "cab_type":                row.get("cab_type"),
                "transmission_guess":      row.get("transmission"),
                "rust_severity":           0,
                "body_damage_severity":    0,
                "tire_wear_severity":      0,
                "interior_wear_severity":  0,
                "frame_concern":           False,
                "mechanical_warning":      False,
            }

            # Translate condition to approximate severity scores
            cond = str(row.get("condition") or "good").lower()
            if cond == "poor":
                feat.update({"rust_severity": 3, "body_damage_severity": 3,
                             "tire_wear_severity": 3, "interior_wear_severity": 3})
            elif cond == "fair":
                feat.update({"rust_severity": 2, "body_damage_severity": 2,
                             "tire_wear_severity": 2, "interior_wear_severity": 2})

            rows.append({
                "features": feat,
                "price":    price,
                "meta": {
                    "year":  row.get("year"),
                    "make":  row.get("make"),
                    "model": row.get("model"),
                    "truck_class": row.get("truck_class"),
                    "mileage": row.get("mileage"),
                    "condition": row.get("condition"),
                    "cab_type":  row.get("cab_type"),
                    "row":   i,
                },
            })

    if len(rows) < 5:
        raise ValueError(
            f"comps.csv has only {len(rows)} usable rows (need at least 5). "
            "Check that the 'price' column is filled in and the file uses comma separators."
        )

    return rows


# -------------------------------------------------------------------------
# GP model
# -------------------------------------------------------------------------

class TruckPricingGP:
    """
    Gaussian Process regressor over the truck feature space.

    The kernel is Matern(nu=2.5) — smoother than RBF, more robust to
    outliers in small datasets — plus a WhiteKernel for observation noise.
    We work in log-price space so that multiplicative price differences
    (e.g. a $10k vs $100k truck) are treated symmetrically.
    """

    def __init__(self):
        self.gp      = None
        self.scaler  = None
        self.comps   = []
        self.X       = None
        self.y_log   = None
        self.fitted  = False
        self._err    = None

    def fit(self, comps):
        self.comps = comps
        if not comps:
            self._err = "No comps loaded."
            return self

        vecs   = [build_feature_vector(c["features"]) for c in comps]
        prices = [c["price"] for c in comps]

        self.X     = np.vstack(vecs)
        self.y_log = np.log(np.array(prices, dtype=np.float64))

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(self.X)

        kernel = (
            Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 2.0))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=6,
            normalize_y=True,
        )
        self.gp.fit(X_scaled, self.y_log)
        self.fitted = True
        return self

    def predict(self, features_dict):
        """
        Returns:
            mean_price      float
            std_price       float  (GP posterior std in dollar space)
            price_low       float
            price_high      float
            log_std         float  (in log space, useful for confidence calc)
        """
        if not self.fitted:
            raise RuntimeError(self._err or "GP not fitted.")

        vec    = build_feature_vector(features_dict).reshape(1, -1)
        X_sc   = self.scaler.transform(vec)
        mu_log, sigma_log = self.gp.predict(X_sc, return_std=True)

        mu_log    = float(mu_log[0])
        sigma_log = float(sigma_log[0])

        mean_price = math.exp(mu_log)
        # 80% credible interval in log-space, then exponentiate
        z          = 1.282        # 80% two-sided
        low_log    = mu_log - z * sigma_log
        high_log   = mu_log + z * sigma_log

        price_low  = math.exp(low_log)
        price_high = math.exp(high_log)
        std_price  = mean_price * sigma_log      # approximate $ std

        return {
            "mean":      _round(mean_price),
            "std":       _round(std_price),
            "price_low":  _round(price_low),
            "price_high": _round(price_high),
            "log_std":   round(sigma_log, 4),
        }

    def nearest_comps(self, features_dict, k=6):
        """
        Return the K nearest comps by Euclidean distance in scaled feature space.
        """
        if not self.fitted:
            return []

        vec  = build_feature_vector(features_dict).reshape(1, -1)
        X_sc = self.scaler.transform(vec)
        dists = np.linalg.norm(self.scaler.transform(self.X) - X_sc, axis=1)
        idx   = np.argsort(dists)[:k]

        result = []
        for i in idx:
            comp = self.comps[i]
            d    = float(dists[i])
            sim  = round(max(0.0, 1.0 - d / (d + 2.0)) * 100, 1)
            result.append({
                **comp["meta"],
                "price":      comp["price"],
                "similarity": sim,
            })
        return result

    def gp_summary(self, features_dict):
        """Short text note passed to the VOI prompt."""
        nc = self.nearest_comps(features_dict, k=3)
        if not nc:
            return "No close comparables found in the dataset."
        prices = [c["price"] for c in nc]
        avg    = sum(prices) / len(prices)
        return (
            f"Nearest {len(nc)} comps average ${avg:,.0f}. "
            f"Spread: ${min(prices):,.0f}–${max(prices):,.0f}."
        )


# -------------------------------------------------------------------------
# Confidence scoring
# -------------------------------------------------------------------------

def confidence_score(gp_result, identity_certainty, coverage_score,
                     n_accepted_photos, n_comps_found):
    """
    Combine GP uncertainty with inspection completeness into a 0–94% score.

    The GP log_std drives the base: low spread = high confidence.
    Identity and coverage penalties apply on top.
    """
    if n_accepted_photos == 0:
        return 0

    # Convert GP log-std to a 0–1 certainty signal.
    # log_std ~0.05 → very certain, ~0.5 → very uncertain
    gp_cert = max(0.0, min(1.0, 1.0 - float(gp_result.get("log_std", 0.3)) / 0.5))

    ident  = max(0.0, min(1.0, float(identity_certainty or 0.3)))
    cov    = max(0.0, min(1.0, float(coverage_score or 0.0)))
    comp_c = min(1.0, n_comps_found / 5.0)   # more comps → more confident

    raw = 0.40 * gp_cert + 0.25 * ident + 0.20 * cov + 0.15 * comp_c

    # Hard cap: very few photos cannot yield high confidence
    photo_ceiling = {0: 0.0, 1: 0.40, 2: 0.55, 3: 0.67, 4: 0.77}.get(
        n_accepted_photos, 0.94
    )
    score = min(raw, photo_ceiling)
    return int(round(max(0.0, min(0.94, score)) * 100))


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def _round(v, step=50):
    return int(round(v / step) * step)
