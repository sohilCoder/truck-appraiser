"""
v2 Pipeline orchestration.

Per photo:
    1. Gate    — usable? truck? angle?
    2. Extract — structured visual features for the GP feature vector

Per valuation (after each new photo):
    3. Merge   — aggregate features across all accepted photos
    4. GP      — predict price + uncertainty from merged feature vector
    5. VOI     — ask LLM which single photo would most reduce uncertainty

Set TRUCKVAL_MOCK=1 for offline demo mode (canned responses, no API calls).
"""

import os
import time

import coverage as cov_module
import gp_pricing as gp_module
import prompts
from llm_client import LLMError, call_json, image_message

MOCK = os.environ.get("TRUCKVAL_MOCK", "").strip() in {"1", "true", "yes"}

# Singleton GP model — loaded once at server startup
_GP = None


def get_gp():
    global _GP
    if _GP is None:
        comps = gp_module.load_comps()
        _GP   = gp_module.TruckPricingGP().fit(comps)
    return _GP


# -------------------------------------------------------------------------
# Call 1 — Gate
# -------------------------------------------------------------------------

def run_gate(data_url):
    if MOCK:
        return _mock_gate()
    result = call_json(
        [image_message(prompts.GATE_PROMPT, data_url)],
        temperature=0.1, max_tokens=600,
    )
    return _clean_gate(result)


def _clean_gate(raw):
    is_truck = bool(raw.get("is_commercial_truck") or raw.get("is_pickup_truck"))
    usable   = bool(raw.get("usable")) and is_truck
    view     = str(raw.get("view") or "unknown").strip().lower()
    tc       = str(raw.get("truck_class") or "pickup").strip().lower()

    return {
        "is_vehicle":        bool(raw.get("is_vehicle")),
        "is_commercial_truck": is_truck,
        "truck_class":       tc,
        "subject":           str(raw.get("subject") or "")[:200],
        "accepted":          usable,
        "view":              view,
        "quality_issues":    [str(q) for q in (raw.get("quality_issues") or [])][:6],
        "rejection_reason":  _s(raw.get("rejection_reason")),
        "guidance":          _s(raw.get("guidance")),
    }


# -------------------------------------------------------------------------
# Call 2 — Feature extraction
# -------------------------------------------------------------------------

def run_extraction(data_url, view):
    if MOCK:
        return _mock_extraction(view)
    result = call_json(
        [image_message(prompts.extraction_prompt(view), data_url)],
        temperature=0.15, max_tokens=2000,
    )
    return _clean_extraction(result)


def _clean_extraction(raw):
    def sev(k): return max(0, min(5, int(float(raw.get(k) or 0))))
    def flt(k, d=0.0): 
        try: return max(0.0, min(1.0, float(raw.get(k) or d)))
        except: return d

    findings = []
    for f in (raw.get("findings") or []):
        if not isinstance(f, dict): continue
        ft = str(f.get("type") or "").strip()
        if not ft: continue
        findings.append({
            "type":        ft,
            "area":        str(f.get("area") or "")[:120],
            "severity":    max(1, min(5, int(float(f.get("severity") or 3)))),
            "confidence":  flt("confidence", 0.7),
        })

    odo = raw.get("mileage")
    try:
        odo = int(float(odo)) if odo is not None else None
        if odo is not None and not (0 < odo < 2_000_000): odo = None
    except: odo = None

    quality = str(raw.get("view_quality") or "good").lower()
    if quality not in ("good", "partial", "poor"): quality = "good"

    return {
        "truck_class":            str(raw.get("truck_class") or "pickup").lower(),
        "make_guess":             _s(raw.get("make_guess")),
        "model_guess":            _s(raw.get("model_guess")),
        "year_guess":             _int(raw.get("year_guess")),
        "year_confidence":        flt("year_confidence"),
        "identity_confidence":    flt("identity_confidence"),
        "mileage":                odo,
        "mileage_units":          _s(raw.get("mileage_units")),
        "cab_type":               _s(raw.get("cab_type")),
        "transmission_guess":     _s(raw.get("transmission_guess")),
        "rust_severity":          sev("rust_severity"),
        "body_damage_severity":   sev("body_damage_severity"),
        "tire_wear_severity":     sev("tire_wear_severity"),
        "interior_wear_severity": sev("interior_wear_severity"),
        "frame_concern":          bool(raw.get("frame_concern")),
        "mechanical_warning":     bool(raw.get("mechanical_warning")),
        "overall_condition":      str(raw.get("overall_condition") or "good").lower(),
        "findings":               findings[:12],
        "view_quality":           quality,
        "assessment_notes":       str(raw.get("assessment_notes") or "")[:400],
    }


# -------------------------------------------------------------------------
# Photo processing (gate + extract)
# -------------------------------------------------------------------------

def process_photo(photo_id, data_url, filename):
    photo = {
        "id": photo_id, "filename": filename,
        "accepted": False, "findings": [],
        "view": "unknown", "view_quality": "good",
        "truck_class": "pickup",
        "identity_confidence": 0.0,
        "error": None,
    }

    try:
        gate = run_gate(data_url)
    except LLMError as exc:
        print(f"[GATE ERROR] {exc}", flush=True)
        photo["error"] = str(exc)
        photo["rejection_reason"] = "The photo could not be analysed. Check the terminal for details."
        photo["guidance"] = "Try again, or switch to offline mock mode (TRUCKVAL_MOCK=1)."
        return photo

    photo.update(gate)
    if not gate["accepted"]:
        if not photo.get("rejection_reason"):
            photo["rejection_reason"] = _default_rejection(gate)
        if not photo.get("guidance"):
            photo["guidance"] = "Upload a clear photo of a commercial truck."
        return photo

    try:
        ext = run_extraction(data_url, gate["view"])
        photo.update(ext)
    except LLMError as exc:
        print(f"[EXTRACTION ERROR] {exc}", flush=True)
        photo["error"] = str(exc)
        photo["assessment_notes"] = "Condition extraction failed — photo counts for coverage but no features extracted."

    return photo


def _default_rejection(gate):
    if not gate.get("is_vehicle"):
        return f"Not a vehicle — looks like {gate.get('subject', 'something else')}."
    if not gate.get("is_commercial_truck"):
        return f"Not a commercial truck — looks like {gate.get('subject', 'another vehicle type')}."
    issues = gate.get("quality_issues") or []
    if issues:
        return "Photo quality issue: " + ", ".join(i.replace("_", " ") for i in issues) + "."
    return "This photo cannot be used to assess the truck."


# -------------------------------------------------------------------------
# Feature merging across photos
# -------------------------------------------------------------------------

def merge_features(photos, typed_specs=None):
    """
    Combine visual features from all accepted photos into one feature dict
    for the GP. Later photos can update earlier findings. Typed specs
    override vision guesses when provided.
    """
    accepted = [p for p in photos if p.get("accepted")]
    if not accepted:
        return {}

    merged = {
        "truck_class":            "pickup",
        "make_guess":             None,
        "model_guess":            None,
        "year_guess":             None,
        "year_confidence":        0.0,
        "identity_confidence":    0.0,
        "mileage":                None,
        "mileage_units":          "miles",
        "cab_type":               None,
        "transmission_guess":     None,
        "rust_severity":          0,
        "body_damage_severity":   0,
        "tire_wear_severity":     0,
        "interior_wear_severity": 0,
        "frame_concern":          False,
        "mechanical_warning":     False,
        "overall_condition":      "good",
    }

    # Aggregate: take the worst severity seen, best identity confidence seen
    for p in accepted:
        if p.get("truck_class") and p["truck_class"] != "pickup":
            merged["truck_class"] = p["truck_class"]

        if p.get("identity_confidence", 0) > merged["identity_confidence"]:
            merged["identity_confidence"] = p["identity_confidence"]
            if p.get("make_guess"):  merged["make_guess"]  = p["make_guess"]
            if p.get("model_guess"): merged["model_guess"] = p["model_guess"]

        if p.get("year_confidence", 0) > merged["year_confidence"] and p.get("year_guess"):
            merged["year_guess"]       = p["year_guess"]
            merged["year_confidence"]  = p["year_confidence"]

        if p.get("mileage") and (merged["mileage"] is None):
            merged["mileage"]       = p["mileage"]
            merged["mileage_units"] = p.get("mileage_units") or "miles"

        if p.get("cab_type") and not merged["cab_type"]:
            merged["cab_type"] = p["cab_type"]

        if p.get("transmission_guess") and not merged["transmission_guess"]:
            merged["transmission_guess"] = p["transmission_guess"]

        for sev_key in ("rust_severity", "body_damage_severity",
                        "tire_wear_severity", "interior_wear_severity"):
            merged[sev_key] = max(merged[sev_key], p.get(sev_key) or 0)

        if p.get("frame_concern"):     merged["frame_concern"]     = True
        if p.get("mechanical_warning"): merged["mechanical_warning"] = True

    # Derive overall_condition from aggregate severities
    max_sev = max(merged["rust_severity"], merged["body_damage_severity"],
                  merged["tire_wear_severity"], merged["interior_wear_severity"])
    if max_sev >= 4:   merged["overall_condition"] = "poor"
    elif max_sev >= 3: merged["overall_condition"] = "fair"
    elif max_sev >= 1: merged["overall_condition"] = "good"
    else:              merged["overall_condition"] = "excellent"

    # Typed specs override vision guesses
    typed = typed_specs or {}
    if typed.get("year"):   merged["year_guess"]  = int(typed["year"]); merged["year_confidence"] = 1.0
    if typed.get("make"):   merged["make_guess"]  = typed["make"];  merged["identity_confidence"] = max(0.75, merged["identity_confidence"])
    if typed.get("model"):  merged["model_guess"] = typed["model"]
    if typed.get("mileage"):
        merged["mileage"]       = int(typed["mileage"])
        merged["mileage_units"] = typed.get("mileage_units") or "miles"
        merged["identity_confidence"] = max(0.6, merged["identity_confidence"])

    return merged


# -------------------------------------------------------------------------
# Call 3 — VOI: what to photograph next
# -------------------------------------------------------------------------

def run_voi(merged_features, uncertainty_pct, missing, gp_summary):
    if MOCK:
        return _mock_voi(missing)
    if not missing:
        return None
    try:
        result = call_json(
            [{"role": "user", "content": prompts.voi_prompt(
                merged_features, uncertainty_pct, missing, gp_summary)}],
            temperature=0.2, max_tokens=400,
        )
        return {
            "view":    str(result.get("recommended_view") or "unknown"),
            "ask":     str(result.get("instruction") or "")[:300],
            "reason":  str(result.get("reason") or "")[:300],
            "impact":  str(result.get("potential_impact") or "")[:200],
        }
    except LLMError as exc:
        print(f"[VOI ERROR] {exc}", flush=True)
        # Fall back to static coverage ranking
        return None


# -------------------------------------------------------------------------
# Session state assembly
# -------------------------------------------------------------------------

def build_session_state(session, force_voi=False):
    photos       = session["photos"]
    accepted     = [p for p in photos if p.get("accepted")]
    typed_specs  = session.get("typed_specs") or {}

    # Truck class: prefer typed, then most recent photo
    truck_class = typed_specs.get("truck_class")
    if not truck_class:
        for p in reversed(photos):
            if p.get("accepted") and p.get("truck_class"):
                truck_class = p["truck_class"]; break
    session["truck_class"] = truck_class

    # Coverage
    cov_state   = cov_module.rebuild(photos, typed_specs, truck_class)
    cov_score   = cov_module.score(cov_state, truck_class)
    missing     = cov_module.missing_areas(cov_state)
    next_static = cov_module.next_photo(cov_state, truck_class)

    session["coverage"] = {
        "score":      round(cov_score * 100),
        "categories": [
            {"key": c["key"], "label": c["label"],
             "status": cov_state[c["key"]]["status"],
             "evidence": cov_state[c["key"]]["evidence"]}
            for c in cov_module.active_categories(truck_class)
            if c["key"] in cov_state
        ],
    }

    # Collect all findings across photos
    all_findings = []
    for p in accepted:
        for f in (p.get("findings") or []):
            all_findings.append({**f, "source_photo": p["id"]})
    session["findings"] = all_findings

    if not accepted:
        session["valuation"]    = None
        session["merged"]       = None
        session["comparables"]  = []
        session["recommendation"] = next_static[0] if next_static else None
        session["next_priorities"] = next_static[:4]
        return session

    # Merge features
    merged = merge_features(photos, typed_specs)
    session["merged"] = merged

    # GP pricing
    try:
        gp       = get_gp()
        gp_res   = gp.predict(merged)
        comps    = gp.nearest_comps(merged, k=6)
        gp_sum   = gp.gp_summary(merged)
        conf     = gp_module.confidence_score(
            gp_res,
            merged.get("identity_confidence", 0.3),
            cov_score,
            len(accepted),
            len(comps),
        )

        session["valuation"] = {
            "price_low":    gp_res["price_low"],
            "price_high":   gp_res["price_high"],
            "price_point":  gp_res["mean"],
            "std":          gp_res["std"],
            "log_std":      gp_res["log_std"],
            "confidence":   conf,
            "coverage_score": round(cov_score * 100),
            "identity_certainty": round(merged.get("identity_confidence", 0) * 100),
            "currency":     "USD",
            "method":       "Gaussian Process regression over curated comp dataset",
            "n_comps_used": len(comps),
        }
        session["comparables"] = comps
        session["error"] = None

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[GP ERROR] {exc}", flush=True)
        session["valuation"]   = None
        session["comparables"] = []
        session["error"]       = str(exc)
        session["recommendation"] = next_static[0] if next_static else None
        session["next_priorities"] = next_static[:4]
        return session

    # VOI recommendation
    uncertainty_pct = 100 - conf
    if force_voi or not session.get("recommendation") or len(accepted) <= 1:
        voi = run_voi(merged, uncertainty_pct, missing, gp_sum)
        if voi:
            session["recommendation"] = voi
            session["next_priorities"] = [voi] + next_static[:3]
        else:
            session["recommendation"]  = next_static[0] if next_static else None
            session["next_priorities"] = next_static[:4]
    else:
        # Re-run VOI every 2nd photo to avoid burning too many API calls
        if len(accepted) % 2 == 0:
            voi = run_voi(merged, uncertainty_pct, missing, gp_sum)
            if voi:
                session["recommendation"] = voi
                session["next_priorities"] = [voi] + next_static[:3]

    return session


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _s(v):
    if v is None: return None
    s = str(v).strip()
    return None if s.lower() in ("null", "none", "unknown", "n/a", "") else s[:80]


def _int(v):
    try:
        n = int(float(v))
        return n if 1900 < n < 2030 else None
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------------------------
# Mock mode
# -------------------------------------------------------------------------

def _mock_gate():
    import random
    if random.random() < 0.15:
        return _clean_gate({"is_vehicle": True, "is_commercial_truck": False,
                            "truck_class": "pickup", "subject": "a motorcycle",
                            "usable": False, "view": "unknown",
                            "rejection_reason": "Not a commercial truck — looks like a motorcycle.",
                            "guidance": "Upload photos of the commercial truck you want valued."})
    views = ["front_three_quarter", "driver_side", "tire", "interior", "rear", "odometer"]
    return _clean_gate({"is_vehicle": True, "is_commercial_truck": True,
                        "truck_class": "semi", "subject": "a semi truck on a lot",
                        "usable": True, "view": random.choice(views),
                        "rejection_reason": None, "guidance": None})


def _mock_extraction(view):
    base = {
        "truck_class": "semi", "make_guess": "Freightliner", "model_guess": "Cascadia",
        "year_guess": 2019, "year_confidence": 0.70, "identity_confidence": 0.65,
        "mileage": 487000 if view == "odometer" else None, "mileage_units": "miles",
        "cab_type": "sleeper", "transmission_guess": "automatic",
        "rust_severity": 2, "body_damage_severity": 1, "tire_wear_severity": 3,
        "interior_wear_severity": 2, "frame_concern": False, "mechanical_warning": False,
        "overall_condition": "fair",
        "findings": [{"type": "tire_wear", "area": "rear drive axle", "severity": 3, "confidence": 0.80}],
        "view_quality": "good",
        "assessment_notes": "Mock extraction — not reading real photos.",
    }
    return _clean_extraction(base)


def _mock_voi(missing):
    ask_map = {
        "tires":    ("tire",     "Close-up of a rear drive-axle tire.",      "Tire condition unknown — could shift estimate significantly."),
        "odometer": ("odometer", "Photograph the instrument cluster.",        "Mileage is unconfirmed and is the strongest pricing signal."),
        "interior": ("interior", "Open the driver's door and photograph the cab.", "Interior wear helps confirm mileage and assess daily-use condition."),
    }
    for key in ("tires", "odometer", "interior"):
        if key in missing:
            v, a, r = ask_map[key]
            return {"view": v, "ask": a, "reason": r, "impact": "Could narrow the range by several thousand dollars."}
    return {"view": "front", "ask": "Straight-on front photo.",
            "reason": "Front view not yet seen.", "impact": "Helps confirm identity and check for frontal damage."}
