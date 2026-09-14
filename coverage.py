"""
Semantic inspection coverage for v2.
Identical logic to v1 but imported cleanly here so v2 is self-contained.
"""

CATEGORIES = [
    {"key": "identity",      "label": "Vehicle identity",    "weight": 1.00, "price_leverage": 1.00,
     "ask": "Take a clear shot showing badges so year, make and model can be confirmed."},
    {"key": "front",         "label": "Front",               "weight": 0.75, "price_leverage": 0.55,
     "ask": "Straight-on photo of the front from about fifteen feet."},
    {"key": "rear",          "label": "Rear",                "weight": 0.70, "price_leverage": 0.55,
     "ask": "Straight-on photo of the rear including bumper and exhaust."},
    {"key": "driver_side",   "label": "Driver side",         "weight": 0.75, "price_leverage": 0.70,
     "ask": "Full-length side photo from the driver's side."},
    {"key": "passenger_side","label": "Passenger side",      "weight": 0.65, "price_leverage": 0.65,
     "ask": "Full-length side photo from the passenger's side."},
    {"key": "tires",         "label": "Tires",               "weight": 0.90, "price_leverage": 0.90,
     "ask": "Close-up of a rear drive-axle tire, square to the tread."},
    {"key": "interior",      "label": "Cab interior",        "weight": 0.85, "price_leverage": 0.70,
     "ask": "Driver's door open — seat, dash and floor in one frame."},
    {"key": "odometer",      "label": "Odometer / hours",    "weight": 0.85, "price_leverage": 0.95,
     "ask": "Instrument cluster lit up, close enough to read the mileage."},
    {"key": "engine_bay",    "label": "Engine bay",          "weight": 0.60, "price_leverage": 0.75,
     "ask": "Hood open or cab tilted — photograph the engine bay."},
    {"key": "undercarriage", "label": "Frame and underside", "weight": 0.70, "price_leverage": 0.85,
     "ask": "Crouch at the rear and photograph under the truck toward the frame rails."},
    {"key": "bed",           "label": "Truck bed",           "weight": 0.50, "price_leverage": 0.35,
     "classes": ["pickup"],
     "ask": "Stand at the tailgate and photograph the full bed floor."},
    {"key": "fifth_wheel",   "label": "Fifth wheel",         "weight": 0.75, "price_leverage": 0.80,
     "classes": ["semi", "day_cab"],
     "ask": "Photograph the fifth wheel plate and locking jaw from the rear of the cab."},
    {"key": "sleeper",       "label": "Sleeper cab",         "weight": 0.60, "price_leverage": 0.50,
     "classes": ["semi"],
     "ask": "Photograph the inside of the sleeper berth."},
    {"key": "cargo_area",    "label": "Cargo area / box",    "weight": 0.65, "price_leverage": 0.60,
     "classes": ["box_truck", "reefer", "flatbed", "dump_truck", "tanker", "tow_truck", "utility", "other_commercial"],
     "ask": "Open rear doors and photograph the full cargo area."},
]

CATEGORY_INDEX = {c["key"]: c for c in CATEGORIES}

VIEW_TO_CATEGORIES = {
    "front":             {"front": "confirmed", "identity": "partial"},
    "rear":              {"rear": "confirmed", "identity": "partial"},
    "driver_side":       {"driver_side": "confirmed", "tires": "partial"},
    "passenger_side":    {"passenger_side": "confirmed", "tires": "partial"},
    "front_three_quarter": {"front": "confirmed", "driver_side": "partial", "identity": "partial"},
    "rear_three_quarter":  {"rear": "confirmed", "passenger_side": "partial"},
    "tire":              {"tires": "confirmed"},
    "interior":          {"interior": "confirmed", "odometer": "partial"},
    "dashboard":         {"interior": "confirmed", "odometer": "partial"},
    "odometer":          {"odometer": "confirmed"},
    "sleeper":           {"sleeper": "confirmed", "interior": "partial"},
    "fifth_wheel":       {"fifth_wheel": "confirmed"},
    "cargo_area":        {"cargo_area": "confirmed"},
    "bed":               {"bed": "confirmed"},
    "engine_bay":        {"engine_bay": "confirmed"},
    "undercarriage":     {"undercarriage": "confirmed"},
    "detail": {}, "unknown": {},
}

_RANK          = {"missing": 0, "partial": 1, "confirmed": 2}
_QUALITY_FACTOR = {"good": 1.0, "partial": 0.75, "poor": 0.45}


def active_categories(truck_class=None):
    out = []
    for c in CATEGORIES:
        classes = c.get("classes")
        if classes is None or (truck_class and truck_class in classes):
            out.append(c)
    return out


def blank_state(truck_class=None):
    return {c["key"]: {"status": "missing", "evidence": []}
            for c in active_categories(truck_class)}


def apply_photo(state, photo):
    view    = photo.get("view") or "unknown"
    quality = photo.get("view_quality") or "good"
    contribs = dict(VIEW_TO_CATEGORIES.get(view, {}))

    if photo.get("mileage") is not None:
        contribs["odometer"] = "confirmed"
    ident = photo.get("identity_confidence") or 0
    if photo.get("make_guess") and ident >= 0.6:
        contribs["identity"] = "confirmed"

    for key, level in contribs.items():
        if key not in state:
            continue
        if quality == "poor" and level == "confirmed":
            level = "partial"
        entry = state[key]
        if _RANK[level] > _RANK[entry["status"]]:
            entry["status"] = level
        if photo["id"] not in entry["evidence"]:
            entry["evidence"].append(photo["id"])
        entry["quality"] = max(entry.get("quality", 0.0),
                               _QUALITY_FACTOR.get(quality, 0.7))
    return state


def rebuild(photos, typed_specs=None, truck_class=None):
    state = blank_state(truck_class)
    for p in photos:
        if p.get("accepted"):
            apply_photo(state, p)
    typed = typed_specs or {}
    if typed.get("year") and typed.get("make") and typed.get("model"):
        if state.get("identity", {}).get("status") == "missing":
            state["identity"]["status"] = "partial"
    if typed.get("mileage") and state.get("odometer", {}).get("status") == "missing":
        state["odometer"]["status"] = "partial"
    return state


def score(state, truck_class=None):
    cats  = active_categories(truck_class)
    total = sum(c["weight"] for c in cats) or 1.0
    earned = 0.0
    for c in cats:
        entry = state.get(c["key"], {"status": "missing"})
        q = entry.get("quality", 1.0) if entry["status"] != "missing" else 0.0
        if entry["status"] == "confirmed":
            earned += c["weight"] * q
        elif entry["status"] == "partial":
            earned += c["weight"] * 0.5 * max(q, 0.6)
    return round(earned / total, 4)


def missing_areas(state):
    return [k for k, v in state.items() if v["status"] == "missing"]


def next_photo(state, truck_class=None):
    cats = active_categories(truck_class)
    ranked = []
    for c in cats:
        entry = state.get(c["key"], {"status": "missing"})
        if entry["status"] == "confirmed" and entry.get("quality", 1.0) >= 0.75:
            continue
        unc = 1.0 if entry["status"] == "missing" else (0.45 if entry["status"] == "partial" else 0.25)
        priority = c["weight"] * unc * c["price_leverage"]
        ranked.append({"key": c["key"], "label": c["label"],
                        "status": entry["status"], "priority": round(priority, 4),
                        "ask": c["ask"]})
    ranked.sort(key=lambda r: r["priority"], reverse=True)
    return ranked
