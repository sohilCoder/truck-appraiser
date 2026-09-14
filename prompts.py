"""
Prompts for the v2 pipeline.

Call 1  gate         — is this a usable commercial-truck image?
Call 2  extraction   — extract structured visual features for the GP feature vector
Call 3  evidence_voi — given current uncertainty, what photo would help most?

The LLM never outputs a price. It outputs observations.
Pricing is done entirely by the GP in gp_pricing.py.
"""

# -------------------------------------------------------------------------
# Call 1 — gate
# -------------------------------------------------------------------------

GATE_PROMPT = """\
You are the intake gate for a commercial truck valuation system.
Examine this photo and report what you can actually see.

Reply with ONLY this JSON, nothing else:

{
  "is_vehicle": true,
  "is_commercial_truck": true,
  "truck_class": "pickup",
  "subject": "brief description of what is in the frame",
  "usable": true,
  "quality_issues": [],
  "view": "front",
  "rejection_reason": null,
  "guidance": null
}

truck_class — pick one of: pickup, semi, day_cab, box_truck, flatbed,
  dump_truck, tanker, tow_truck, utility, reefer, other_commercial.
view — pick one of: front, rear, driver_side, passenger_side,
  front_three_quarter, rear_three_quarter, tire, interior, dashboard,
  odometer, sleeper, fifth_wheel, cargo_area, engine_bay, undercarriage,
  detail, unknown.
quality_issues — any of: blurry, too_dark, overexposed, too_far,
  obstructed, not_a_truck, wrong_subject.
Set usable=false if not a truck or too poor to assess condition.
No text outside the JSON.\
"""

# -------------------------------------------------------------------------
# Call 2 — visual feature extraction
# -------------------------------------------------------------------------

# Base + angle-specific suffix combined in extraction_prompt()
_EXTRACTION_BASE = """\
You are a commercial truck condition inspector extracting structured features
for a machine-learning pricing model. Report ONLY what is visible in this photo.
Do NOT guess at things outside the frame.

Reply with ONLY this JSON object, nothing else:

{
  "truck_class": "pickup",
  "make_guess": null,
  "model_guess": null,
  "year_guess": null,
  "year_confidence": 0.0,
  "identity_confidence": 0.0,
  "mileage": null,
  "mileage_units": null,
  "cab_type": null,
  "transmission_guess": null,
  "rust_severity": 0,
  "body_damage_severity": 0,
  "tire_wear_severity": 0,
  "interior_wear_severity": 0,
  "frame_concern": false,
  "mechanical_warning": false,
  "overall_condition": "good",
  "findings": [
    {
      "type": "surface_rust",
      "area": "rear wheel arch",
      "severity": 3,
      "confidence": 0.85
    }
  ],
  "view_quality": "good",
  "assessment_notes": "one sentence on what this angle does and does not show"
}

Field rules:
- truck_class: pickup | semi | day_cab | box_truck | flatbed | dump_truck |
    tanker | tow_truck | utility | reefer | other_commercial
- make_guess / model_guess: exact badge text if readable, else null
- year_guess: integer or null; year_confidence 0-1
- identity_confidence: how confident you are in make+model+year together (0-1)
- mileage: integer if odometer visible, else null
- mileage_units: "miles" | "km" | null
- cab_type: sleeper | day_cab | crew_cab | extended_cab | regular_cab |
    cab_over | null
- All severity fields 0 (none visible) to 5 (severe/safety concern)
- overall_condition: excellent | good | fair | poor
- findings: list of specific observed defects (can be empty)
- view_quality: good | partial | poor
- No dollar values anywhere in this response.\
"""

_VIEW_FOCUS = {
    "tire": "TIRE FOCUS: assess tread depth across full width, sidewall cracking or "
            "bulges, visible wear bars, dry rot, recap indicators, and rim damage. "
            "Set tire_wear_severity carefully — this is the highest-value field for "
            "commercial trucks.",
    "interior": "INTERIOR FOCUS: judge seat bolster and upholstery wear, dash cracks, "
                "headliner condition, pedal wear, steering wheel wear. Pedal wear is a "
                "useful cross-check against claimed mileage.",
    "dashboard": "DASHBOARD FOCUS: read any warning lights, read the odometer/hour meter "
                 "if visible (fill mileage field), judge dash cracking and gauge wear.",
    "odometer": "ODOMETER FOCUS: read the exact mileage or hours from the display — this "
                "is the primary goal. Report in mileage field. Note any warning lights.",
    "front": "FRONT FOCUS: judge grille, bumper, headlights, hood alignment, panel gaps, "
             "frontal damage evidence. Note any grille badges for make/model identification.",
    "rear": "REAR FOCUS: judge rear bumper, tail lights, exhaust, lower panel rust, "
            "tailgate or cargo door condition.",
    "driver_side": "DRIVER-SIDE FOCUS: full length of the truck — doors, rocker panels, "
                   "fuel tank and straps, wheel arches, lower rust. Note overall stance.",
    "passenger_side": "PASSENGER-SIDE FOCUS: doors, rocker panels, battery box or fuel "
                      "tank, wheel arches, lower rust.",
    "front_three_quarter": "FRONT THREE-QUARTER FOCUS: overall stance, panel alignment, "
                            "paint consistency, aero kit condition, frontal damage.",
    "rear_three_quarter": "REAR THREE-QUARTER FOCUS: rear body, frame rails if visible, "
                           "exhaust, rust on rear panels.",
    "engine_bay": "ENGINE BAY FOCUS: visible leaks (oil, coolant, air), corrosion, belt "
                  "and hose condition. Set mechanical_warning=true if leaks are visible.",
    "undercarriage": "UNDERCARRIAGE FOCUS: distinguish surface rust from structural rust "
                     "or cracks on frame rails and cross-members. Set frame_concern=true "
                     "if structural rust or visible cracks are found.",
    "fifth_wheel": "FIFTH WHEEL FOCUS: plate wear, locking jaw condition, frame mounting "
                   "integrity. Note any cracks or deformation around the coupling area.",
    "sleeper": "SLEEPER FOCUS: bunk condition, storage compartment wear, HVAC unit, "
               "curtains, signs of long-term heavy use.",
    "cargo_area": "CARGO AREA FOCUS: floor condition and rust, wall integrity, door seals "
                  "and hinges, tie-down points, signs of corrosive or heavy cargo.",
}

_DEFAULT_FOCUS = ("Judge whatever is clearly visible. Note in assessment_notes "
                  "what this angle does not allow you to assess.")


def extraction_prompt(view):
    focus = _VIEW_FOCUS.get(view, _DEFAULT_FOCUS)
    return _EXTRACTION_BASE + "\n\n" + focus + "\n\nNo text outside the JSON."


# -------------------------------------------------------------------------
# Call 3 — evidence value-of-information ranking
# -------------------------------------------------------------------------

def voi_prompt(current_features, uncertainty_pct, missing_areas, gp_summary):
    """
    Given what we know and the GP's current uncertainty, ask the LLM which
    single photo would most reduce pricing uncertainty.
    """
    feat_str = "\n".join(f"  {k}: {v}" for k, v in current_features.items() if v is not None)
    missing_str = ", ".join(missing_areas) if missing_areas else "none"

    return f"""\
You are advising a truck buyer on which single photo to request next.
The pricing model has estimated a range but uncertainty is {uncertainty_pct:.0f}%.

Current known features:
{feat_str}

Inspection areas not yet photographed: {missing_str}

Pricing model notes: {gp_summary}

Which ONE photo would most reduce pricing uncertainty?
Consider: what unknown features most affect price for this truck class?

Reply with ONLY this JSON:

{{
  "recommended_view": "tire",
  "instruction": "Take a close-up of one rear drive-axle tire, square to the tread.",
  "reason": "Tire condition is unknown and significantly affects value for this class.",
  "potential_impact": "Could shift estimate by several thousand dollars in either direction."
}}

recommended_view must be one of: front, rear, driver_side, passenger_side,
front_three_quarter, rear_three_quarter, tire, interior, dashboard, odometer,
sleeper, fifth_wheel, cargo_area, engine_bay, undercarriage.
No text outside the JSON.\
"""
