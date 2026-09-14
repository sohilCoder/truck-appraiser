# TruckVal v2 — Commercial Truck Valuation Appraiser from Photos

Estimate the market value of any commercial truck from photos alone.
All pricing is deterministic and grounded in a Gaussian Process posterior.

## How it works

1. **Vision gate** — First, each photo is checked using a Vision LLM to determine
   whether it is a usable image of a commercial truck. Non-trucks, blurry shots,
   and bad angles are rejected.

2. **Feature extraction** — The Vision LLM analyzes accepted photos,
   extracts features from the image, and outputs structured condition features
   as a parsed JSON object.

3. **Feature merging** - After all photos are processed, findings are merged
   across photos into a single feature directory. The worst severity feature found
   in any of the pictures is kept. The identity field (make, model, year) is taken
   from the photo with the highest 'identity_confidence'. Typed details by the user
   override vision inferences.

5. **Gaussian Process pricing** — extracted features are assembled into a
   27-dimensional numeric vector. The vector is scaled by a 'StandardScaler'
   fitted on the dataset and passed to a 'GaussianProcessRegressor' with a
   Matérn(ν=2.5) kernel plus a `WhiteKernel` for observation noise. The GP
   works in log-price space, so a $10k truck and a $100k truck are treated symmetrically.
   The GP posterior returns a mean and standard deviation. The price range is
   an 80% confidence interval: 'exp(μ ± 1.282σ)' in log space. The standard
   deviation encodes the certainty of the model (a tight neighbourhood of similar
   comps produces a narrow band, a sparse one produces a wide band). The K nearest
   comps are found by Euclidean distance in the scaled feature space and shown
   to the user with similarity scores, making the pricing explainable to the user.

6. **Confidence scoring** - Confidence is calculated using 0.40 x GP_certainty +
   0.25 x identity_certainty + 0.20 x coverage_score + 0.15_comp_density where
   GP_certainty = max(0,1 - log_std/0.5). Confidence capped at 94%.

7. **Photo recommendations** - After pricing, the system calls the LLM a third time
   with the current feature dictionary, the uncertainty, and a list of missing
   inspection area. The model returns which single photo would reduce the uncertainty
   by the greatest amount and provide the model with the  information that would inform
   its next decision the most in JSON format. The system tracks coverage of 10 inspection
   areas (identity, front, rear, both sides, tires, interior, odometer, engine bay, and
   underside) so that redundant photos don't move the confidence. Coverage adapts to truck
   class.


## Running locally

**Requirements:** Python 3.8+, scikit-learn, numpy. No other dependencies.

```bash
pip install scikit-learn numpy
```

Copy `.env.example` to a new file called `.env` and paste your API key
into the correct line for your provider. The file has instructions for
which line to edit depending on your provider (Gemini, Groq, or OpenAI).

```bash
cp .env.example .env
```



Place your `comps.csv` file in the same folder as `server.py`. A sample
dataset of 46 listings is included to demonstrate the necessary
formatting. Replace it with your own data.

Start the server:

```bash
python3 server.py
```

Open **http://localhost:8001** in any browser. 

**Offline demo mode** — runs the full pipeline with canned responses, no
API calls, no internet required. Useful for demos on unreliable networks:

```bash
# Mac / Linux
TRUCKVAL_MOCK=1 python3 server.py

# Windows
set TRUCKVAL_MOCK=1
python3 server.py
```

The interface displays a banner when mock mode is active.

**To change the port:**

```bash
# Mac / Linux
PORT=8080 python3 server.py

# Windows
set PORT=8080
python3 server.py
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| Vision LLM | Google Gemini 3.6 Flash | Free tier — swappable to Groq or OpenAI by changing one line in `.env` |
| Pricing model | scikit-learn `GaussianProcessRegressor` | Matérn(ν=2.5) kernel, WhiteKernel noise, log-price space |
| Comp dataset | `comps.csv` | Plain CSV, hand-curated — no external pricing API |
| Backend | Python 3.8+ stdlib `http.server` | No framework, no database, no pip installs beyond sklearn/numpy |
| Frontend | Vanilla JS + HTML + CSS | No framework, no build step, no CDN dependencies for logic |
| Session state | In-memory Python dict | No persistence — sessions live for the life of the server process |
| Languages | English / Turkish | Full UI translation toggle built into the frontend |
| Deployment | `python3 server.py` | Runs on any machine with Python 3.8+, including localhost |
