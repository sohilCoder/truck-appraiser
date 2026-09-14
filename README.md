# TruckVal v2 — Commercial Truck Valuation Appraiser from Photos

Estimate the market value of any commercial truck from photos alone.
All pricing is deterministic and grounded in a Gaussian Process posterior.

## How it works

1. **Vision gate** — First, each photo is checked using a Vision LLM to determine
   whether this is a usable image of a commercial truck. Non-trucks, blurry shots,
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

7. **Active evidence loop** — after each photo, the system identifies the
   single most valuable missing observation and asks for it by name.
   The range narrows as evidence improves.

## Supports all commercial truck classes

Pickups · Semi / 18-wheelers · Box trucks · Flatbeds · Dump trucks ·
Tankers · Tow trucks · Utility trucks · Refrigerated trucks

## Running locally

Requires Python 3.8+, scikit-learn, and numpy.

```bash
pip install scikit-learn numpy
```

Add your API key to `.env`:
