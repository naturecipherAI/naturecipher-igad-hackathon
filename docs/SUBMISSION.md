# Devpost submission copy

Paste-ready. Both sections are inside the 250-word limit.

---

## Project Overview (247 words)

Satellites cannot photograph October in July. That is the constraint every
seasonal drought forecast runs into, and it is not a latency problem — it is that
the observations early-warning systems depend on, CHIRPS rainfall and MODIS
vegetation, only exist for months that have already happened.

Cascade Bridge closes that gap. It trains XGBoost regressors to synthesise those
satellite indicators from atmospheric variables alone, so a drought classifier
built on 51 satellite-derived features can be driven three months ahead by an
ECMWF SEAS5 seasonal forecast. The satellite record is not discarded; it is
distilled into the bridges at training time, and no satellite feed is needed at
forecast time.

The intended users are the people who must act before a failed season becomes a
food-security emergency: ICPAC and national meteorological agencies issuing
regional outlooks, county drought committees in Kenya's ASALs deciding whether to
pre-position water trucking and fodder, and the anticipatory-action funds that
release money on a trigger. This does not replace ICPAC's Drought Watch or
HUSIKA; it adds a forecast layer where those systems currently monitor.

Coverage is eleven counties across three ASAL clusters — Turkana, Marsabit,
Samburu, Isiolo, Wajir, Mandera, Garissa, Kitui, Makueni, Machakos and
Tharaka-Nithi — home to some of the region's most drought-exposed pastoral and
agropastoral communities.

The forecasts are labelled experimental, and the dashboard publishes the base
rate, the baselines and the confidence interval alongside every metric. An early
warning system that overstates its own confidence is worse than none.

---

## Solution Details (249 words)

A drought classifier trained on satellite indicators is unusable for forecasting,
because its inputs do not exist for future months. Cascade Bridge makes them
exist, by chaining four XGBoost regressors:

```
SEAS5 (51 ensemble members, atmosphere only)
  Bridge 1  ERA5 vars              -> CHIRPS rainfall proxy
  Bridge 2  + CHIRPS proxy         -> MODIS NDVI proxy
  Bridge 3  + NDVI proxy           -> LST day / night proxies
        -> 51 features -> XGBoost classifier -> P(drought)
```

Each bridge consumes the previous one's output, mirroring the physical ordering:
rainfall drives greenness, greenness moderates surface temperature. Between
bridges the pipeline derives SPI (gamma-fitted), SPEI (log-logistic), VCI, TCI and
VHI. Bias correction maps the SEAS5 anomaly onto the ERA5 climatology; PET uses
Hargreaves; soil moisture, which SEAS5 does not provide, uses a disclosed
persistence assumption.

The decision threshold is 0.25, not 0.5 — an encoded judgement that a missed
drought costs roughly three times a false alarm. The dashboard exposes it as a
slider, so a county with pre-positioned funds and a treasury releasing a large
disbursement can each read the same probabilities at their own risk tolerance.
Changing the threshold changes the decision, never the forecast.

An LLM layer on Groq turns the issue into an advisory: grounded Q&A over the
forecast, and a generated county bulletin exported to PDF. The model writes the
prose; every number in the bulletin's table is rendered from the forecast JSON,
so a mis-stated figure cannot reach the data.

Stack: Python 3.11, XGBoost 2.0.3 (CPU), xarray/cfgrib, pandas, scipy, boto3, AWS
S3/EC2. Static dashboard with MapLibre GL, Cloudflare Pages plus Pages Functions,
Groq for inference.

We publish the ERA5-only baseline — the same classifier without the bridges. If
the bridges add nothing over their own inputs, that number says so.

---

## Technical Information

**Stack:** Python 3.11 · XGBoost 2.0.3 · xarray + cfgrib · pandas · scipy · boto3 ·
AWS S3/EC2 · MapLibre GL JS · Cloudflare Pages + Pages Functions · Groq (LLM)

**Repository:** https://github.com/naturecipherAI/naturecipher-igad-hackathon

**Live dashboard:** https://naturecipher-drought.pages.dev

**Data:** ECMWF SEAS5 (Copernicus CDS), ERA5 reanalysis, CHIRPS (UCSB/USAID),
MODIS MOD13Q1 + MOD11A2 (NASA LP DAAC), IPC phase classifications. Full citations
in `docs/DATA_ACKNOWLEDGMENTS.md`.

**Boundaries:** geoBoundaries gbOpen KEN ADM1 (source RCMRD, public domain).

---

## Demo video script — 5 minutes

Judges are ICPAC's own early-warning team. They know drought EWS and they know
Drought Watch. Show the method and the honesty, not a feature tour.

**0:00–0:35 — The constraint**
Open on the dashboard, October selected, ASAL North in rust. "This is a drought
probability for October 2026, issued in July. It uses no satellite data, because
the satellite images for October 2026 do not exist yet. That is the problem."

**0:35–1:30 — The cascade**
Architecture diagram. Walk the four bridges in order. Land the point: the
satellite record is not discarded, it moves from inference time to training time.
Name the physical ordering — rainfall drives greenness drives surface temperature.

**1:30–2:30 — Live interaction**
Scrub Sep → Oct → Nov; ASAL North flips to drought. Drag the threshold from 0.25
to 0.35 and let the signal disappear. Say the quiet part: "This issue's only
drought call clears the line by 0.012. We show you that rather than hide it."
Hover a county, show the uncovered counties rendering as uncovered.

**2:30–3:30 — Evidence**
Scroll to validation. Read out the base rate, the majority-class baseline and the
ERA5-only baseline. "Here is what the cascade adds over its own inputs." Then the
confidence interval and the open-gaps boxes.

**3:30–4:30 — Who acts on it**
Threshold as encoded cost ratio. County committee vs national treasury reading the
same forecast at different risk tolerances. Position beside Drought Watch and
HUSIKA, not against them.

**4:30–5:00 — Honest close**
Experimental status, the limitations list, and what is next: full multi-year
hindcast, ensemble spread, per-cell conditions field.
