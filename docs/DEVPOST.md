# Devpost — everything to paste

Live: https://naturecipher-drought.pages.dev
Repo: https://github.com/naturecipherAI/naturecipher-igad-hackathon

---

# 1. Project Story

*(paste into "About the project")*

## Inspiration

You cannot photograph October in July.

Every drought early-warning system in the region runs on satellite pictures —
rainfall from CHIRPS, vegetation greenness from MODIS. The pictures are excellent.
They also only exist for months that have already happened.

So the tools we have are very good at telling a county what *is* happening, and
structurally unable to tell it what *will*. And by the time a failed season shows
up from orbit, the livestock are already thin, the boreholes are already queuing,
and the money that could have moved fodder is three months late.

We kept coming back to the same question: what if the satellite data for a future
month could be produced before the month arrives?

## What it does

Cascade Bridge forecasts the probability of drought one to three months ahead for
eleven counties in Kenya's arid and semi-arid lands, and it uses **no satellite
data at the moment it runs**.

A county officer opens it and can:

- see which counties are flagged, and by how much
- move the decision threshold to their own risk tolerance and watch the map answer
- ask a plain question — *"which counties should pre-position water trucking?"*
- generate the bulletin they would actually circulate, as a PDF

## How we built it

**The product is four machine-learning models.** The dashboard is a window onto
them.

We trained XGBoost regressors on 35 years of paired history — atmosphere on one
side, satellite observations on the other — until they learned what the
satellites *would* see given only the weather. Then we chained them in the order
the physics runs:

```
Seasonal weather forecast (ECMWF SEAS5, atmosphere only)
   |
   |-- Model 1 --> rainfall           learned from CHIRPS
   |-- Model 2 --> vegetation         learned from MODIS NDVI
   |-- Model 3 --> land temperature   learned from MODIS LST
   |
   '-- 51 features --> Model 4: drought classifier --> probability
```

Rain drives greenness; greenness moderates ground heat. Each model consumes the
one before it. That chaining is the bridge.

The satellite record is not discarded. **It moves** — out of inference time, where
it cannot exist, into training time, where there are decades of it.

Around the models: a Python pipeline pulling ECMWF SEAS5 and deriving the drought
indices, a static dashboard with a MapLibre choropleth over real county
boundaries, and two small serverless functions that put a language model on top —
one for grounded questions, one that drafts the county bulletin.

## Challenges we ran into

**We found four bugs in our own physics, and fixing them changed every number.**

The bias correction was multiplying each forecast by its own anomaly ratio, which
squares the departure instead of removing it — a forecast at 1.3× normal rainfall
came out at 1.69×. Potential evapotranspiration had been wired to a runoff
variable; in the ASALs runoff is close to zero, so the water-balance index was
computing rainfall minus roughly nothing. SPI at 3, 6 and 12 months were three
copies of the same single-month calculation, so a year-long deficit was invisible.
And three vegetation-health features were being stripped from the input and
silently refilled with zeros.

Every one of those was quiet. Nothing crashed. The dashboard looked fine. We only
found them by reading the pipeline against the methods document line by line —
which is the argument for writing the methods document first.

**The harder challenge was intellectual honesty.** Our headline accuracy came from
a window that spans the worst Horn of Africa drought in forty years, when drought
is likely the *majority* class. A model that always answered "drought" could beat
us. So we built the tool that could disprove our own contribution: a baseline
script that publishes the drought base rate, a majority-class floor, and — the
uncomfortable one — the same classifier with the bridges removed.

Because the bridges are derived from the atmospheric inputs, they cannot invent
information those inputs did not already carry. If the cascade does not beat
ERA5-only, our contribution is *forecastability*, not accuracy. We publish that
number either way.

## Accomplishments we're proud of

- Four chained models that generate satellite-derived indicators for months that
  have not happened
- Shipping the baseline that could sink the whole idea, rather than the metric
  that flatters it
- A threshold slider that makes the policy judgement visible: 0.25 encodes the
  claim that missing a drought costs about three times a false alarm, and a
  county can move it
- A language-model layer that **refuses to estimate**. Ask it something the data
  does not contain and it says so. In an early-warning tool, a confident
  hallucination is a casualty
- Numbers in the generated PDF are rendered from the forecast file, not written
  by the model — so a mis-stated figure cannot reach the table

## What we learned

An early-warning system that overstates its confidence is worse than none. That
sounds like a slogan; it turned into engineering. It is why the base rate sits
next to the accuracy, why the interval is on the page, why a signal clearing the
line by 0.012 is labelled as too thin to resolve, and why the assistant is built
to decline.

We also learned that "satellite-independent" — how we first described this — was
the wrong claim and a weaker one. Latency is not the binding problem at a
three-month lead. **The binding problem is that the future has not been
observed.** Naming it correctly made the architecture obvious.

## What's next

- Carry the ensemble spread. SEAS5 gives 51 members and we average them, so every
  probability is currently a point estimate with no error bar
- A full multi-year hindcast across all three regions, replacing a single
  initialization
- Forecast verification: keep every issued forecast and score it against what
  actually happened, so skill is measured rather than asserted
- Extend the county set, and test the method in a second IGAD member state — the
  bridges are trained per region on public data, and nothing about them is
  Kenya-specific

We are not trying to replace ICPAC's Drought Watch or HUSIKA. Those monitor the
present, and they do it well. This adds the layer in front.

---

# 2. Built with

*(≤ 25 tags — paste as comma-separated)*

```
python, xgboost, scikit-learn, pandas, numpy, scipy, xarray, cfgrib, geopandas,
ecmwf-seas5, era5, chirps, modis, aws-s3, aws-ec2, cloudflare-pages,
cloudflare-workers, groq, llama, maplibre-gl, javascript, geojson, parquet,
machine-learning, geospatial
```

---

# 3. Try it out links

```
https://naturecipher-drought.pages.dev
https://github.com/naturecipherAI/naturecipher-igad-hackathon
```

---

# 4. Voiceover script — ElevenLabs, under 2 minutes

Conversational, direct, low jargon. ~250 words, about 1 min 50 at a natural pace.
Slashes mark breaths, not pauses to read aloud.

> You can't photograph October in July.
>
> Drought warning systems run on satellite pictures. Rainfall, vegetation,
> ground temperature. They're good pictures — but they only exist for months
> that already happened.
>
> So today's tools can tell a county what *is* happening. They can't tell it
> what's coming. And by the time a failed season shows up from space, the
> livestock are thin and the money that could have moved fodder is three months
> late.
>
> This is Cascade Bridge.
>
> We trained four models on thirty-five years of history, until they learned
> what the satellites *would* see given only the weather. Feed them a seasonal
> forecast, and they produce the rainfall layer. That feeds the vegetation
> model. That feeds the ground temperature model. Those feed the drought
> classifier.
>
> Rain drives greenness. Greenness moderates heat. Each model hands off to the
> next — that's the cascade.
>
> The satellite record isn't thrown away. It moves. Out of the moment we run,
> where it can't exist, into training, where there's decades of it.
>
> Here's October, forecast in July. Turkana, Marsabit, Samburu, Isiolo — flagged.
>
> Now watch. That's the decision threshold. A county with fodder ready can act
> earlier. A treasury moving real money can wait. Same forecast. Their call.
>
> Ask it anything — it answers only from this data, and refuses when the number
> isn't there.
>
> And this is the bulletin a drought committee would actually circulate. One
> click, ready to send.
>
> We also publish the baseline that could prove us wrong. Because a warning
> system that oversells itself is worse than none.
>
> Cascade Bridge. By Nature Cipher.

---

# 5. Ten slides

**1 — Title**
Cascade Bridge · Drought forecasts for months no satellite has seen yet
by Nature Cipher · IGAD Hackathon 2026
*Visual: logo mark on dark, live URL small at the base.*

**2 — The problem**
"You cannot photograph October in July."
Drought warning runs on satellite pictures. Pictures only exist for the past.
*Visual: a satellite image with the right-hand third blanked out and labelled "not yet observed".*

**3 — What that costs**
By the time a failed season is visible from orbit: livestock thin, water points
dry, response money three months late.
Eleven ASAL counties. Some of the most drought-exposed communities in the region.
*Visual: Kenya map, the eleven counties picked out.*

**4 — The idea**
Produce the satellite data *before* the satellite can.
Train models on 35 years of paired history — atmosphere in, satellite out — then
run them on a weather forecast.
*Visual: one arrow, atmosphere → satellite layers, labelled "learned".*

**5 — The models (this is the product)**
Weather forecast → rainfall → vegetation → land temperature → drought probability.
Rain drives greenness. Greenness moderates heat. Each model feeds the next.
*Visual: the cascade strip from the live dashboard.*

**6 — The key move**
The satellite record isn't discarded. It moves: from inference time, where it
can't exist, to training time, where there's decades of it.
*Visual: two boxes, an arrow moving "satellite data" from the right box to the left.*

**7 — Live**
Scrub the month. Drag the threshold. Hover a county.
0.25 encodes a judgement: missing a drought costs ~3× a false alarm. A county can
move it.
*Visual: GIF — month scrub, then threshold drag.*

**8 — From forecast to action**
Ask a plain question, grounded strictly in this issue's data — it refuses rather
than guesses.
Generate the bulletin a drought committee circulates. PDF, one click.
Numbers come from the data file, not the language model.
*Visual: GIF — chat answer, then bulletin → PDF.*

**9 — We publish what could disprove us**
Base rate · majority-class baseline · the same classifier with the bridges removed.
If the cascade doesn't beat ERA5-only, the contribution is forecastability, not
accuracy. Either way it's on the page.
*Visual: the validation panel, baselines table visible.*

**10 — Where this goes**
Ensemble spread · multi-year hindcast · forecast verification · a second IGAD
member state.
Beside ICPAC Drought Watch and HUSIKA, not instead of them.
*Visual: logo, live URL, repo URL, contact.*

---

# 6. Image gallery — what to capture

3:2 ratio, PNG. In this order:

1. Dashboard hero — masthead with logo, callout, cascade strip visible
2. Map with October selected, ASAL North flagged, a county popup open
3. Threshold slider mid-drag, count changed
4. Chat panel with a real grounded answer
5. Generated bulletin, print preview
6. Validation panel showing base rate and baselines
7. Architecture diagram from `docs/ARCHITECTURE.md`

---

# 7. GIFs — exact recipe

Three GIFs, each under 8 seconds and under 5 MB. Record at 1280×800, browser
zoom 100%, dark theme (the accent reads stronger).

**GIF 1 — "the forecast moves"** *(slide 7)*
Start on Sep. Click Oct → ASAL North turns rust. Click Nov → back to green.
Pause one beat on Oct. ~5s.

**GIF 2 — "the decision is yours"** *(slide 7)*
Threshold at 0.25, one signal flagged. Drag slowly to 0.35 — the callout flips to
"No drought signal". Drag back. Let the counter be readable at both ends. ~6s.

**GIF 3 — "forecast to action"** *(slide 8)*
Click a suggested question. Answer appears. Cut to clicking Generate bulletin,
bulletin renders. ~8s.

Capture with ScreenToGif (Windows) or `ffmpeg`:

```bash
ffmpeg -i capture.mp4 -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];\
[a]palettegen[p];[b][p]paletteuse" -loop 0 out.gif
```

12 fps and 960 px wide keeps a 6-second clip near 3 MB.
