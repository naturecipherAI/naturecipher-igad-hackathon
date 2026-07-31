/**
 * Grounded Q&A over the current forecast issue.
 *
 * Runs as a Cloudflare Pages Function at /api/chat. The Groq key lives in a
 * Cloudflare secret and never reaches the browser.
 *
 * The model is given the forecast and validation JSON verbatim and told to
 * answer only from them. That constraint is the point: an early-warning
 * assistant that invents a probability is worse than no assistant, so the
 * prompt makes refusing to answer the correct behaviour when the data does not
 * contain it.
 */

import { groqChat, json, fetchJSON } from "./_groq.js";

const MAX_HISTORY = 12;

const MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];

function buildSystemPrompt(forecast, validation) {
  const run = forecast.run || {};
  const cfg = forecast.config || {};

  const lines = (forecast.regions || []).flatMap(region =>
    (region.forecasts || []).map(f =>
      `  ${region.name} | ${MONTHS[f.valid_month]} ${f.valid_year} | ` +
      `lead ${f.lead_months}mo | P(drought)=${f.drought_prob} | ` +
      `margin vs threshold ${f.margin >= 0 ? "+" : ""}${f.margin} | ${f.signal}`));

  const counties = (forecast.regions || [])
    .map(r => `  ${r.name} (${r.climate_zone}, ${r.mean_annual_precipitation_mm}mm/yr): ` +
              `${(r.counties || []).join(", ")}`);

  return `You are the analyst for Cascade Bridge, a drought early-warning forecast for Kenya's Arid and Semi-Arid Lands. You answer questions from county drought committees, ICPAC staff, and anticipatory-action funds.

CURRENT ISSUE
Initialized: ${run.init_date} from ${run.source}
Status: ${run.status}
Decision threshold: ${cfg.threshold} — ${cfg.threshold_rationale}

FORECASTS (this is the complete set; there are no others)
${lines.join("\n")}

COVERAGE
${counties.join("\n")}

VALIDATION
${validation ? JSON.stringify(validation.backtest || {}, null, 1) : "not available"}

HOW TO ANSWER

Ground every number in the data above. If a question needs a figure that is not
there, say so plainly rather than estimating. Never invent a probability, a
county, a date, or a metric.

The threshold is a policy choice, not a property of the forecast. Changing it
changes the decision, never the probability. If someone asks whether to act,
give them the probability and its margin over their threshold, and let them own
the call.

State uncertainty honestly. These forecasts are experimental: they come from a
single SEAS5 initialization, the ensemble spread is discarded so every
probability is a point estimate with no error bar, and soil moisture uses a
persistence assumption. A margin under 0.05 is narrower than this pipeline can
resolve — say so when it applies.

Distinguish a forecast from a hindcast. The January-March 2026 result used ERA5
reanalysis available after the fact; it is not evidence of ahead-of-time skill.

Counties outside the eleven listed have no model coverage. Do not guess for them.

Be brief and concrete. Two or three short paragraphs, no headers, no preamble,
no markdown. Lead with the number the person asked for.`;
}

export async function onRequestPost({ request, env }) {
  if (!env.GROQ_API_KEY) {
    return json({ error: "Chat is not configured on this deployment." }, 503);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Malformed request body." }, 400);
  }

  const history = Array.isArray(body.messages) ? body.messages.slice(-MAX_HISTORY) : [];
  if (!history.length) return json({ error: "No message provided." }, 400);

  const origin = new URL(request.url).origin;
  const [forecast, validation] = await Promise.all([
    fetchJSON(`${origin}/forecast.v2.json`),
    fetchJSON(`${origin}/validation.json`),
  ]);
  if (!forecast) return json({ error: "Forecast data is unavailable." }, 503);

  try {
    const { text, model } = await groqChat(env, {
      system: buildSystemPrompt(forecast, validation),
      messages: history.map(m => ({
        role: m.role === "assistant" ? "assistant" : "user",
        content: String(m.content || "").slice(0, 4000),
      })),
      maxTokens: 900,
    });
    return json({ reply: text || "No answer returned.", model });
  } catch (err) {
    return json({ error: err.message }, err.status === 429 ? 429 : 502);
  }
}
