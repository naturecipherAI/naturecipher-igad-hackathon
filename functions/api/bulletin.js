/**
 * Draft the narrative for a county drought bulletin.
 *
 * Runs at /api/bulletin. Returns structured sections rather than a prose blob,
 * so the client owns the layout. The model writes the words; every number in the
 * bulletin's table is re-rendered client-side from the forecast JSON, so a
 * mis-stated figure cannot reach the data table.
 */

import { groqChat, json, fetchJSON } from "./_groq.js";

const MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];

const URGENCIES = ["monitor", "prepare", "act"];

const SHAPE = `{
  "headline": "one sentence, under 20 words, stating the operative finding",
  "situation": "2-3 sentences on what the forecast says across the covered regions",
  "confidence": "2-3 sentences on how far to trust this issue",
  "recommended_actions": [
    {"region": "region name", "action": "one concrete step, under 25 words",
     "urgency": "monitor | prepare | act"}
  ]
}`;

/** Keep only what the print layout can render, so a malformed field degrades to absent. */
function sanitise(raw) {
  const str = v => (typeof v === "string" ? v.trim() : "");
  const actions = Array.isArray(raw.recommended_actions) ? raw.recommended_actions : [];
  return {
    headline: str(raw.headline),
    situation: str(raw.situation),
    confidence: str(raw.confidence),
    recommended_actions: actions
      .map(a => ({
        region: str(a && a.region),
        action: str(a && a.action),
        urgency: URGENCIES.includes(a && a.urgency) ? a.urgency : "monitor",
      }))
      .filter(a => a.region && a.action),
  };
}

export async function onRequestPost({ request, env }) {
  if (!env.GROQ_API_KEY) {
    return json({ error: "Bulletin generation is not configured on this deployment." }, 503);
  }

  const origin = new URL(request.url).origin;
  const forecast = await fetchJSON(`${origin}/forecast.v2.json`);
  if (!forecast) return json({ error: "Forecast data is unavailable." }, 503);

  let threshold = (forecast.config || {}).threshold;
  try {
    const body = await request.json();
    if (typeof body.threshold === "number") threshold = body.threshold;
  } catch {
    // No body is fine; fall back to the issued threshold.
  }

  const rows = (forecast.regions || []).flatMap(region =>
    (region.forecasts || []).map(f => ({
      region: region.name,
      month: `${MONTHS[f.valid_month]} ${f.valid_year}`,
      lead: f.lead_months,
      prob: f.drought_prob,
      margin: Number((f.drought_prob - threshold).toFixed(3)),
      flagged: f.drought_prob >= threshold,
    })));

  const flagged = rows.filter(r => r.flagged);
  const thin = flagged.filter(r => r.margin < 0.05);

  const system = `You draft drought bulletins for county drought committees in Kenya's ASALs.

Write for a county officer deciding whether to pre-position water trucking and
fodder. Use only the numbers you are given. Name counties, not just region
labels. Size each action to what the evidence supports: "monitor" when below
threshold, "prepare" when flagged on a thin margin, "act" only on a clear
margin. Do not overstate; these forecasts are experimental.

Reply with JSON only, matching this shape exactly:
${SHAPE}`;

  const prompt = `Issued from the ${forecast.run.init_date} SEAS5 initialization. Status: ${forecast.run.status}.
Decision threshold in use: ${threshold}.

FORECASTS
${rows.map(r => `${r.region} | ${r.month} | lead ${r.lead}mo | P=${r.prob} | margin ${r.margin >= 0 ? "+" : ""}${r.margin} | ${r.flagged ? "FLAGGED" : "below threshold"}`).join("\n")}

COUNTIES
${(forecast.regions || []).map(r => `${r.name}: ${(r.counties || []).join(", ")}`).join("\n")}

${flagged.length === 0
  ? "No region is above the threshold this issue. Say so directly; do not manufacture urgency."
  : `Flagged: ${flagged.map(r => `${r.region} ${r.month}`).join("; ")}.`}
${thin.length ? `Margins under 0.05, narrower than this pipeline resolves: ${thin.map(r => `${r.region} ${r.month} at +${r.margin}`).join("; ")}. Say this plainly in the confidence section.` : ""}

Mention in the confidence section that this is a single initialization, that
ensemble spread is not carried so each probability is a point estimate, and that
soil moisture uses a persistence assumption.

Return the JSON object now.`;

  try {
    const { text, model } = await groqChat(env, {
      system,
      messages: [{ role: "user", content: prompt }],
      maxTokens: 1600,
      temperature: 0.3,
      jsonMode: true,
    });

    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      return json({ error: "Bulletin came back unparseable." }, 502);
    }

    const narrative = sanitise(parsed);
    if (!narrative.headline || !narrative.situation) {
      return json({ error: "Bulletin came back incomplete." }, 502);
    }

    return json({
      narrative,
      issued: forecast.generated_at,
      init_date: forecast.run.init_date,
      status: forecast.run.status,
      threshold,
      rows,
      model,
    });
  } catch (err) {
    return json({ error: err.message }, err.status === 429 ? 429 : 502);
  }
}
