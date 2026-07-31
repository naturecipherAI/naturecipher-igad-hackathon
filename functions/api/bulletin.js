/**
 * Draft the narrative for a county drought bulletin.
 *
 * Runs at /api/bulletin. Returns structured sections, not prose blob, so the
 * client renders them into the print layout rather than the model deciding
 * typography. The model writes words; the numbers come from the forecast and
 * are re-rendered client-side from the same JSON, so a hallucinated figure
 * cannot reach the PDF's data table.
 */

const MODEL = "claude-opus-5";
const MAX_TOKENS = 2000;

const MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];

const SCHEMA = {
  type: "object",
  properties: {
    headline: {
      type: "string",
      description: "One sentence, under 20 words, stating the operative finding.",
    },
    situation: {
      type: "string",
      description: "2-3 sentences on what the forecast says across the covered regions.",
    },
    confidence: {
      type: "string",
      description:
        "2-3 sentences on how far to trust this issue: experimental status, single " +
        "initialization, no ensemble spread, and any signal whose margin is under 0.05.",
    },
    recommended_actions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          region: { type: "string" },
          action: { type: "string", description: "One concrete step, under 25 words." },
          urgency: { type: "string", enum: ["monitor", "prepare", "act"] },
        },
        required: ["region", "action", "urgency"],
        additionalProperties: false,
      },
    },
  },
  required: ["headline", "situation", "confidence", "recommended_actions"],
  additionalProperties: false,
};

export async function onRequestPost({ request, env }) {
  if (!env.ANTHROPIC_API_KEY) {
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
      counties: region.counties || [],
      month: `${MONTHS[f.valid_month]} ${f.valid_year}`,
      lead: f.lead_months,
      prob: f.drought_prob,
      margin: Number((f.drought_prob - threshold).toFixed(3)),
      flagged: f.drought_prob >= threshold,
    })));

  const flagged = rows.filter(r => r.flagged);
  const thin = flagged.filter(r => r.margin < 0.05);

  const prompt = `Draft a drought bulletin for county drought committees in Kenya's ASALs.

Issued from the ${forecast.run.init_date} SEAS5 initialization. Status: ${forecast.run.status}.
Decision threshold in use: ${threshold}.

FORECASTS
${rows.map(r => `${r.region} | ${r.month} | lead ${r.lead}mo | P=${r.prob} | margin ${r.margin >= 0 ? "+" : ""}${r.margin} | ${r.flagged ? "FLAGGED" : "below threshold"}`).join("\n")}

COUNTIES
${(forecast.regions || []).map(r => `${r.name}: ${(r.counties || []).join(", ")}`).join("\n")}

${flagged.length === 0
  ? "No region is above the threshold this issue. Say so directly; do not manufacture urgency."
  : `Flagged: ${flagged.map(r => `${r.region} ${r.month}`).join("; ")}.`}
${thin.length ? `Margins under 0.05 (narrower than this pipeline resolves): ${thin.map(r => `${r.region} ${r.month} at +${r.margin}`).join("; ")}. Say this plainly.` : ""}

Write for a county officer deciding whether to pre-position water trucking and
fodder. Use only the numbers above. Name counties, not just region labels.
Give one action per region, sized to what the evidence supports: "monitor" when
below threshold, "prepare" when flagged on a thin margin, "act" only on a clear
margin. Do not overstate. These forecasts are experimental.`;

  const upstream = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      output_config: { effort: "medium", format: { type: "json_schema", schema: SCHEMA } },
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!upstream.ok) {
    const detail = await upstream.text();
    return json({ error: `Model request failed (${upstream.status}).`, detail: detail.slice(0, 300) }, 502);
  }

  const data = await upstream.json();
  if (data.stop_reason === "refusal") {
    return json({ error: "The model declined to draft this bulletin." }, 502);
  }

  const text = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("");
  let narrative;
  try {
    narrative = JSON.parse(text);
  } catch {
    return json({ error: "Bulletin came back unparseable." }, 502);
  }

  // Numbers are returned from the forecast, not from the model, so the table in
  // the PDF cannot drift from the data even if the prose does.
  return json({
    narrative,
    issued: forecast.generated_at,
    init_date: forecast.run.init_date,
    status: forecast.run.status,
    threshold,
    rows,
  });
}

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}
