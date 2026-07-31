/**
 * Shared Groq client for the Pages Functions.
 *
 * Groq serves an OpenAI-compatible chat-completions API, so the wire shape here
 * is OpenAI's, not Anthropic's: a Bearer token, the system prompt as the first
 * element of `messages`, and the reply at choices[0].message.content.
 *
 * The model id is read from the GROQ_MODEL binding. Groq retires hosted models
 * on short notice, so keeping it in config means a deprecation is a dashboard
 * change rather than a redeploy. List what your key can currently reach with:
 *   curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
 */

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_MODEL = "llama-3.3-70b-versatile";

export function groqModel(env) {
  return env.GROQ_MODEL || DEFAULT_MODEL;
}

/**
 * Call Groq and return the assistant's text.
 *
 * Throws an Error carrying `status` so callers can distinguish a rate limit from
 * a bad request without parsing message strings.
 */
export async function groqChat(env, { system, messages, maxTokens = 1200,
                                      temperature = 0.2, jsonMode = false }) {
  const payload = {
    model: groqModel(env),
    max_tokens: maxTokens,
    // Low but non-zero: these are advisory answers over fixed data, where
    // stability matters more than variety.
    temperature,
    messages: [{ role: "system", content: system }, ...messages],
  };
  if (jsonMode) payload.response_format = { type: "json_object" };

  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${env.GROQ_API_KEY}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const detail = (await res.text()).slice(0, 300);
    const err = new Error(
      res.status === 429
        ? "Rate limited by Groq. Wait a moment and try again."
        : `Groq request failed (${res.status}). ${detail}`
    );
    err.status = res.status;
    throw err;
  }

  const data = await res.json();
  const choice = (data.choices || [])[0] || {};
  const text = (choice.message || {}).content || "";

  if (choice.finish_reason === "length") {
    // Surfaced rather than swallowed: a truncated bulletin is not parseable JSON,
    // and a truncated answer is worse than an error in an advisory context.
    const err = new Error("Response hit the token limit before finishing.");
    err.status = 502;
    throw err;
  }

  return { text: text.trim(), model: data.model };
}

export function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
