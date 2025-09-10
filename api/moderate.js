// Open-source NLP moderation for Vercel (Node.js)
// Combines TFJS toxicity model + lexicon/profanity with leetspeak normalization
import * as tf from "@tensorflow/tfjs"; // pure JS backend (no native deps)
import * as toxicity from "@tensorflow-models/toxicity";
import leo from "leo-profanity";

// ---- Config (from ENV) ----
const THRESH = Number(process.env.TOXICITY_THRESHOLD || "0.85");
const LABELS =
  (process.env.TOXICITY_LABELS || "identity_attack,insult,obscene,sexual_explicit,threat,severe_toxicity")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
const CORS_ALLOW_ORIGIN = process.env.CORS_ALLOW_ORIGIN || "*";
const CUSTOM_WORDS = (process.env.FOUL_CUSTOM_WORDS || "")
  .split(",")
  .map(s => s.trim())
  .filter(Boolean);

// Cache model between invocations (serverless cold-start safe)
let modelPromise;
async function getModel() {
  if (!modelPromise) {
    // NOTE: We can also vendor model assets later if desired.
    modelPromise = toxicity.load(THRESH, LABELS);
  }
  return modelPromise;
}

// Leetspeak/obfuscation normalizer
function normalize(text) {
  return text
    .toLowerCase()
    .replace(/[!¡1|i]/g, "i")
    .replace(/[@4]/g, "a")
    .replace(/[$5]/g, "s")
    .replace(/0/g, "o")
    .replace(/3/g, "e")
    .replace(/7/g, "t")
    .replace(/[^a-z0-9\s@]/g, " ") // keep @mentions
    .replace(/\s{2,}/g, " ")
    .trim();
}

// Prepare leo-profanity
leo.clearList();
leo.loadDictionary();         // english base
if (CUSTOM_WORDS.length) {
  leo.add(CUSTOM_WORDS);
}

// CORS helper
function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(204).end();

  if (req.method !== "POST") {
    return res.status(405).json({ error: "method_not_allowed" });
  }

  try {
    const { text = "", fromUser = "", botName = "" } = req.body || {};
    if (!text || typeof text !== "string") {
      return res.status(400).json({ error: "text_required" });
    }

    // Anti-loop: skip if message appears to come from the bot
    if (botName && fromUser && `${fromUser}`.toLowerCase().includes(`${botName}`.toLowerCase())) {
      return res.json({ foul: false, skipped: true, reason: "self_message" });
    }

    const norm = normalize(text);

    // 1) Lexicon/profanity pass
    const lexiconProfane = leo.check(norm);

    // 2) NLP pass (TFJS toxicity)
    const model = await getModel();
    const preds = await model.classify([norm]); // returns array of {label, results:[{probabilities, match}]}
    // Build score map
    const scores = {};
    let toxicHit = false;
    for (const p of preds) {
      const score = p.results?.[0]?.probabilities?.[1] ?? 0; // prob of "toxic" class
      scores[p.label] = Number(score.toFixed(4));
      if (LABELS.includes(p.label) && score >= THRESH) toxicHit = true;
    }

    const foul = Boolean(lexiconProfane || toxicHit);

    return res.json({
      foul,
      input: text,
      norm,
      reasons: {
        lexiconProfanity: lexiconProfane,
        toxicityScores: scores,
        threshold: THRESH,
        labels: LABELS
      }
    });
  } catch (err) {
    console.error("moderation_error:", err?.message || err);
    return res.status(500).json({ error: "moderation_failed" });
  }
}
