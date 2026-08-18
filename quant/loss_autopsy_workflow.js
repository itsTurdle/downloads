export const meta = {
  name: 'loss-autopsy',
  description: 'Ten independent analysts autopsy disjoint batches of IBS-system losses vs matched winners; synthesis finds replicated loss-signatures',
  phases: [
    { title: 'Autopsy', detail: '10 agents, one disjoint batch each' },
    { title: 'Synthesize', detail: 'story built only from the rules, not the trades' },
  ],
}

const FEATURES = ["ibs","ret1","gap","vol20","rvratio","dist200","dist50","rsi2","downstreak","dow","atrratio","mom20","ddhigh","prioribs"]

const RULE_SCHEMA = {
  type: "object",
  properties: {
    rules: { type: "array", maxItems: 3, items: {
      type: "object",
      properties: {
        feature: { type: "string", enum: FEATURES },
        op: { type: "string", enum: ["<", ">"] },
        threshold: { type: "number" },
        story: { type: "string", description: "one sentence: WHY this condition marks a bad entry" },
        evidence: { type: "string", description: "one sentence: the separation you actually saw in your batch (counts/means)" }
      },
      required: ["feature","op","threshold","story"], additionalProperties: false
    }},
    no_signal: { type: "boolean", description: "true if your batch shows no honest separation" }
  },
  required: ["rules"], additionalProperties: false
}

const GLOSSARY = `System: buy the Dow (DIA) at close when IBS<0.2 (close pinned to the day's low); exit when IBS>0.7 or after 5 days. You get LOSING trades and an equal number of WINNING trades, with features measured AT ENTRY (all legal to filter on):
ibs=entry-day IBS; ret1=entry-day return %; gap=open vs prior close %; vol20=annualized 20d vol %; rvratio=5d vol / 60d vol; dist200/dist50=% above SMA200/SMA50; rsi2=2-period RSI; downstreak=consecutive down closes; dow=weekday 0=Mon..4=Fri; atrratio=entry-day true range / ATR14; mom20=20d return %; ddhigh=% below 252d high; prioribs=previous day's IBS.
OUTCOME fields (pnl, hold, maxadv, win) are for understanding only — rules may use ONLY entry features.`

phase('Autopsy')
const reads = await parallel(args.batches.map((b, i) => () => agent(
`You are one of ten independent quant analysts. Each of you sees a DIFFERENT small batch of trades from the same system; you must judge only from yours.

${GLOSSARY}

YOUR BATCH — ${b.losses.length} losses:
${JSON.stringify(b.losses)}

${b.winners.length} winners:
${JSON.stringify(b.winners)}

Task: find what separates the losses from the winners IN THIS BATCH. Propose at most 3 filter rules, each "feature op threshold" marking a LOSS signature (condition true = skip the trade). Prefer rules with visible separation in your own counts, and say what that separation was. If nothing honestly separates, say so via no_signal=true and return zero or one weak rule rather than inventing patterns. Do not use outcome fields in rules.`,
  { label: `autopsy:${i}`, phase: 'Autopsy', schema: RULE_SCHEMA }
)))

const rules = []
reads.forEach((r, i) => {
  if (r && r.rules) r.rules.forEach(x => rules.push(Object.assign({ analyst: i }, x)))
})
const nosig = reads.filter(r => r && r.no_signal).length
log(`collected ${rules.length} candidate rules from ${reads.filter(Boolean).length} analysts (${nosig} reported no signal)`)

phase('Synthesize')
const STORY_SCHEMA = {
  type: "object",
  properties: {
    story: { type: "string", description: "2-4 sentences: the coherent mechanism behind the replicated loss signatures" },
    top: { type: "array", maxItems: 4, items: {
      type: "object",
      properties: {
        feature: { type: "string", enum: FEATURES },
        op: { type: "string", enum: ["<", ">"] },
        threshold: { type: "number" },
        support: { type: "number", description: "how many independent analysts proposed something equivalent" },
        rationale: { type: "string" }
      },
      required: ["feature","op","threshold","support"], additionalProperties: false
    }}
  },
  required: ["story","top"], additionalProperties: false
}
const synth = await agent(
`Ten analysts independently examined DISJOINT batches of losing trades (vs matched winners) from the same Dow mean-reversion system. You see ONLY their proposed rules, never the trades. Independent replication across analysts is the signal; one-off clever rules are probably batch noise.

${GLOSSARY}

Their rules (analyst = which of the ten proposed it):
${JSON.stringify(rules)}

Cluster equivalent rules (same feature, same direction, similar threshold). Rank the signatures by cross-analyst support. Pick a consensus threshold per signature (median of the cluster). Then tell the story: what market situation do the replicated signatures jointly describe, and why would the snap-back fail there? Be honest about which rules look like noise.`,
  { label: 'synthesize', phase: 'Synthesize', schema: STORY_SCHEMA }
)
return { candidate_rules: rules, synthesis: synth }