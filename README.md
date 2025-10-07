# Tri‑Tier LLM Memory: A Practical Architecture with Tier‑Aware Trimming, Budgets, and Episodic Manifests

**Authors:** Nickson Ariemba

**Keywords:** conversational memory, retrieval, summarization, context window, episodic memory, caches

---

## Abstract

We present a practical, production‑oriented memory architecture for large language models (LLMs) that preserves long‑horizon coherence while respecting strict context‑window limits. The design organizes information into three tiers - **New** (N), **Current** (C), and **Old** (O) - and enforces explicit token budgets with **tier‑aware LLM‑driven trimming**, deterministic **promotion/eviction** via a scoring function, and **episodic manifests** that index external long‑term memory. This yields a small, stable **working set** (C) for multi‑turn continuity, a compact stream of **fresh inputs** (N), and a scalable **archive** (O) that rehydrates details only when needed. We provide a formalization, reference TypeScript implementation, and an ablation‑style rationale for each component.

---

## 1. Motivation

Modern LLM applications must balance **continuity** (remembering facts, plans, and constraints across many turns) with **bounded context windows** and **cost/latency constraints**. Naïve strategies - blind truncation, unstructured logs, or constant full‑history retrieval - either lose critical state, exceed budgets, or introduce instability. We aim to:

* **Preserve task continuity** across many turns without ballooning the prompt.
* **Bound context size** using explicit token budgets per tier.
* **Reduce drift** with a stable working set and deterministic selection.
* **Scale memory** via episodic manifests that live outside the prompt but are retrievable on demand.

---

## 2. Overview of the Approach

We maintain three tiers per turn (t):

* **New ((N_t))**: fresh inputs this turn (user message, tool outputs). Trim **aggressively** to remove fluff and duplicates.
* **Current ((C_t))**: the **working set** that persists over a short horizon (the L2 cache). Trim **conservatively**; protect variables, plans, and assumptions.
* **Old ((O_t))**: the archive of **episode manifests** (compact summaries + tags + links to external content). No raw logs; rehydrate on demand into (N_t).

Key mechanisms:

* **Budgets** ((B_N, B_C)) bind token usage for (N_t) and (C_t).
* **Tier‑aware LLM trimming**: trimming prompts differ by tier policy.
* **Scored selection** packs the most useful items into (C_t) under budget.
* **Eviction → manifest**: what falls out of (C_t) is summarized into an episode manifest stored in (O_t).

---

## 3. Formalization (LaTeX)

We denote by (\operatorname{tokens}(X)) the token length of a set or sequence (X). Let (\operatorname{t}^{\tau,\alpha}(X,B)) be an LLM‑based trim operator for tier (\tau\in{\mathrm{NEW},\mathrm{CURRENT},\mathrm{OLD}}) and aggressiveness (\alpha\in{\mathrm{high},\mathrm{medium},\mathrm{low}}) that returns (X') with (\operatorname{tokens}(X')\le B).

```latex
% Budgets and token constraints
\operatorname{tokens}(N_t) \le B_N,\qquad \operatorname{tokens}(C_t) \le B_C.

% Ingest, optional rehydration from manifests
N_t
= \operatorname{t}^{\mathrm{NEW},\mathrm{high}}\!\Big(
   \mathrm{ingest}(\mathrm{user}_t, \mathrm{tools}_t)
   \cup \;\mathrm{rehydrate}(\mathrm{retrieve}(O_{t-1}, q_t)),
   B_N\Big).

% Score-then-pack into the working set
C_t
= \operatorname{t}^{\mathrm{CURRENT},\mathrm{low}}\!\Big(
   \operatorname{topK}_{\mathrm{score}_t}(C_{t-1} \cup N_t),
   B_C\Big).

% Evictions become episode manifests in the archive
O_t
= O_{t-1} \cup \mathrm{manifest}(C_{t-1} \setminus C_t).

% Item score for selection into C_t
\mathrm{score}_t(x)
= w_r\,\mathrm{recency}_t(x)
+ w_f\,\mathrm{frequency}_t(x)
+ w_u\,\mathrm{userPriority}(x)
+ w_d\,\mathrm{dependency}(x)
- w_s\,\mathrm{size}(x).
```

**Interpretation.** We (i) trim new inputs + optional rehydrated summaries into (N_t) under (B_N); (ii) select high‑value items for (C_t) with a budgeted pack under (B_C); (iii) convert evictions (C_{t-1}\setminus C_t) into compact manifests appended to (O_t). (O_t) holds only manifests and links to external storage.

---

## 4. Tier‑Aware LLM Trimming

For each tier, the trimming prompt instructs the model what to preserve or remove:

* **NEW:** aggressively deduplicate, remove phatic text, keep fresh facts, constraints, and references.
* **CURRENT:** preserve nuance, variable bindings, plans, partial results; merge repeats, avoid over‑compression.
* **OLD:** produce an episode **manifest** (summary + salient tags + rehydration hints + external references).

Safety: If the LLM overshoots the target token budget, a final guard compresses to fit, ensuring (\operatorname{tokens}(N_t)\le B_N) and (\operatorname{tokens}(C_t)\le B_C).

---

## 5. TypeScript Reference Implementation (Functional -  No Classes)

Below is a purely functional implementation: no classes, only functions that accept and return immutable state objects. This makes behavior explicit, easy to test, and simple to serialize.

```ts
// tri-tier-memory.functional.ts

/**
 * WHAT: A functional, production-oriented 3-tier memory manager for LLM apps.
 * WHY: Preserve long-horizon continuity while respecting strict token budgets.
 * HOW: Split into New (N_t), Current (C_t), Old (O_t); use tier-aware LLM trimming,
 *      deterministic scored selection into C_t, and manifest-only archive in O_t.
 */

// ---------- Types ----------
export type Tier = "NEW" | "CURRENT" | "OLD";
export type Aggressiveness = "high" | "medium" | "low";

export interface Tokenizer { count(text: string): number }
export interface MemoryItem {
  id: string; content: string; tags: string[];
  userPriority: number; // 0..1
  dependencies: string[];
  createdTs: number; lastTouchedTs: number; lastTouchedTurn: number;
  frequency: number; trimNotes?: string[];
}
export interface EpisodeManifest {
  episodeId: string; topic: string; dateRange: [string,string];
  summary256: string; salienceTags: string[]; links: string[];
  rehydrationHints: string[]; sizeEstimateTokens: number;
  createdTs: number; lastUpdatedTs: number;
}
export interface Budgets { B_total: number; B_N: number; B_C: number; B_sys: number }
export interface ScoringWeights { w_recency: number; w_frequency: number; w_user_priority: number; w_dependency: number; w_size: number }

export interface State {
  turnIdx: number;
  N: MemoryItem[]; C: MemoryItem[]; O: EpisodeManifest[];
  budgets: Budgets; tok: Tokenizer; w: ScoringWeights;
  ttlTurns: number;
  trimFn: (content: string, targetTokens: number, tier: Tier, aggr: Aggressiveness) => Promise<{ text: string, notes: string[] }>
}

// ---------- Defaults ----------
function wordCountSpacesOnly(s: string): number {
  if (!s) return 0;
  const parts = s.trim().split(' ').filter(Boolean);
  return parts.length;
}
export const defaultTokenizer: Tokenizer = { count: wordCountSpacesOnly };
export const defaultWeights: ScoringWeights = { w_recency: 0.40, w_frequency: 0.20, w_user_priority: 0.25, w_dependency: 0.20, w_size: 0.15 };
export function budgetsFromTotal(B_total: number, fracN = 0.2, fracC = 0.6): Budgets {
  const B_N = Math.floor(B_total * fracN), B_C = Math.floor(B_total * fracC);
  return { B_total, B_N, B_C, B_sys: B_total - B_N - B_C };
}

// ---------- Utilities ----------
const uid = () => (globalThis.crypto && 'randomUUID' in globalThis.crypto ? (globalThis.crypto as any).randomUUID() : `${Date.now()}-${Math.random()}`);
const nowSec = () => Math.floor(Date.now()/1000);
const clamp = (x: number, lo=0, hi=1) => Math.max(lo, Math.min(hi, x));
const NL = String.fromCharCode(10);
const isoDate = () => new Date().toISOString().slice(0,10);
const tokens = (tok: Tokenizer, items: MemoryItem[]) => items.reduce((s,x)=>s+tok.count(x.content),0);
const uniq = (items: MemoryItem[]) => { const s = new Set<string>(); const out: MemoryItem[]=[]; for (const it of items){ if(!s.has(it.id)){ s.add(it.id); out.push(it);} } return out; };
function naiveSummarize(tok: Tokenizer, text: string, target: number){ if (tok.count(text) <= target) return text; const words = text.split(' '); return words.slice(0, Math.max(1,target)).join(' '); }

function toWordsLowerASCII(text: string): string[] {
  const out: string[] = [];
  let cur = '';
  const lower = (text || '').toLowerCase();
  for (let i=0; i<lower.length; i++){
    const c = lower.charCodeAt(i);
    const isAZ = c >= 97 && c <= 122; // a-z
    const is09 = c >= 48 && c <= 57;  // 0-9
    if (isAZ || is09){ cur += lower[i]; }
    else { if (cur) { out.push(cur); cur = ''; } }
  }
  if (cur) out.push(cur);
  return out;
}

// ---------- LLM Trim (functional adapter) ----------
export function makeTrimFn(
  tok: Tokenizer,
  llm: (payload: { system: string; user: string; targetTokens: number; tier: Tier; aggressiveness: Aggressiveness }) => Promise<{ trimmed: string; removed_notes?: string[] }|string> | null
) {
  const system = [
    'You are a careful memory-trimmer for a conversation.',
    '- Keep FACTS, bindings (entities, IDs), active tasks and constraints.',
    '- Remove fluff, repetition, and boilerplate. Do not invent.',
    '- Fit within TARGET_TOKENS with ~5% headroom.',
    'Return JSON: {"trimmed":"...","removed_notes":["..."]}'
  ].join(NL);

  const userTmpl = (tier: Tier, aggr: Aggressiveness, target: number, content: string) => [
    `Tier: ${tier} | Aggressiveness: ${aggr}`,
    `Target tokens: ${target}`,
    'Guidelines:',
    '- NEW: dedupe aggressively; compress verbose wording.',
    '- CURRENT: preserve nuance and references; merge repeats.',
    '- OLD: compact MANIFEST (summary + salient tags); no raw log.',
    '',
    'Input:',
    content
  ].join(NL);

  return async (content: string, targetTokens: number, tier: Tier, aggr: Aggressiveness) => {
    if (!llm) return { text: naiveSummarize(tok, content, targetTokens), notes: [] };
    try {
      const payload = { system, user: userTmpl(tier, aggr, targetTokens, content), targetTokens, tier, aggressiveness: aggr };
      const res = await llm(payload);
      const text = typeof res === 'string' ? res : (res.trimmed ?? '');
      const notes = typeof res === 'string' ? [] : (res.removed_notes ?? []);
      const bounded = tok.count(text) > targetTokens ? naiveSummarize(tok, text, targetTokens) : text;
      return { text: bounded, notes };
    } catch {
      return { text: naiveSummarize(tok, content, targetTokens), notes: [] };
    }
  };
}

// ---------- Scoring ----------
const recencyScore = (turnIdx: number, it: MemoryItem) => Math.exp(-0.35 * Math.max(0, turnIdx - it.lastTouchedTurn));
const dependencyScore = (C: MemoryItem[], it: MemoryItem) => {
  if (!it.dependencies?.length) return 0; const ids = new Set(C.map(x=>x.id));
  const ok = it.dependencies.filter(d=>ids.has(d)).length; return clamp(ok / it.dependencies.length);
};
const sizePenalty = (tok: Tokenizer, it: MemoryItem) => clamp(tok.count(it.content) / 200);
export const score = (state: State, it: MemoryItem) =>
  state.w.w_recency * recencyScore(state.turnIdx, it)
+ state.w.w_frequency * clamp(it.frequency / 5)
+ state.w.w_user_priority * clamp(it.userPriority)
+ state.w.w_dependency * dependencyScore(state.C, it)
- state.w.w_size * sizePenalty(state.tok, it);

// ---------- Core helpers ----------
export const ingest = (state: State, blocks: string[], tags: string[] = []): MemoryItem[] => {
  const out: MemoryItem[]=[]; for (const b of blocks){ if(!b||!b.trim()) continue; out.push({
    id: uid(), content: b.trim(), tags: [...tags], userPriority: 0, dependencies: [],
    createdTs: nowSec(), lastTouchedTs: nowSec(), lastTouchedTurn: state.turnIdx, frequency: 1, trimNotes: []
  }) } return out;
};

export async function llmTrimItems(state: State, items: MemoryItem[], B: number, tier: Tier, aggr: Aggressiveness): Promise<MemoryItem[]> {
  let copy = items.map(x=>({ ...x }));
  const tk = (x: MemoryItem) => state.tok.count(x.content);
  const total = () => copy.reduce((s,x)=>s+tk(x),0);
  if (total() <= B) return copy;

  copy.sort((a,b)=>tk(b)-tk(a));
  for (const it of copy){
    if (total() <= B) break;
    const overshoot = total() - B;
    const target = Math.max(1, tk(it) - Math.max(1, Math.floor(overshoot / 2)));
    const { text, notes } = await state.trimFn(it.content, target, tier, aggr);
    it.content = text; it.trimNotes = [...(it.trimNotes||[]), ...notes];
  }
  if (total() > B){
    const ratio = B / Math.max(1, total());
    for (const it of copy){
      const target = Math.max(1, Math.floor(tk(it)*ratio));
      if (tk(it) > target){
        const { text, notes } = await state.trimFn(it.content, target, tier, aggr);
        it.content = text; it.trimNotes = [...(it.trimNotes||[]), ...notes];
      }
    }
  }
  const out: MemoryItem[]=[]; let run=0; for (const it of copy){ const t=tk(it); if(run+t>B) break; out.push(it); run+=t; }
  return out;
}

export const retrieveFromO = async (state: State, query: string, k=5) => {
  const bag = new Set(toWordsLowerASCII(query));
  const sc = state.O.map(m=>{
    const text = [m.topic, m.summary256, ...m.salienceTags, ...m.rehydrationHints].join(' ').toLowerCase();
    const words = new Set(toWordsLowerASCII(text)); let overlap=0; for(const w of words) if(bag.has(w)) overlap++;
    return { m, s: overlap };
  }).sort((a,b)=>b.s-a.s);
  return sc.filter(x=>x.s>0).slice(0,k).map(x=>x.m);
};

export async function manifestFromItems(state: State, items: MemoryItem[], topicHint=""): Promise<EpisodeManifest> {
  const concat = items.map(x=>x.content).join(NL+NL);
  const { text } = await state.trimFn(concat, 256, "OLD", "high");
  const tags = Array.from(new Set(items.flatMap(x=>x.tags))).sort();
  const size = items.reduce((s,x)=>s+state.tok.count(x.content),0);
  return { episodeId: uid(), topic: topicHint || (tags[0]||"episode"), dateRange: [isoDate(), isoDate()],
    summary256: text, salienceTags: tags, links: [], rehydrationHints: tags.slice(0,8), sizeEstimateTokens: size,
    createdTs: nowSec(), lastUpdatedTs: nowSec() };
}

export async function archiveEvicted(state: State, evicted: MemoryItem[], topicHint=""): Promise<State> {
  if (!evicted.length) return state;
  const m = await manifestFromItems(state, evicted, topicHint);
  return { ...state, O: [...state.O, m] };
}

// ---------- Public API ----------
export function initState(params: { budgets: Budgets; tok?: Tokenizer; weights?: ScoringWeights; ttlTurns?: number; trimFn: State["trimFn"]; }): State {
  const { budgets, tok=defaultTokenizer, weights=defaultWeights, ttlTurns=8, trimFn } = params;
  return { turnIdx: 0, N: [], C: [], O: [], budgets, tok, w: weights, ttlTurns, trimFn };
}

export async function turn(state: State, params: { userMsg?: string; toolOutputs?: string[]; query?: string; pullFromO?: boolean; kRetrieve?: number; topicHint?: string; }): Promise<{ state: State; forLLM: string; }>{
  let s = { ...state, turnIdx: state.turnIdx + 1 };
  const userMsg = params.userMsg ?? ''; const toolOutputs = params.toolOutputs ?? [];
  const query = params.query ?? ''; const pullFromO = params.pullFromO ?? true;
  const kRetrieve = params.kRetrieve ?? 4; const topicHint = params.topicHint ?? '';

  let N = ingest(s, [userMsg, ...toolOutputs]);
  N = N.map(n=>({ ...n, lastTouchedTurn: s.turnIdx, lastTouchedTs: nowSec(), frequency: n.frequency+1 }));
  N = await llmTrimItems(s, N, s.budgets.B_N, "NEW", "high");

  if (pullFromO && query){
    const manifests = await retrieveFromO(s, query, kRetrieve);
    if (manifests.length){
      const text = manifests.map(m=>`[${m.topic}] ${m.summary256}`).join(NL);
      const R = ingest(s, [text], ["rehydrated"]);
      N = await llmTrimItems(s, [...N, ...R], s.budgets.B_N, "NEW", "high");
    }
  }

  const candidates = uniq([...s.C, ...N]).filter(it => it.userPriority >= 1 || (s.turnIdx - it.lastTouchedTurn) <= s.ttlTurns);
  const sorted = [...candidates].sort((a,b)=>score(s,b)-score(s,a));
  const packed: MemoryItem[] = []; let used=0;
  for (const it of sorted){ const t = s.tok.count(it.content); if (used + t <= s.budgets.B_C){ packed.push(it); used += t; } }
  let C = packed; if (used > s.budgets.B_C){ C = await llmTrimItems(s, packed, s.budgets.B_C, "CURRENT", "low"); }

  const prevIds = new Set(s.C.map(x=>x.id));
  const nextIds = new Set(C.map(x=>x.id));
  const evicted = s.C.filter(x=>!nextIds.has(x.id));
  s = await archiveEvicted(s, evicted, topicHint);

  s = { ...s, N, C };
  const forLLM = [...s.C, ...s.N].map(x=>x.content).join(NL+NL);
  return { state: s, forLLM };
}

export async function pin(state: State, id: string, priority=1): Promise<State>{
  const upd = (arr: MemoryItem[]) => arr.map(it => it.id===id ? { ...it, userPriority: Math.max(it.userPriority, priority) } : it);
  return { ...state, N: upd(state.N), C: upd(state.C) };
}

export async function unpin(state: State, id: string): Promise<State>{
  const upd = (arr: MemoryItem[]) => arr.map(it => it.id===id ? { ...it, userPriority: 0 } : it);
  return { ...state, N: upd(state.N), C: upd(state.C) };
}

export function exportState(state: State){ return state; }
```

### 5.1 Code Documentation -  What / Why / How

* **`initState`** *(WHAT)* creates a fresh immutable state with budgets, tokenizer, weights, TTL, and a `trimFn`. *(WHY)* Centralizes config; enables pure functional flow. *(HOW)* Returns a `State` object; you can serialize it directly.
* **`makeTrimFn`** *(WHAT)* builds a tier‑aware LLM trimming function. *(WHY)* LLM decides what to keep/remove per tier. *(HOW)* Sends a system+user prompt; if the model overshoots, a guard enforces the token target.
* **`ingest`** *(WHAT)* converts raw strings to `MemoryItem`s. *(WHY)* Normalize inputs from user and tools. *(HOW)* Adds IDs, timestamps, tags.
* **`llmTrimItems`** *(WHAT)* compresses a set of items to fit a token budget. *(WHY)* Bound prompt size. *(HOW)* Iteratively trims largest items with LLM, then proportionally; prefix‑packs if still oversized.
* **`retrieveFromO`** *(WHAT)* retrieves episode manifests by keyword overlap. *(WHY)* Cheap baseline to rehydrate history. *(HOW)* Scores overlap on {topic, summary, tags, hints}; replace with vector/BM25 later.
* **`manifestFromItems`** *(WHAT)* creates an **episode manifest** for evicted items. *(WHY)* Keep archive compact and searchable. *(HOW)* LLM summarizes to ~256 tokens, extracts tags/hints.
* **`archiveEvicted`** *(WHAT)* appends a new manifest to `O`. *(WHY)* Preserve value from evictions for future rehydration. *(HOW)* Calls `manifestFromItems` and updates state immutably.
* **`score`** *(WHAT)* ranks items for the working set. *(WHY)* Deterministic, budgeted packing of (C_t). *(HOW)* Combines recency, frequency, user priority, dependency alignment, and size penalty.
* **`turn`** *(WHAT)* runs one conversation turn. *(WHY)* Advance state with strict budgets and caching semantics. *(HOW)* `N_t` via ingest+trim (+rehydration), `C_t` via scored pack (+trim), evictions → `O_t` manifests. Returns `{ state, forLLM }`.
* **`pin`/`unpin`** *(WHAT)* user‑controlled retention. *(WHY)* Keep critical items in (C_t). *(HOW)* Adjust `userPriority`.
* **`exportState`** *(WHAT)* serialize. *(WHY)* Save/restore across sessions. *(HOW)* Return the `State` object.

## **Testing tip.** Because everything is pure and returns new state, you can snapshot‑test each step and mock `trimFn` to verify budget behavior deterministically.

## 6. What This Buys You

* **Continuity without bloat:** Caches the task’s working set, trimming judiciously.
* **Predictable cost:** Hard budgets (B_N,B_C) prevent overruns.
* **Stability:** Deterministic selection avoids summary drift and thrash.
* **Scalability:** Archive manifests keep long‑term memory cheap and retrievable.

---

## 7. Limitations & Future Work

* **Summarization quality** is bounded by the LLM used for trimming; weak models may over‑compress.
* **Retrieval quality** from manifests ((O_t)) benefits from vector search; the simple keyword baseline can be upgraded.
* **Task boundaries**: detecting episode boundaries for manifests can be learned rather than rule‑based.

---

## 8. Related Work (short)

* **Virtual/tiered context** for LLMs: *MemGPT* formalizes an OS‑like memory hierarchy and virtual context management.
* **Episodic memory and reflection**: *Generative Agents* introduces recency/relevance/importance‑scored memories and reflective summaries to guide agent behavior.
* **Framework memories**: libraries such as *LlamaIndex* provide short‑/long‑term memories with token limits and retrieval‑augmented prompting.
* **Long‑context models / external memory**: *Transformer‑XL*, *Memorizing Transformers*, and more recent decoupled‑memory approaches augment or extend the backbone network.

**Our difference:** a production‑first synthesis with (i) tier‑aware LLM trimming policies, (ii) cache semantics for the working set (TTL, pinning), (iii) manifest‑only archives with rehydration hints, and (iv) deterministic, budgeted selection - each with explicit guardrails for window size.

---

## 9. Conclusion

A three‑tier memory with tier‑aware LLM trimming, budgeted selection, and episodic manifests offers a simple, robust path to long‑horizon coherence. It makes the prompt feel larger without sacrificing stability or cost control, and it cleanly separates the **working set** from **archival memory** with clear upgrade paths (better retrieval, better trimmers) as systems evolve.
