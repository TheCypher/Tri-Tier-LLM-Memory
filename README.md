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

## 5. TypeScript Reference Implementation

Below is a compact TypeScript implementation of the core data types and control loop. It is framework‑agnostic: plug in your tokenizer, LLM client, and persistence.

```ts
// tri-tier-memory.ts

// ----- Types -----
export interface Tokenizer {
  count(text: string): number;
}

export interface LLMTrimPayload {
  system: string;
  user: string;
  targetTokens: number;
  tier: "NEW" | "CURRENT" | "OLD";
  aggressiveness: "high" | "medium" | "low";
}

export type LLMFn = (p: LLMTrimPayload) => Promise<{ trimmed: string; removed_notes?: string[] } | string>;

export interface MemoryItem {
  id: string;
  content: string;
  tags: string[];
  userPriority: number; // 0..1
  dependencies: string[];
  createdTs: number;
  lastTouchedTs: number;
  lastTouchedTurn: number;
  frequency: number;
  trimNotes?: string[];
}

export interface EpisodeManifest {
  episodeId: string;
  topic: string;
  dateRange: [string, string]; // ISO dates
  summary256: string;
  salienceTags: string[];
  links: string[]; // external refs/IDs
  rehydrationHints: string[];
  sizeEstimateTokens: number;
  createdTs: number;
  lastUpdatedTs: number;
}

export interface Budgets {
  B_total: number;
  B_N: number;
  B_C: number;
  B_sys: number;
}

export function budgetsFromTotal(B_total: number, fracN = 0.2, fracC = 0.6): Budgets {
  const B_N = Math.floor(B_total * fracN);
  const B_C = Math.floor(B_total * fracC);
  const B_sys = B_total - B_N - B_C;
  return { B_total, B_N, B_C, B_sys };
}

export interface ScoringWeights {
  w_recency: number;
  w_frequency: number;
  w_user_priority: number;
  w_dependency: number;
  w_size: number; // subtracted
}

export const defaultWeights: ScoringWeights = {
  w_recency: 0.40,
  w_frequency: 0.20,
  w_user_priority: 0.25,
  w_dependency: 0.20,
  w_size: 0.15,
};

// ----- Utilities -----
function uid() { return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`; }
function nowSec() { return Math.floor(Date.now() / 1000); }
function clamp(x: number, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, x)); }

export const defaultTokenizer: Tokenizer = {
  count(text: string) { return text ? (text.trim().split(/\s+/).length) : 0; }
};

// Naive summarizer as last-ditch guard (if LLM overshoots or is absent)
function naiveSummarize(text: string, target: number, tok: Tokenizer): string {
  if (tok.count(text) <= target) return text;
  const words = text.split(/\s+/);
  return words.slice(0, Math.max(1, target)).join(" ");
}

// ----- LLM Trimmer -----
export class LLMTrimmer {
  constructor(private llmFn: LLMFn | null, private tok: Tokenizer = defaultTokenizer) {}

  system = `You are a careful memory-trimmer for a conversation.\n` +
           `- Keep FACTS, bindings (entities, IDs), active tasks and constraints.\n` +
           `- Remove fluff, repetition, and boilerplate. Do not invent.\n` +
           `- Fit within TARGET_TOKENS with ~5% headroom.\n` +
           `Return JSON: {"trimmed":"...","removed_notes":["..."]}`;

  userTemplate(tier: "NEW"|"CURRENT"|"OLD", aggr: "high"|"medium"|"low", target: number, content: string) {
    return `Tier: ${tier} | Aggressiveness: ${aggr}\nTarget tokens: ${target}\n` +
           `Guidelines:\n- NEW: dedupe aggressively; compress verbose wording.\n` +
           `- CURRENT: preserve nuance and references; merge repeats.\n` +
           `- OLD: produce a compact MANIFEST (summary + salient tags); no raw log.\n\n` +
           `Input:\n${content}`;
  }

  async trim(content: string, targetTokens: number, tier: "NEW"|"CURRENT"|"OLD", aggr: "high"|"medium"|"low"): Promise<{text: string, notes: string[]}> {
    if (!this.llmFn) {
      return { text: naiveSummarize(content, targetTokens, this.tok), notes: [] };
    }
    const payload: LLMTrimPayload = {
      system: this.system,
      user: this.userTemplate(tier, aggr, targetTokens, content),
      targetTokens, tier, aggressiveness: aggr,
    };
    try {
      const res = await this.llmFn(payload);
      let text = typeof res === "string" ? res : (res.trimmed ?? "");
      const notes = typeof res === "string" ? [] : (res.removed_notes ?? []);
      if (this.tok.count(text) > targetTokens) {
        text = naiveSummarize(text, targetTokens, this.tok);
      }
      return { text, notes };
    } catch {
      return { text: naiveSummarize(content, targetTokens, this.tok), notes: [] };
    }
  }
}

// ----- Memory Manager -----
export class MemoryManager {
  private turnIdx = 0;
  private N: MemoryItem[] = [];
  private C: MemoryItem[] = [];
  private O: EpisodeManifest[] = [];

  constructor(
    private budgets: Budgets,
    private tok: Tokenizer = defaultTokenizer,
    private w: ScoringWeights = defaultWeights,
    private ttlTurns = 8,
    private trimmer = new LLMTrimmer(null, defaultTokenizer),
  ) {}

  private tokens(items: MemoryItem[]) { return items.reduce((s, x) => s + this.tok.count(x.content), 0); }

  private ingest(blocks: string[], tags: string[] = []): MemoryItem[] {
    const out: MemoryItem[] = [];
    for (const b of blocks) {
      if (!b || !b.trim()) continue;
      out.push({
        id: uid(), content: b.trim(), tags: [...tags], userPriority: 0,
        dependencies: [], createdTs: nowSec(), lastTouchedTs: nowSec(),
        lastTouchedTurn: this.turnIdx, frequency: 1, trimNotes: []
      });
    }
    return out;
  }

  private uniq(items: MemoryItem[]) {
    const seen = new Set<string>();
    const out: MemoryItem[] = [];
    for (const it of items) { if (!seen.has(it.id)) { seen.add(it.id); out.push(it); } }
    return out;
  }

  private recencyScore(it: MemoryItem) {
    const dt = Math.max(0, this.turnIdx - it.lastTouchedTurn);
    return Math.exp(-0.35 * dt);
  }

  private dependencyScore(it: MemoryItem) {
    if (!it.dependencies?.length) return 0;
    const idsInC = new Set(this.C.map(x => x.id));
    const ok = it.dependencies.filter(d => idsInC.has(d)).length;
    return clamp(ok / it.dependencies.length);
  }

  private sizePenalty(it: MemoryItem) {
    return clamp(this.tok.count(it.content) / 200);
  }

  private score(it: MemoryItem) {
    return this.w.w_recency * this.recencyScore(it)
         + this.w.w_frequency * clamp(it.frequency / 5)
         + this.w.w_user_priority * clamp(it.userPriority)
         + this.w.w_dependency * this.dependencyScore(it)
         - this.w.w_size * this.sizePenalty(it);
  }

  private async llmTrimItems(items: MemoryItem[], B: number, tier: "NEW"|"CURRENT"|"OLD", aggr: "high"|"medium"|"low"): Promise<MemoryItem[]> {
    const copy = items.map(x => ({ ...x }));
    const tk = (x: MemoryItem) => this.tok.count(x.content);
    if (copy.reduce((s, x) => s + tk(x), 0) <= B) return copy;

    copy.sort((a, b) => tk(b) - tk(a));
    const total = () => copy.reduce((s, x) => s + tk(x), 0);

    for (const it of copy) {
      if (total() <= B) break;
      const overshoot = total() - B;
      const target = Math.max(1, tk(it) - Math.max(1, Math.floor(overshoot / 2)));
      const { text, notes } = await this.trimmer.trim(it.content, target, tier, aggr);
      it.content = text; it.trimNotes = [...(it.trimNotes || []), ...notes];
    }

    if (total() > B) {
      const ratio = B / Math.max(1, total());
      for (const it of copy) {
        const target = Math.max(1, Math.floor(tk(it) * ratio));
        if (tk(it) > target) {
          const { text, notes } = await this.trimmer.trim(it.content, target, tier, aggr);
          it.content = text; it.trimNotes = [...(it.trimNotes || []), ...notes];
        }
      }
    }

    // Prefix-pack if still too big
    const out: MemoryItem[] = [];
    let running = 0;
    for (const it of copy) {
      const t = tk(it);
      if (running + t > B) break;
      out.push(it); running += t;
    }
    return out;
  }

  private async retrieveFromO(query: string, k = 5): Promise<EpisodeManifest[]> {
    const bag = new Set((query || "").toLowerCase().match(/\w+/g) || []);
    const scored = this.O.map(m => {
      const text = [m.topic, m.summary256, ...m.salienceTags, ...m.rehydrationHints].join(" ").toLowerCase();
      const words = new Set(text.match(/\w+/g) || []);
      let overlap = 0; for (const w of words) if (bag.has(w)) overlap++;
      return { m, s: overlap };
    }).sort((a, b) => b.s - a.s);
    return scored.filter(x => x.s > 0).slice(0, k).map(x => x.m);
  }

  private async manifestFromItems(items: MemoryItem[], topicHint = ""): Promise<EpisodeManifest> {
    const iso = () => new Date().toISOString().slice(0, 10);
    const concat = items.map(x => x.content).join("\n\n");
    const { text } = await this.trimmer.trim(concat, 256, "OLD", "high");
    const tags = Array.from(new Set(items.flatMap(x => x.tags))).sort();
    const size = items.reduce((s, x) => s + this.tok.count(x.content), 0);
    return {
      episodeId: uid(), topic: topicHint || (tags[0] || "episode"),
      dateRange: [iso(), iso()], summary256: text, salienceTags: tags,
      links: [], rehydrationHints: tags.slice(0, 8), sizeEstimateTokens: size,
      createdTs: nowSec(), lastUpdatedTs: nowSec(),
    };
  }

  private async archiveEvicted(evicted: MemoryItem[], topicHint = "") {
    if (!evicted.length) return;
    const m = await this.manifestFromItems(evicted, topicHint);
    this.O.push(m);
  }

  // ----- Public API -----
  async turn(params: {
    userMsg?: string;
    toolOutputs?: string[];
    query?: string;
    pullFromO?: boolean;
    kRetrieve?: number;
    topicHint?: string;
  }): Promise<{ N: MemoryItem[]; C: MemoryItem[]; O: EpisodeManifest[]; forLLM: string; }> {
    this.turnIdx += 1;
    const userMsg = params.userMsg ?? "";
    const toolOutputs = params.toolOutputs ?? [];
    const query = params.query ?? "";
    const pullFromO = params.pullFromO ?? true;
    const kRetrieve = params.kRetrieve ?? 4;
    const topicHint = params.topicHint ?? "";

    // 1) Ingest → N_t
    let N = this.ingest([userMsg, ...toolOutputs]);
    for (const n of N) { n.lastTouchedTurn = this.turnIdx; n.lastTouchedTs = nowSec(); n.frequency++; }
    N = await this.llmTrimItems(N, this.budgets.B_N, "NEW", "high");

    // 2) Optional rehydration from O
    if (pullFromO && query) {
      const manifests = await this.retrieveFromO(query, kRetrieve);
      if (manifests.length) {
        const text = manifests.map(m => `[${m.topic}] ${m.summary256}`).join("\n");
        const R = this.ingest([text], ["rehydrated"]);
        N = await this.llmTrimItems([...N, ...R], this.budgets.B_N, "NEW", "high");
      }
    }

    // 3) Merge → C_t (TTL + scored pack)
    const candidates = this.uniq([...this.C, ...N]).filter(it => it.userPriority >= 1 || (this.turnIdx - it.lastTouchedTurn) <= this.ttlTurns);
    const sorted = [...candidates].sort((a, b) => this.score(b) - this.score(a));
    const packed: MemoryItem[] = [];
    let used = 0;
    for (const it of sorted) {
      const t = this.tok.count(it.content);
      if (used + t <= this.budgets.B_C) { packed.push(it); used += t; }
    }
    let C = packed;
    if (used > this.budgets.B_C) {
      C = await this.llmTrimItems(packed, this.budgets.B_C, "CURRENT", "low");
    }

    // 4) Evict & archive manifests
    const prevIds = new Set(this.C.map(x => x.id));
    const nextIds = new Set(C.map(x => x.id));
    const evicted = this.C.filter(x => !nextIds.has(x.id));
    await this.archiveEvicted(evicted, topicHint);

    this.N = N; this.C = C;

    const forLLM = [...this.C, ...this.N].map(x => x.content).join("\n\n");
    return { N: this.N, C: this.C, O: this.O, forLLM };
  }

  async pin(id: string, priority = 1): Promise<boolean> {
    for (const it of [...this.C, ...this.N]) { if (it.id === id) { it.userPriority = Math.max(it.userPriority, priority); return true; } }
    return false;
  }

  async unpin(id: string): Promise<boolean> {
    for (const it of [...this.C, ...this.N]) { if (it.id === id) { it.userPriority = 0; return true; } }
    return false;
  }

  exportState() {
    return { turnIdx: this.turnIdx, budgets: this.budgets, weights: this.w, ttlTurns: this.ttlTurns, N: this.N, C: this.C, O: this.O };
  }
}
```

**Plug‑in points.** Replace `Tokenizer` with your model’s tokenizer, implement `LLMFn` for your provider, and persist `EpisodeManifest` objects in your database or object store. The class returns a `forLLM` string ready to be appended to your system/instruction prompt.

---

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
