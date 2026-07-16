# Ouroboros — X launch thread (draft)

**1/**
Your AI agent produces exhaust every session: tool calls, compressions, repeated requests, hard decisions, finished tasks.

Stock agents let it dissipate.

I built a plugin pack that eats it. 🐍

**Ouroboros — the self-improvement flywheel for Hermes Agent.** 6 plugins, 1 loop. 🧵

**2/**
The loop:

🧠 REMEMBER everything →
🛡 TRUST it to act →
🏛 DECIDE in council →
🤖 AUTOMATE the repetition →
⚒ LEARN from the residue →
→ better agent → better sessions → repeat.

Each plugin feeds the next. The output of the loop is the input.

**3/**
🧠 `echo` — your agent can't *not* remember.

Every turn, before the model sees your message, echo FTS5-searches your own past sessions and injects the 2–3 most relevant excerpts. Local. Zero config. No embeddings, no external service.

"Why did the gateway die last week?" — last week's session is already in front of the model.

**4/**
🗄 `archive` — the first real alternative context engine for Hermes.

Compression today is summarize-and-discard. Archive wraps the stock compressor and *seals every discarded window into a searchable archive first* — and gives the agent tools to pull anything back, mid-conversation.

Compression without amnesia.

**5/**
🛡 `seatbelt` — run /yolo with a safety net.

Declarative YAML policy over EVERY tool call: block rules, human-approval escalation, per-tool rate limits, per-session call budgets (the runaway-loop circuit breaker), and a 90-day audit trail.

~/.ssh writes: blocked. .env reads: ask a human. 500-call loop: tripped.

**6/**
🏛 `council` — don't ask one agent what five would argue about.

One question → 2–5 tool-using subagents with adversarial lenses (analyst / skeptic / pragmatist / contrarian) → a judge that scores each position, synthesizes the strongest answer, and **preserves the dissent**.

/council redteam — your plan gets attacked before production attacks it.

**7/**
🤖 `autopilot` — "you asked this 5 times, want it automated?"

It mines your history for recurring intents, drafts self-contained cron jobs, and queues them for one-tap approval. Dismissed never resurfaces. Nothing is created without your yes.

The compound-interest plugin.

**8/**
⚒ `forge` — your usage is the dataset.

Harvests your best real sessions (heuristics + rubric LLM pass), converts them to ShareGPT trajectories, dedupes, appends to a local JSONL.

A personal fine-tuning corpus of *your own corrected workflows* — growing while you sleep.

**9/**
Everything is:
• local-first (SQLite + your existing model — no new keys, no new services)
• fail-soft (a plugin error never breaks a turn)
• independent (enable any subset — they compound)

One command to see the whole loop: `/ouroboros`

**10/**
Install:

git clone github.com/JamesFincher/hermes-ouroboros
./install.sh
hermes plugins enable seatbelt echo archive council autopilot forge ouroboros

The snake that eats its tail. The agent that eats its history.

⭐ if the loop spins for you.

---

## Suggested quote-tweet / reply hooks

- "The ContextEngine ABC shipped with an empty plugins/context_engine/ directory. This is what it was for."
- "session_search is a tool your agent must remember to call. echo is what happens when it doesn't have to."
- "MoA gives one turn three advisors. council gives one question five armed adversaries."
- "batch_runner generates synthetic training data. forge harvests the corpus you already paid for — in tokens and in tears."

## Asset checklist

- [x] Repo banner (assets/banner.svg)
- [ ] 30-sec terminal capture of /ouroboros → echo answer → /council verdict (asciinema → gif)
- [ ] Pin thread after posting; reply with repo link in tweet 1 and 10
