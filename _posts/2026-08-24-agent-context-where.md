---
layout: single
classes: wide
author_profile: true
title: "Agents and Context: Where We Stand"
seo_title: "Agent harnesses and Context: Current state of the art agent harnesses about context management"
excerpt: "Why agent context windows fail in practice, how current systems compact context, and why context management may become part of the agent action space."
published: false
---

If you have used AI agents for long enough, you know that an agent is only as good as the context it receives.

The model might be capable of solving the task. But if the relevant observation has disappeared under fifty tool calls, three failed plans, and thousands of lines of logs, that capability does not help much. Context is the agent’s working memory, and agent trajectories are particularly good at filling it with noise.

A larger context window helps, but it also brings **context rot**: as the context grows, the model gradually becomes [less effective at using it](https://www.trychroma.com/research/context-rots). Context rot shows up in different shapes. One well-known example is [lost in the middle](https://arxiv.org/abs/2307.03172), where models struggle to use relevant information buried in the middle of a long prompt. But it is not just about position: context length alone [can hurt performance](https://arxiv.org/abs/2510.05381), even when the model retrieves the right information and the additional tokens contain almost no distractions.

A useful way to think about this is that every context window has a *smart zone*, where the model remains reliable, and a *dumb zone*, where its performance starts to degrade. This is not a fixed threshold: where the *dumb zone* begins depends on the model, the task, and how much noise has accumulated in the trajectory. 

> The maximum context window and the useful context window are not the same thing.

Why context rot happens is [an interesting question in its own right](https://proceedings.neurips.cc/paper_files/paper/2024/file/b1d35561c4a4a0e0b6012b2af531e149-Paper-Conference.pdf), and likely traces back to softmax attention having full support: as context length grows, the model struggles to distribute attention weight sharply enough to keep every token equally addressable. For now we ignore this question and focus on how context is managed today, then ask where things go next.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/agent-context/context-dumb-zone.svg" alt="Diagram showing a context window split into a smart zone and a dumb zone" style="max-width: 100%; width: 900px; display: block; margin: 1.5rem auto;">


## Compressing context at different levels
Something clear from the previous lines: 
 
> newer and smarter ways to handle long context are urgently needed.

Different approaches attack this problem at from different angles.

Every token added to the context also grows the model's KV cache — the cached key/value projections it uses to avoid recomputing attention over the whole prefix. Squeezing context can happen at either level: shrinking what the model sees, or shrinking how it stores what it has already seen.

At the **runtime level**, [KV cache compression/eviction/quantization](https://github.com/NVIDIA/kvpress) operates below the text seen by the agent. It prunes, merges, or quantizes cached key-value representations, reducing context memory usage and decoding cost without rewriting the conversation. The context still looks the same, but the model retains only an approximation of its internal representation. This has become a prolific research field over the past few years. Nevertheless, turning the memory savings reported in papers into actual latency and throughput gains is much harder: production inference engines rely on paged memory, fused attention kernels, continuous batching, prefix caching, and CUDA graphs, and changing the shape or precision of the cache can require modifications across this entire stack. Also, lower memory usage [does not](https://arxiv.org/abs/2607.08057) automatically produce end-to-end gains: the result also depends on conversion costs, kernel boundaries, and how well the method is integrated into the runtime.

At the **architecture level**, the model architecture itself is redesigned to work with a smaller context/cache in the first place. There are several examples of this. Some stay within the attention paradigm and attempt to make it cheaper: multi query, grouped query, sliding window attention, and [multi-head Latent Attention](https://arxiv.org/abs/2405.04434), which compresses keys and values into a low-rank latent vector before caching, shrinking the KV cache without discarding tokens outright. Others abandon softmax attention altogether, folding the sequence into a fixed-size state and trading exact token-level access for bounded memory. [State Space Models](https://arxiv.org/abs/2111.00396) and [Mamba](https://arxiv.org/abs/2312.00752) are examples of this approach.

Recent models often combine these ideas. [DeepSeek-V4](https://arxiv.org/abs/2606.19348) uses Compressed Sparse Attention and Heavily Compressed Attention to reduce the sequence dimension of its KV cache, while [Kimi K3](https://arxiv.org/abs/2607.24653) mixes three recurrent Kimi Delta Attention layers with one Gated MLA layer, using the recurrent state for efficiency and periodic full attention to preserve expressivity. Either way, the context is not shorter from the agent's perspective — the model just stores and processes it in a compressed representation.

At the **application level**, the simplest and most model-agnostic solution is *context compaction*. When the conversation approaches a threshold, **the system** asks a model to summarize the history, replaces the original messages with the summary, and continues.

If you have used a coding agent for a long enough session, you have probably seen this message:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/agent-context/context-compacted.png" alt="A coding agent terminal showing the message Context compacted" style="max-width: 100%; width: 420px; display: block; margin: 1.25rem auto;">

Compaction is effective, but coarse-grained. It works until it does not. A summary may preserve the current plan while quietly dropping why an earlier approach failed, a constraint introduced twenty turns ago, or the one error message that finally made the bug understandable. In coding, this can happen at exactly the worst moment: once the agent has accumulated enough task-specific state to make real progress.

It is also worth understanding what compaction costs. Before generating a token, the model must first read the input and build its KV cache. This initial phase is called *prefill*. To compact a conversation, the model first processes the existing history and generates a summary. The rewritten history must then be processed again before the agent can continue. In other words, compaction adds another generation and another prefill, while also invalidating much of the previously cached prompt.

Current implementations therefore keep the decision simple: **the harness watches the token count and triggers compaction near the limit**. This means that **compaction is a decision of the harness, not of the model.**

## What comes next?

The next natural next step is to let agents manage their own context.

The agent knows what it is trying to do. It knows which search result was a dead end, which error message is still relevant, and which intermediate plan has been superseded. In principle, it should be better placed than a fixed threshold to decide what to preserve, summarize, archive, or retrieve.

This does not mean removing the harness. Modern harnesses still implement a large amount of orchestration, and for good reason. Hard limits, safety constraints, and protocol correctness should remain deterministic. But deciding *which information still matters* is a semantic decision that may benefit from the agent’s understanding of the task.

Context management may therefore require both: the harness enforces the budget, while the agent decides how to spend it.

This is already a research direction, and the attempts so far can be grouped into two broad approaches.

The first is training, distilling, or RL-ing the model to manage context. [AgentFold](https://arxiv.org/abs/2510.24699) and [Context-Folding](https://arxiv.org/abs/2510.11967) turn folding or compression into agent actions. [ACM](https://arxiv.org/abs/2607.23809) and [ContextPilot](https://arxiv.org/abs/2608.28476) add context offloading and retrieval tools, then teach the agent when to use them.

The second is trying to one-shot the problem: give an unmodified model clearer state about its own context and see whether it can manage the budget without a learned policy. [VISTA](https://arxiv.org/abs/2606.30005) calls this *context proprioception*: block sizes, recency, archive status, remaining budget, and reversible archive/recovery tools exposed directly to the model. [SideQuest](https://arxiv.org/pdf/2602.22603) takes a similar training-free stance at the KV cache level, using a "companion" model to perform the compression.

But simply giving a model a `delete_context()` tool is not enough. The model must know how much space remains, which blocks are expensive, what has already been archived, and what it may need later. Even strong models struggle to infer this state from the raw prompt and to choose the right moment to intervene.

Training-free self-management therefore appears possible, but it is not yet something frontier models do reliably on their own.

## Context as an agent API

In [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents), we believe context will eventually become part of the agent’s action space.

Context and event history are first-class objects. Through [`self.context` and `self.events`](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/AGENTS.md#visibility), an agent can pin context, inspect and query previous events, and [collapse a selected event range](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/skills/nooa-context-and-state/SKILL.md#events) into a compact summary. These APIs can be exposed directly to the model, allowing the same agent that performs the task to decide what should remain in its working context.

The harness still enforces the hard limits. The agent controls the semantics.

We are not claiming that self-managed context is solved. The goal is to make the capability native and testable: can today’s models learn—or perhaps simply be prompted—to maintain their own working memory better than a fixed compaction policy?

> The next useful context window may not be a larger one. It may be one the agent knows how to manage.

