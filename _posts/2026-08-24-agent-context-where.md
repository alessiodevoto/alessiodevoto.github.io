---
layout: single
classes: wide
author_profile: true
title: "Agents and Context: Where We Stand"
seo_title: "Agent harnesses and Context: Current state of state of art agent harnesses about context management"
excerpt: "Why agent context windows fail in practice, how current systems compact context, and why context management may become part of the agent action space."
published: true
---

If you have used AI agents for long enough, you know that an agent is only as good as the context it receives.

The model might be capable of solving the task. But if the relevant observation has disappeared under fifty tool calls, three failed plans, and thousands of lines of logs, that capability does not help much. Context is the agent’s working memory, and agent trajectories are particularly good at filling it with noise.

A larger context window helps, but it does not solve the problem. Long context comes with a bunch of issues. One is [lost in the middle](https://arxiv.org/abs/2307.03172): models can be much worse at using relevant information when it appears in the middle of a long prompt than when it appears near the beginning or the end. More recent work shows something even less convenient: [context length alone can hurt performance](https://arxiv.org/abs/2510.05381), even when the model retrieves the relevant information correctly and the additional tokens contain almost no distraction. This degradation is often called [context rot](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): as context grows, the model gradually becomes less effective at using it.

Practitioners sometimes call the unreliable part of the window the *dumb zone*, as opposed to the *smart zone* where the model is still reliable ([reference](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)). I would treat it as a useful operational metaphor, not a fixed benchmark threshold: where degradation starts depends on the model, the task, and the amount of noise in the trajectory.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/agent-context/context-dumb-zone.svg" alt="Diagram showing a context window split into a smart zone and a dumb zone" style="max-width: 100%; width: 900px; display: block; margin: 1.5rem auto;">

So this can all be summarized as: 

> The maximum context window and the useful context window are not the same thing.

And new and smarted way to handle long context are urgently needed.

## Compressing context at different levels

Different approaches attack this problem at different levels.

At the *runtime level*, [KV cache compression/eviction](https://github.com/NVIDIA/kvpress) operates below the text seen by the agent. It prunes, merges, or quantizes cached key-value representations, reducing memory usage and decoding cost without rewriting the conversation. The prompt still looks the same, but the model retains only an approximation of its internal representation. This has become a prolific research field over the past few years. However, turning the memory savings reported in papers into actual latency and throughput gains is much harder. Production inference engines rely on paged memory, fused attention kernels, continuous batching, prefix caching, and CUDA graphs, and changing the shape or precision of the cache can require modifications across this entire stack. Also, lower memory usage [does not](https://arxiv.org/abs/2607.08057) automatically produce end-to-end gains: the result also depends on conversion costs, kernel boundaries, and how well the method is integrated into the runtime.

At the *architecture level*, the model itself is redesigned to need less cache in the first place. There are several examples of this. Some are "variations" of the attention architecture: multi query, grouped query, sliding window and and the more recent [Multi-head Latent Attention](https://arxiv.org/abs/2405.04434) that compresses the keys and values of each token into a lower-dimensional latent representation. Others use a different operation altogether, go further and fold the sequence into a fixed-size state, trading exact token-level access for bounded memory. State Space Models and Mambda are examples of this approach. Recent models combine these ideas: [DeepSeek-V4](https://arxiv.org/abs/2606.19348) uses Compressed Sparse Attention and Heavily Compressed Attention to reduce the sequence dimension of its KV cache, while [Kimi K3](https://arxiv.org/abs/2607.24653) mixes three recurrent Kimi Delta Attention layers with one global MLA layer, using the recurrent state for efficiency and periodic full attention to preserve expressivity. Either way, the context is not shorter from the agent's perspective — the model just stores and processes it in a compressed representation.

At the *application level*, the simplest model-agnostic solution is *context compaction*. When the conversation approaches a threshold, the system asks a model to summarize the history, replaces the original messages with the summary, and continues.

If you have used a coding agent for a long enough session, you have probably seen this message:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/agent-context/context-compacted.png" alt="A coding agent terminal showing the message Context compacted" style="max-width: 100%; width: 420px; display: block; margin: 1.25rem auto;">

Compaction is effective, but coarse-grained. It works until it does not. A summary may preserve the current plan while quietly dropping why an earlier approach failed, a constraint introduced twenty turns ago, or the one error message that finally made the bug understandable. In coding, this can happen at exactly the worst moment: once the agent has accumulated enough task-specific state to make real progress.

<blockquote class="twitter-tweet" data-dnt="true" data-align="center" style="margin-left: auto; margin-right: auto;">
  <p lang="en" dir="ltr">The most dreadful output from a coding agent:<br><br>Context compacted</p>
  &mdash; Jean-Francois Puget (@JFPuget) <a href="https://x.com/JFPuget/status/2092598869167120704">August 26, 2026</a>
</blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

It is also worth understanding what compaction costs. Before generating a token, the model must first read the input and build its KV cache. This initial phase is called *prefill*. To compact a conversation, the model first processes the existing history and generates a summary. The rewritten history must then be processed again before the agent can continue. In other words, compaction adds another generation and another prefill, while also invalidating much of the previously cached prompt.

Current implementations therefore keep the decision simple: **the harness watches the token count and triggers compaction near the limit**. **Compaction is therefore a decision of the harness, not of the model.**

> The harness knows that the context is full. The agent does not.

## What comes next?

The next natural next step is to let agents manage their own context.

The agent knows what it is trying to do. It knows which search result was a dead end, which error message is still relevant, and which intermediate plan has been superseded. In principle, it should be better placed than a fixed threshold to decide what to preserve, summarize, archive, or retrieve.

This does not mean removing the harness. Modern harnesses still implement a large amount of orchestration, and for good reason. Hard limits, safety constraints, and protocol correctness should remain deterministic. But deciding *which information still matters* is a semantic decision that may benefit from the agent’s understanding of the task.

Context management may therefore require both: the harness enforces the budget, while the agent decides how to spend it.

This is already a research direction, and the attempts so far can be grouped into two broad approaches.

The first is training, distilling, or RL-ing the model to manage context. [AgentFold](https://arxiv.org/abs/2510.24699) and [Context-Folding](https://arxiv.org/abs/2510.11967) turn folding or compression into agent actions. [ACM](https://arxiv.org/abs/2607.23809) and [ContextPilot](https://arxiv.org/abs/2608.28476) add context offloading and retrieval tools, then teach the agent when to use them.

The second is trying to one-shot the problem: give an unmodified model clearer state about its own context and see whether it can manage the budget without a learned policy. [VISTA](https://arxiv.org/abs/2606.30005) calls this *context proprioception*: block sizes, recency, archive status, remaining budget, and reversible archive/recovery tools exposed directly to the model.

But simply giving a model a `delete_context()` tool is not enough. The model must know how much space remains, which blocks are expensive, what has already been archived, and what it may need later. Even strong models struggle to infer this state from the raw prompt and to choose the right moment to intervene.

Training-free self-management therefore appears possible, but it is not yet something frontier models do reliably on their own.

## Context as an agent API

In [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents), we believe context will eventually become part of the agent’s action space.

Context and event history are first-class objects. Through `self.context` and `self.events`, an agent can inspect its history, query previous events, and collapse a selected range into a compact summary. These APIs can be exposed directly to the model, allowing the same agent that performs the task to decide what should remain in its working context.

The harness still enforces the hard limits. The agent controls the semantics.

We are not claiming that self-managed context is solved. The goal is to make the capability native and testable: can today’s models learn—or perhaps simply be prompted—to maintain their own working memory better than a fixed compaction policy?

> The next useful context window may not be a larger one. It may be one the agent knows how to manage.
