---
layout: single
classes: wide
author_profile: true
title: "Your First Object-Oriented Agent"
seo_title: "Build your first AI agent with NOOA — tools, typing, and state in plain Python"
published: true
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessiodevoto/labs-OO-Agents/blob/first-notebook-openai/notebook_tutorials/01_your_first_agent.ipynb) &nbsp; [![GitHub](https://img.shields.io/badge/notebook-GitHub-black?logo=github)](https://github.com/alessiodevoto/labs-OO-Agents/blob/first-notebook-openai/notebook_tutorials/01_your_first_agent.ipynb)

> This tutorial walks you through the core ideas behind [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents) (NVIDIA Object-Oriented Agents). We'll build a `BaristaAgent` that recommends drinks to sleepy customers. Along the way we'll see that a NOOA agent is **just a Python object** — you add tools by adding methods, spin up new agents by instantiating the class, and strongly type its outputs like any regular Python function.

We're not going to sell you on a whole new paradigm — you won't walk away having learned some exotic new way to build software. What you *will* walk away with is plain Python, plus a small sprinkle of magic.

---

### Prerequisites

Install NOOA from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv add "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main"
```

NOOA is compatible with any LiteLLM-supported model — hosted or local. For hosted providers, you'll need an API key. Local providers (Ollama, vLLM, any OpenAI-compatible endpoint) need no API key, just an `api_base`.

---

### Setup

Pick a provider below. Replace `"your-api-key"` with a real key for hosted providers; local providers don't need one.

```python
from nooa.unifiedllm.registry import get_llm_client

# model = get_llm_client("claude-haiku-4-5", api_key="your-api-key")                                        # Anthropic
# model = get_llm_client("ollama_chat/qwen3:1.7b", api_base="http://localhost:11434")                       # Ollama (local, no key)
# model = get_llm_client("hosted_vllm/Qwen/Qwen3-1.7B", api_base="http://localhost:8000/v1")                # vLLM (local, no key)
model = get_llm_client("gpt-5.5", api_key="your-api-key")                                                # OpenAI
```

---

### Your First Agent

Let's define our first agent. In NOOA, you build an agent by subclassing `Agent` — the only required argument is the LLM that will power it. Here is a complete, working agent. It's small enough to read every line.

```python
from nooa import Agent

class BaristaAgent(Agent, llm=model):
    """You are a friendly barista at a small neighborhood cafe."""  # this becomes the agent's system prompt

    async def recommend_drink(self, customer_request: str) -> str:
        """Recommend a single drink to the customer based on what they just told you.
        Be warm and concise — one string sentence is plenty."""
        ...
```

Instantiate and call it. Notice that we `await` the method — generation methods are async.

```python
barista = BaristaAgent()
result = await barista.recommend_drink("Hi, I feel so sleepy, I need caffeine ☕️.")
print(result)
```

```
You sound like you need a classic double espresso—bright, bold, and ready to wake you right up.
```

That's it — a friendly recommendation, ready to serve. So what just happened?

Behind the scenes, NOOA does the work that keeps the interface this Pythonic. A few things worth noticing:

- the **class docstring** became the agent's system prompt
- the **method docstring** became the task description
- the **ellipsis (`...`)** is how you tell NOOA "this method is *agentic* — hand it off to the LLM instead of running it as regular Python"

Concretely, NOOA quietly assembles a prompt that looks roughly like:

```
<system prompt>
You are a friendly barista at a small neighborhood cafe.

<available methods>
recommend_drink(customer_request: str) -> str
Recommend a single drink to the customer based on what they just told you. Be warm and concise — one sentence is plenty.
</available methods>
```

Plus a few other pieces we'll unpack later. Let's call the agent one more time, just for fun:

```python
result = await barista.recommend_drink("I would love something that works with a cornetto, but as you know, no capuccinos after 11am.")
print(result)
```

```
A creamy cappuccino would be lovely with a cornetto—soft foam, gentle espresso, and just the right breakfast-café feel.
```

> 📝 **Takeaway:** the agent is an object.

---

### Adding Tools (That Is, Adding Class Methods)

The barista just offered a strong caffeine kick — but it's 9pm and a double espresso is definitely not the move. How do we teach the agent to respect a "no caffeine after 4pm" policy?

In most agent frameworks, you'd register a tool, describe it in a JSON schema, and wire it into the runtime. In NOOA, you just **add a method to the class**. Anything the agent can see on itself, it can call as a tool.

Let's add our first tool to `BaristaAgent`.

```python
from datetime import datetime
from nooa import Agent

class BaristaAgent(Agent, llm=model):
    """You are a friendly barista at a small neighborhood cafe."""

    def is_only_decaf_hour(self) -> bool:
        """Return True if we should only be serving decaf right now. After 2pm we go decaf-only so our customers can still sleep tonight."""
        return datetime.now().hour >= 14

    async def recommend_drink(self, customer_request: str) -> str:
        """Recommend a single drink to the customer based on what they just told you.
        Be warm and concise — one string sentence is plenty."""
        ...

barista = BaristaAgent()
await barista.recommend_drink("Hi, I feel so sleepy.")
```

```
'Since it's decaf-only right now, I'd make you a cozy decaf latte to perk up your mood without keeping you up later.'
```

Two things to sit with:

- **You didn't register `is_only_decaf_hour` anywhere.** No `@tool`, no JSON schema, no `tools=[...]` list. The framework rendered `doc(self)` into the system prompt (that `<self>` block you saw earlier), and the LLM discovered the method the same way you'd discover it while reading someone else's code.
- **The generation method used the helper without being told to.** Nothing in `recommend_drink`'s docstring mentions `is_only_decaf_hour`. The LLM spotted a method that looked relevant, called it, and folded the result into its recommendation. That's the whole "tools are just methods" idea in a single page.

> 📝 **Takeaway:** in NOOA, ordinary Python methods and agentic methods live side by side on the same class. The agent freely calls the deterministic ones as tools — no registration, no schema, no glue code.

---

### Strong Typing

What if our cafe only serves a fixed menu, and we want the agent to *only* recommend drinks we actually offer?

Most agent frameworks deal in a single data type: text. Text gets passed to tools, text gets exchanged between agents, text comes back as output — and then you painstakingly parse it into JSON, hoping the model got the shape right (which, even with the best models, it doesn't always). NOOA takes a different approach: because the agent lives inside a Python program, we can strongly type everything.

Let's put a real type on the return value of `recommend_drink`:

```python
from enum import Enum
from nooa import Agent

# Define a type for the return value
class Drink(Enum):
    ESPRESSO = "espresso"
    CAPPUCCINO = "cappuccino"
    FLAT_WHITE = "flat white"

class BaristaAgent(Agent, llm=model):
    """You are a friendly barista at a small neighborhood cafe."""

    def is_only_decaf_hour(self) -> bool:
        """Return True if we should only be serving decaf right now. After 4pm we go decaf-only so our customers can still sleep tonight."""
        return datetime.now().hour >= 16

    async def recommend_drink(self, customer_request: str) -> tuple[str, Drink]:
        """Pick the single best drink for the customer from the menu, based on what they told you."""
        ...

barista = BaristaAgent()
reason, drink = await barista.recommend_drink("Hi, I feel so sleepy.")
print("Barista: ", reason)
print("Recommended drink: ", drink)
print(type(drink))
```

```
Barista:  It's decaf-only right now, but I'd make you a cozy decaf cappuccino.
Recommended drink:  Drink.CAPPUCCINO
<enum 'Drink'>
```

This isn't just prompt engineering — **NOOA enforces the return type at runtime**. If the LLM returns something that isn't a valid `Drink`, the framework retries until it produces one. You get real Python objects back, not strings you have to reparse.

```python
# To see what "validation" actually means, try constructing a Drink from
# something that isn't on the menu. Python's Enum machinery rejects it:
try:
    Drink("matcha")
except ValueError as e:
    print(f"ValueError: {e}")

# When the LLM returns a value that doesn't match the declared type, NOOA
# catches this same error, feeds it back into the next turn as a hint, and
# asks the model to try again. You never see the invalid value in your code.
```

```
ValueError: 'matcha' is not a valid Drink
```

> 📝 **Takeaway:** strong typing all the way through.

---

### Agent State

Our cafe has a finite stash of coffee beans, and every drink burns through a few. Since our agent is *just a Python object*, giving it state is as easy as adding a field in `__init__`:

```python
from nooa import Agent
from typing import Union

class Drink(Enum):
    ESPRESSO = "espresso"
    CAPPUCCINO = "cappuccino"
    FLAT_WHITE = "flat white"
    TEA = "tea"

class BaristaAgent(Agent, llm=model):
    """You are a friendly barista at a small neighborhood cafe."""

    def __init__(self, coffee_beans: int) -> None:
        super().__init__()
        self.coffee_beans = coffee_beans

    def is_only_decaf_hour(self) -> bool:
        """Return True if we should only be serving decaf right now. After 6pm we go decaf-only so our customers can still sleep tonight."""
        return datetime.now().hour >= 18

    def serve(self, drink: Drink) -> Drink:
        """Serve a drink. Deducts 100 beans unless it's tea."""
        if drink != Drink.TEA:
            self.coffee_beans -= 100
        return drink

    async def recommend_drink(self, customer_request: str) -> tuple[str, Union[Drink, None]]:
        """Recommend a single drink to the customer based on what they just told you.
        Be warm and concise — one sentence is plenty.
        We currently have {self.coffee_beans} beans left. If we ran out, recommend a tea.
        Call self.serve(drink) with your chosen drink before returning."""
        ...
```

```python
barista = BaristaAgent(coffee_beans=200)
for _ in range(3):
    barista_answer, drink = await barista.recommend_drink("Something to keep me going, please.")
    print(f"Answer: {barista_answer}\nServed: {drink}\nBeans left: {barista.coffee_beans}")
```

```
Answer: Absolutely — an espresso should give you a nice little boost.
Served: Drink.ESPRESSO
Beans left: 100
Answer: Absolutely — an espresso should give you a nice little boost.
Served: Drink.ESPRESSO
Beans left: 0
Answer: I'm out of coffee beans just now, but a bright cup of tea will still keep you nicely refreshed.
Served: Drink.TEA
Beans left: 0
```

> 📝 **Takeaway:** the agent "sees" everything on itself — including itself.

---

### What Is Happening Under the Hood?

NOOA doesn't hide the prompt from you. `print_prompt` renders exactly what would be sent to the LLM for a given method call — the system prompt, the agent introspection block, and the task. Take a look:

```python
import nooa
await nooa.print_prompt(barista.recommend_drink, customer_request="stressed and running late")
```

<details>
<summary>Click to expand the full prompt output</summary>

`````
=== SYSTEM PROMPT  [BaristaAgent] ===

<system_prompt expr="self._resolve_system_prompt()">
You are a friendly barista at a small neighborhood cafe.
</system_prompt>

<strategy_prompt>
## Strategy

Jupyter-like Python session. Parameters pre-loaded as locals; state persists across cells. Use `await` directly, `print`/`pprint` to debug, `doc(obj)` to inspect types. You MUST call a tool each turn — **plain-text responses do NOT end the session**. To finish, call `return_result(value)`. Repeated text-only responses will abort the run with an error.

**Your two tools:**
- `execute_python(code)` — run a code cell
- `return_result(value)` — submit your final answer (also callable from inside `execute_python`)

## When to use which tool

Use `return_result(...)` directly for simple answers determinable from the inputs alone (yes/no, one field, a single lookup).

Use `execute_python(...)` for lists/batches, arithmetic, multi-step computation, transforms, or iteration. Always iterate in code — never construct large arrays by hand.

For language tasks (classification, extraction, interpretation), use LLM reasoning — answer directly via `return_result`, or delegate to a `@strategy(PredictStrategy())` standalone function (see below). Don't keyword-match or regex.

## Returning computed results

After computing in code, call `return_result(variable)` **from within** `execute_python()`. This passes the variable directly. Do NOT re-type computed values in a separate `return_result` tool call.

## Helpers

Define helpers at the top of the cell and call them by name. Existing methods on `self` are usable via `await self.method(...)`. Helpers persist as REPL locals across cells in this session.

```python
def normalize(x):
    return x.strip().lower()

cleaned = [normalize(v) for v in values]
```

## Fan-out generation

For per-item LLM work over a list, decorate a standalone async function with `@strategy(PredictStrategy())` and an ellipsis body. `asyncio.gather` runs the calls in parallel.

```python
@strategy(PredictStrategy())
async def detect_language(message: str) -> str:
    """Return the ISO 639-1 language code for {message} (e.g. 'en', 'fr', 'de', 'ja')."""
    ...

codes = await asyncio.gather(*(detect_language(m) for m in messages))
return_result(codes)
```

For iterative sub-tasks that need code execution, use `@strategy(CodeActStrategy())`. The sub-task must be strictly simpler than the current call to avoid infinite recursion.

## Restrictions (will throw)

- `eval`, `exec`, `compile`, `__import__`, `input`, `breakpoint`
- `globals`, `locals`, `vars`, `asyncio.run`, `loop.run_until_complete`
- Attaching callables to the agent: `self.foo = fn`, `setattr(self, 'foo', fn)`, `type(self).foo = fn`
</strategy_prompt>

<execution_context>
## Execution Context

These names are already in scope inside `execute_python()` (state persists across cells) — call them, don't re-import or re-define. Use `doc(name)` to inspect any type or function in detail.

```python
import nooa
from datetime import datetime
from enum import Enum
from nooa import Agent
from nooa.unifiedllm import get_llm_client
from typing import Union

class BaristaAgent: ...
class Drink: ...
```
Also in scope: exit, get_ipython, open, quit.
Always available without import: `self`, `print()`, `pprint()`, `doc()`, `return_result()`, plus stdlib `asyncio` and `typing`.
</execution_context>

<self expr="doc(type(self))">
class BaristaAgent:
    """You are a friendly barista at a small neighborhood cafe."""

    coffee_beans: coffee_beans

    def is_only_decaf_hour(self) -> bool:
        """Return True if we should only be serving decaf right now. After 4pm we go decaf-only so our customers can still sleep tonight."""
    def serve(self, drink: Drink) -> Drink:
        """Serve a drink. Deducts 100 beans unless it's tea."""
    async def recommend_drink(self, customer_request: str) -> tuple[str, Drink | None]:
        """
        Recommend a single drink to the customer based on what they just told you.
        Be warm and concise — one sentence is plenty.
        We currently have {self.coffee_beans} beans left. If we ran out, recommend a tea.
        Call self.serve(drink) with your chosen drink before returning.
        """
## Referenced Types
class Drink(Enum):
    ESPRESSO = 'espresso'
    CAPPUCCINO = 'cappuccino'
    FLAT_WHITE = 'flat white'
    TEA = 'tea'
</self>

=== TASK PROMPT  [BaristaAgent.recommend_drink] ===

## Task: recommend_drink

Recommend a single drink to the customer based on what they just told you.
Be warm and concise — one sentence is plenty.
We currently have 200 beans left. If we ran out, recommend a tea.
Call self.serve(drink) with your chosen drink before returning.

You are executing `recommend_drink` — code runs in the Execution Context above. Calling `self.recommend_drink(...)` would recurse.

=== PREFILL  [CodeActStrategy] ===

# Inspecting inputs for recommend_drink().
print(f"Task: recommend_drink()")
print(f"\ncustomer_request ({type(customer_request).__name__}):")
pprint(customer_request, max_length=25, max_string=2000, max_depth=4)
`````

</details>

Look through the output. There's no hidden state — this is exactly what the LLM sees. A few blocks worth naming:

- **`<system_prompt>`** — opens with your class docstring. This is the persona the model wears for every method on the class.
- **`<strategy_prompt>`** — a compact rulebook for how to act each turn: what tools exist (`execute_python`, `return_result`), when to use which, and how to finish a run. This is what teaches the LLM to "inhabit" the framework, and it's the same for every agent.
- **`<execution_context>`** — the imports, types, and helpers that will be in scope when the LLM writes code. Anything you import at module level shows up here.
- **`<self>`** — auto-generated documentation of the agent's public methods and fields, rendered from `doc(type(self))`. This is how the LLM discovers what the agent can do — no separate tool registry needed.
- **Task prompt** — your method docstring plus the rendered argument values for this call. Rendered live at call time.

That's the whole prompt. No hidden system messages, no template files, no per-tool JSON schemas glued on the side. The reason the "rulebook" stays short is that the framework is plain Python, and the model already knows Python.

> **Coming later:** `print_prompt` only shows the *outgoing* prompt. A later notebook introduces the live trace viewer for watching an entire run unfold — the LLM response, generated code, helper calls, retries, validation, and final return value.

---

### Recap

Five things the barista taught us:

- **Ellipsis `...` marks a generation method.** No decorator, no separate registry. If the body is `...`, the LLM implements it.
- **The class docstring is the system prompt; the method docstring is the task.** Rewriting a prompt means editing a docstring.
- **`{self.attr}` in docstrings is live.** Change the attribute, and the next call sees the new value.
- **Every non-hidden method on `self` is a tool.** No `@tool` decorator, no JSON schema, no registration step.
- **The return type annotation is the output contract.** Pydantic models are validated and retried automatically.
