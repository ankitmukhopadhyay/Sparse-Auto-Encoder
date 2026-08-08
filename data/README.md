# Data

Every file here is a JSON object keyed by the six categories: `math`, `code`, `factual`, `creative`, `emotional`, `reasoning`.

| File | Contents | Written by |
|---|---|---|
| `prompts.json` | The 300 prompts used in the analysis, 50 per category | notebook 00 |
| `prompts_metadata.json` | Provenance for each of the 300, aligned by index | notebook 00 |
| `prompt_pool.json` | 13,851 candidates that survived quality filtering, before balanced sampling | notebook 00 |
| `prompts_original.json` | The 120 hand written prompts from the first iteration, kept for the before/after comparison in the EDA | authored by hand |

## How prompts.json was built

Candidates come from two public corpora of real user queries. Search Arena 24k supplies factual, reasoning and creative through its `primary_intent` annotation. WildChat 4.8M supplies code, math and emotional through its `topic` field, streamed over the HuggingFace API rather than downloaded, since the full corpus is 27 GB.

Source labels do not line up one to one with our six categories, so notebook 00 maps them and then sub-classifies where a single source label spans two of ours. WildChat's `tutoring_or_teaching` splits between math and reasoning on math keyword detection, and `greetings_and_chitchat` passes an emotional sub-filter.

Five filters run before anything enters the pool:

1. Length between 3 and 200 words
2. Alphabetic character ratio above 0.5, which rejects URL and symbol dominated text
3. Type token ratio above 0.3, which rejects repetition spam
4. At most 512 GPT-2 tokens, so extraction never truncates a prompt
5. Exact match deduplication across both sources

Balanced sampling then draws 50 per category with `random.sample` under seed 42.

## Metadata shape

`prompts_metadata.json[category][i]` describes `prompts.json[category][i]`:

```json
{ "source": "wildchat", "original_topic": "tutoring_or_teaching", "sub": "math_keyword" }
{ "source": "search_arena", "original_intent": "Analysis" }
```

## Caveats worth knowing before you use these

The pool is unbalanced by construction. Factual has 7,004 candidates, emotional has 208, so the two categories are sampled at very different rates from their sources. The prompts are multilingual, because the source corpora are, and the filters do not enforce English. Some prompts labeled math are word problems wrapped in long context rather than bare arithmetic. That is what people actually type, and it is the reason we stopped writing our own.
