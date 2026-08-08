# Prompt-Feature Activation in LLMs: Project Explainer

**CS 463 — Machine Learning, Spring 2025 | University of San Francisco**  
**Author:** Ankit Mukhopadhyay

This document explains every concept and implementation detail from the project at two levels:
- **"The Real Deal"** — for an ML researcher or veteran engineer
- **"If I Were 5"** — for someone with zero technical background

---

## Table of Contents

1. [What Is This Project About?](#1-what-is-this-project-about)
2. [The Model: GPT-2 Small](#2-the-model-gpt-2-small)
3. [Layers and the Residual Stream](#3-layers-and-the-residual-stream)
4. [Superposition: The Core Problem](#4-superposition-the-core-problem)
5. [Sparse Autoencoders (SAEs)](#5-sparse-autoencoders-saes)
6. [Features: What SAEs Give Us](#6-features-what-saes-give-us)
7. [Our Dataset: 300 Real Human Prompts](#7-our-dataset-300-real-human-prompts)
8. [Feature Extraction Pipeline](#8-feature-extraction-pipeline)
9. [Sparsity Analysis](#9-sparsity-analysis)
10. [Category Similarity and UMAP](#10-category-similarity-and-umap)
11. [Separability: Fisher's Discriminant Ratio](#11-separability-fishers-discriminant-ratio)
12. [Discriminative Features](#12-discriminative-features)
13. [Activation Patching: Causal Verification](#13-activation-patching-causal-verification)
14. [Feature Steering](#14-feature-steering)
15. [Key Findings Summary](#15-key-findings-summary)

---

## 1. What Is This Project About?

### If I Were 5...

Imagine you have a very smart robot friend. When you ask it a math question, something different happens inside its brain than when you ask it to write a story. We wanted to peek inside the robot's brain and see *which parts light up* for different kinds of questions. Do math questions light up the "math part"? Do story questions light up the "story part"? That's what we figured out!

### The Real Deal

This is a **mechanistic interpretability** study. We investigate how different categories of prompts (math, code, factual, creative, emotional, reasoning) activate distinct sets of **sparse autoencoder (SAE) features** in GPT-2 Small's residual stream. The core research question:

> *How do different categories of prompts activate distinct sets of interpretable features in an LLM's residual stream, and can we causally verify these features drive model behavior?*

We use pre-trained SAEs from the `gpt2-small-res-jb` release (Joseph Bloom, 24,576 features per layer), extract feature activations for 300 real human prompts across 3 layers, analyze discriminative patterns, and perform causal verification via activation patching and feature steering.

---

## 2. The Model: GPT-2 Small

### If I Were 5...

Our robot is called GPT-2. It's a word-guessing champion — you say the beginning of a sentence, and it guesses what word comes next, over and over. It has 124 million tiny knobs inside (called "parameters") and a brain made of 12 floors (called "layers"). It reads your words, thinks through all 12 floors, and then gives you an answer.

### The Real Deal

GPT-2 Small is a 124M-parameter autoregressive transformer with:
- **12 transformer layers** (blocks), each containing multi-head self-attention (12 heads) and an MLP
- **d_model = 768**: the residual stream dimensionality
- **Vocabulary**: 50,257 BPE tokens
- We load it via `transformer_lens` (HookedTransformer) for hook-based activation caching
- Running on CUDA (RTX 3050 Ti) in bfloat16 for extraction (float32 for patching stability)

```python
model = HookedTransformer.from_pretrained("gpt2", dtype="bfloat16", fold_ln=False,
                                           center_writing_weights=False, center_unembed=False)
```

---

## 3. Layers and the Residual Stream

### If I Were 5...

Think of the robot's brain as a tall building with 12 floors. Your question walks in at the ground floor and climbs up. On each floor, it picks up new ideas:
- **Bottom floors (0–3)**: "Oh, these are English words, and they're in a question format"
- **Middle floors (4–8)**: "Ah, this is a math problem about adding numbers"
- **Top floors (9–11)**: "I should answer with the number 42"

There's a hallway that runs through all the floors — we call it the **residual stream**. Everything the robot learns on each floor gets added to this hallway.

### The Real Deal

The **residual stream** is the central 768-dimensional vector that persists across all layers. Each layer reads from it (via attention and MLP) and writes back to it additively. This is the core of the transformer's computation:

```
x_0 = embed(tokens)
x_l = x_{l-1} + attn_l(x_{l-1}) + mlp_l(x_{l-1} + attn_l(x_{l-1}))
output = unembed(x_12)
```

We extract the residual stream at the **`hook_resid_pre`** hook point (before each block), at layers **2 (early)**, **6 (middle)**, and **10 (late)** to study how representations evolve through depth.

---

## 4. Superposition: The Core Problem

### If I Were 5...

Imagine you have a toy box with only 768 slots, but you need to store THOUSANDS of different toys — math toys, story toys, feelings toys, fact toys. What do you do? You get sneaky! You stack the teddy bear AND the puzzle piece in the same slot. Each slot holds a jumble of mixed-up toys. This is called **superposition** — the robot's brain has to squeeze way more ideas than it has slots for.

The problem? When you look at one slot, you can't tell what's in it — it's a messy mix of everything.

### The Real Deal

**Superposition** (Elhage et al., 2022) occurs when a neural network represents more features (semantic concepts) than it has dimensions. GPT-2 has d_model=768 but must encode thousands of distinct concepts. Features are stored as nearly-orthogonal directions in a high-dimensional space, making individual neurons **polysemantic** — each neuron responds to multiple unrelated concepts.

This is why naive per-neuron interpretation fails: neuron #500 might fire for "Python code", "the word 'function'", and "formal writing style" simultaneously. We need a tool to decompose these mixed signals into clean, individual concepts.

---

## 5. Sparse Autoencoders (SAEs)

### If I Were 5...

Now imagine you have a **magic sorting machine**! You dump in the jumbled mess from the toy box, and out comes 24,576 tiny labeled drawers. Each drawer has EXACTLY ONE toy in it! Drawer #20655 has the "math toy," drawer #1881 has the "code toy," and so on. Most drawers are empty (the machine only opens the ones that matter for your question). This magic sorter is the **Sparse Autoencoder**!

### The Real Deal

An SAE is trained to reconstruct the 768-dim residual stream through an overcomplete bottleneck:

```
encode(x) = ReLU(W_enc @ x + b_enc)    →  [24,576]-dim sparse vector
decode(z) = W_dec @ z + b_dec           →  [768]-dim reconstruction
```

**Key properties:**
- **Overcomplete**: 24,576 features >> 768 dimensions (32× expansion)
- **Sparse**: L1 penalty ensures only ~30–350 features are nonzero per input
- **Monosemantic**: each feature ideally corresponds to one interpretable concept
- **Pre-trained**: we use `gpt2-small-res-jb` (Joseph Bloom) — no SAE training needed

```python
sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre")
features = sae.encode(residual_stream)  # [24,576] sparse vector
```

---

## 6. Features: What SAEs Give Us

### If I Were 5...

Features are like individual LEGO bricks. A LEGO castle is made of hundreds of tiny pieces — a red brick, a blue brick, a window piece. Features work the same way! Feature #22431 means "the robot is thinking about something complex and emotional." Feature #7484 means "the robot is recalling a fact." When you ask a question, different bricks light up depending on what KIND of question it is.

### The Real Deal

Each of the 24,576 SAE features is a learned direction in the 768-dim residual stream space. The SAE's decoder matrix `W_dec` has shape [24,576, 768] — each row is a feature direction. When feature *i* fires (nonzero activation), it means the input's residual stream has a significant component along that direction.

Our top discriminative features (layer 6, selectivity = mean_target - mean_others):

| Category | Top Feature | Selectivity Score |
|----------|------------|-------------------|
| Reasoning | #22431 | 2.11 |
| Emotional | #22431 | 1.70 |
| Factual | #7484 | 1.56 |
| Code | #1881 | 1.17 |
| Creative | #13534 | 1.06 |
| Math | #20655 | 0.99 |

Notable: feature #22431 is shared between reasoning and emotional — it may encode a general "complex query requiring deliberation" concept.

---

## 7. Our Dataset: 300 Real Human Prompts

### If I Were 5...

We collected 300 questions that REAL people asked to AI chatbots! Not made-up questions — real ones from the internet. We sorted them into 6 types: math (50), code (50), facts (50), creative writing (50), emotional/feelings (50), and brain teasers/reasoning (50). Then we fed them all to our robot to see what happens inside.

### The Real Deal

Following instructor feedback to avoid synthetic prompts, we sourced real human queries from two HuggingFace datasets:

| Source | Size | Key Feature | Categories Filled |
|--------|------|-------------|-------------------|
| **Search Arena 24k** | 24,069 conversations | Expert-labeled `primary_intent` (Cohen's kappa 0.812) | factual, reasoning, creative |
| **WildChat 4.8M** | 2.75M messages | `topic` labels from real user-LLM chats | code, math, emotional |

**Quality filtering pipeline:**
- Length bounds: 3–200 words
- Alpha ratio > 0.5 (reject garbled text)
- Type-token ratio > 0.3 (reject repetitive text)
- GPT-2 token limit: ≤512 tokens
- Deduplication across sources

Final dataset: **300 prompts** (50 per category), producing **900 feature vectors** (300 × 3 layers).

---

## 8. Feature Extraction Pipeline

### If I Were 5...

Here's how we peek inside the robot's brain:
1. We type a question
2. The question gets turned into number-tokens
3. The tokens travel up through all 12 floors
4. At floors 2, 6, and 10, we grab the hallway data (residual stream)
5. We run it through the magic sorting machine (SAE)
6. We write down which drawers lit up!

We focus on the LAST token's data because it's heard everything before it — like the student who sat through the entire lecture.

### The Real Deal

```
Prompt → GPT-2 Tokenizer → Forward pass with cache
       → Extract residual stream at last token, layers [2, 6, 10]
       → SAE.encode() → Sparse feature vector [24,576]
```

**Why last token?** In causal (autoregressive) attention, each token can only attend to preceding tokens. The last token position accumulates context from the entire prompt via the attention mechanism, making it the richest single-position representation.

**Implementation:**
```python
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens)
resid = cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :]  # last token
features = sae.encode(resid.unsqueeze(0)).squeeze(0)        # [24,576]
```

Total extraction: 300 prompts × 3 layers = 900 feature vectors, each of dimension 24,576. Runtime: ~37 seconds on CUDA.

---

## 9. Sparsity Analysis

### If I Were 5...

Out of 24,576 drawers, only a small number actually light up for any question! But here's the cool part — different types of questions light up different AMOUNTS:

- **Code questions**: 344 drawers light up (like a big colorful party!)
- **Emotional questions**: 321 drawers (also lots of feelings to process!)
- **Creative writing**: 130 drawers
- **Reasoning**: 81 drawers
- **Math**: 34 drawers (just a few precise tools!)
- **Facts**: 26 drawers (the quietest — just recall one thing!)

By the top floor (layer 10), everyone settles down to about 160–205 drawers. The bottom floors are where the big differences show up.

### The Real Deal

**L0 sparsity** (count of nonzero SAE features) at layer 2:

| Category | Avg L0 | Interpretation |
|----------|--------|----------------|
| Code | 344.2 | Broad: syntax, semantics, library knowledge, logic |
| Emotional | 321.2 | Broad: empathy, social context, sentiment, advice |
| Creative | 129.9 | Moderate: imagery, style, narrative structure |
| Reasoning | 81.1 | Moderate: logic, conditionals, deduction |
| Math | 34.1 | Narrow: precise numerical/arithmetic features |
| Factual | 26.3 | Narrowest: specific knowledge retrieval |

**Layer progression**: By layer 10, all categories converge to 160–205 active features. This convergence suggests early layers perform categorical discrimination while later layers transform toward a uniform next-token prediction format.

---

## 10. Category Similarity and UMAP

### If I Were 5...

If you look at the light patterns for two math questions, they look VERY similar — almost the same drawers glow! But compare a math question to a poem question, and the patterns look totally different. It's like how two cats look alike, but a cat looks nothing like a spaceship.

We also made a magic map where every question becomes a dot, and similar questions cluster together. The math dots bunch up in one corner, the poem dots in another!

### The Real Deal

**Cosine similarity** of mean activation vectors between category pairs reveals structured relationships. At layer 6, reasoning and factual categories show highest inter-category similarity (~0.7+), while code and emotional are most distinct from other categories.

**UMAP** (Uniform Manifold Approximation and Projection) reduces the 24,576-dim feature space to 2D for visualization. With `n_neighbors=10, min_dist=0.1`, we observe clear categorical clustering at layer 6, confirming that SAE features capture semantically meaningful structure.

---

## 11. Separability: Fisher's Discriminant Ratio

### If I Were 5...

We asked: "At which floor of the building is it EASIEST to tell question types apart?" We measured this with a special score called the Fisher ratio — higher means easier to tell apart.

Surprise! Floor 2 (the bottom) was the best! The robot figures out "this is a math question" very early on, then mixes everything together as it goes up.

### The Real Deal

**Fisher's discriminant ratio** = between-class variance / within-class variance:

| Layer | Fisher Ratio | Interpretation |
|-------|-------------|----------------|
| 2 (early) | **0.0287** | Best separation — rapid input categorization |
| 6 (middle) | 0.0235 | Categories blending toward generation format |
| 10 (late) | 0.0259 | Slight recovery — output-specific differentiation |

The U-shaped pattern (best at 2, dip at 6, partial recovery at 10) suggests:
- **Early layers**: rapid categorical discrimination from surface features
- **Middle layers**: blending representations for contextual processing  
- **Late layers**: re-differentiating as the model prepares distinct output distributions

---

## 12. Discriminative Features

### If I Were 5...

Some drawers are SUPER picky — they only light up for one type of question. Drawer #22431 almost ONLY lights up for reasoning and emotional questions. Drawer #20655 almost ONLY lights up for math. We call these "discriminative features" because they help us discriminate (tell apart) the question types.

### The Real Deal

**Discriminative features** are identified by selectivity score: `mean(target_category) - mean(all_others)`. Top 5 per category at layer 6:

**Math**: 20655 (0.99), 3547 (0.78), 14494 (0.76), 14985 (0.72), 17308 (0.63)  
**Code**: 1881 (1.17), 22193 (1.09), 15530 (1.08), 13377 (0.93), 16657 (0.89)  
**Factual**: 7484 (1.56), 8545 (1.52), 19417 (0.92), 6222 (0.91), 12766 (0.82)  
**Creative**: 13534 (1.06), 708 (0.83), 21201 (0.79), 19384 (0.74), 16405 (0.72)  
**Emotional**: 22431 (1.70), 6581 (1.54), 1753 (1.07), 9815 (0.91), 19617 (0.89)  
**Reasoning**: 22431 (2.11), 8545 (1.63), 7484 (1.53), 6222 (1.46), 19820 (0.92)

**Notable overlaps:**
- Feature 22431 tops both emotional (1.70) and reasoning (2.11) — a "complex deliberation" detector
- Features 8545, 7484, 6222 appear in both factual and reasoning — knowledge retrieval overlaps with logical analysis
- No top-5 overlap between math/code and emotional/reasoning — clean separation of technical vs. evaluative processing

---

## 13. Activation Patching: Causal Verification

### If I Were 5...

This is the coolest part — we do BRAIN SURGERY on the robot! Here's how:
1. We ask the robot a math question: "The answer to 15 plus 27 is"
2. We ask a story question: "Once upon a time in a forest"
3. We SWAP a piece of the math brain into the story brain
4. Does the story suddenly turn into math? YES (kind of)!

When we swapped the ENTIRE floor, the story output changed a lot — it started looking math-like. But when we only swapped 10 specific drawers, barely anything changed. That means math knowledge is SPREAD everywhere, not in just a few special drawers — like butter on toast!

### The Real Deal

**Activation patching** is a causal intervention technique. We run clean (math) and corrupt (creative) prompts, cache residual streams, and swap activations at specific layers.

**Experiment setup:**
- Clean prompt: `"The answer to 15 plus 27 is"`
- Corrupt prompt: `"Once upon a time in a forest"`
- Metric: KL divergence between patched output and clean output (lower = more similar to math)

**Results — Layer sweep:**

| Layer Patched | KL Divergence | Interpretation |
|---------------|---------------|----------------|
| 0 | 5.08 | Minimal effect — too early |
| 2 | 2.42 | Some shift |
| 6 | 0.64 | Strong shift |
| 8 | 0.16 | Very strong |
| 10 | 0.007 | Nearly fully restored |
| 11 | 0.000 | **Fully restores clean output** |

KL decreases **monotonically** from layer 0 to 11 — later layers carry progressively more output-relevant causal information.

**Whole-residual vs. Surgical patching (layer 6):**

| Method | KL | Effect |
|--------|-----|--------|
| No patch (baseline) | 3.61 | — |
| Full residual patch | 0.64 | Massive shift (82% reduction) |
| Surgical patch (top-10 math features) | 3.48 | Tiny shift (3.6% reduction) |

The top-10 math features account for only **~4.5%** of the total causal effect. Conclusion: math-relevant information is **distributed** across hundreds of features in superposition, not concentrated in a few "math neurons." This directly confirms the superposition hypothesis in practice.

---

## 14. Feature Steering

### If I Were 5...

What if you could push the robot's thoughts? Like, the robot is thinking about something normal, but you sneak in and turn the "math dial" up to 20! Will the robot start talking about math?

We tried it! We turned up the 10 strongest math drawers by 20× while the robot was answering a normal question. The output changed a bit — but didn't become clearly math-like. The math thoughts are too spread out to push with just 10 drawers. You'd need hundreds!

### The Real Deal

**Feature steering** amplifies SAE feature directions during generation:

```python
steering_vector = sae.W_dec[math_feature_indices].mean(dim=0)  # mean of top-10 math directions
# During forward pass: residual += alpha * steering_vector (alpha=20)
```

**Setup:**
- Neutral prompt: `"Tell me something interesting about the world."`
- Steering: top-10 math feature directions, alpha=20, applied at layer 6

**Result:** Output was measurably altered but not clearly math-like. This is consistent with the patching finding — math behavior requires coordinated activation across many more features than the top 10. More aggressive steering (100+ features, multi-layer intervention) would be needed for stronger behavioral shifts.

---

## 15. Key Findings Summary

### If I Were 5...

Here's what we learned peeking inside the robot's brain:

1. **Different questions DO light up different drawers!** Math lights up different ones than stories.
2. **The robot figures out what kind of question it is on the BOTTOM floors!** Floor 2 is where it happens.
3. **Code and emotional questions light up LOTS of drawers (344 and 321!), facts and math light up very few (26 and 34).** Coding and feelings need lots of different ideas at once!
4. **Math knowledge is spread EVERYWHERE** like butter on toast — you can't find it in just a few drawers.
5. **Some drawers are shared!** Drawer #22431 lights up for BOTH reasoning AND emotional questions — maybe it's the "thinking hard" drawer.
6. **The TOP floors make the final decisions** about what to say — floor 11 is the boss.

### The Real Deal

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Category-specific features exist | Selectivity scores up to 2.11; UMAP clustering | LLMs develop specialized internal circuits for different task types |
| Early layers discriminate most | Fisher ratio: L2=0.0287 > L6=0.0235 | Rapid input categorization, then convergence toward generation |
| Sparsity varies dramatically | Code L0=344 vs Factual L0=26 at L2 | Task complexity maps to representational breadth |
| Math info is distributed | Surgical patch captures only 4.5% of effect | Superposition hypothesis confirmed in practice |
| Meaningful feature sharing | #22431 shared (reasoning+emotional) | Higher-order features span related capabilities |
| Late layers are causally dominant | KL→0 at layer 11 | Output decisions made in final layers |

**What this means for mechanistic interpretability:**
SAE features provide a useful but incomplete lens into model internals. Features are genuinely interpretable and category-specific, but the causal structure is highly distributed — no small set of features fully captures any capability. This is exactly what the superposition hypothesis predicts and represents a key challenge for interpretability-based model control.

---

## Technical Stack Reference

| Component | Specification |
|-----------|---------------|
| Model | GPT-2 Small (124M params, 12 layers, d=768) |
| SAE | gpt2-small-res-jb (24,576 features/layer) |
| Hook point | `hook_resid_pre` (residual stream before each block) |
| Target layers | 2, 6, 10 |
| Dataset | 300 prompts (50/category × 6 categories) |
| Sources | Search Arena 24k + WildChat 4.8M |
| Extraction | Last-token residual → SAE.encode() |
| Hardware | RTX 3050 Ti (4.3GB VRAM), CUDA |
| Libraries | transformer_lens, sae_lens, torch, numpy, matplotlib, umap-learn, sklearn |

---

## Notebook Pipeline

```
00_data_preparation.ipynb    → Load datasets, EDA, quality filter, save 300 prompts
01_setup_and_model.ipynb     → Install deps, load GPT-2 + SAE, save config
02_feature_extraction.ipynb  → Extract 900 feature vectors (300 prompts × 3 layers)
03_prompt_analysis.ipynb     → Similarity, UMAP, sparsity, separability, discriminative features
04_activation_patching.ipynb → Causal verification: layer sweep, surgical patching, steering
05_final_report.ipynb        → Compile all results with visualizations and reflection
```

---

*Generated for the CS 463 Final Project — Mechanistic Interpretability with Sparse Autoencoders*
