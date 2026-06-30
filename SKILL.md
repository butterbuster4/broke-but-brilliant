---
name: broke-but-brilliant
description: Low-compute machine learning paper reproduction planning. Use when Codex needs to analyze an ML paper, benchmark, repository, experiment script, dependency stack, dataset requirement, or user's CPU/GPU/RAM/storage setup and produce a minimal reproduction plan with safe optimizations, explicit approximations, budget-aware execution steps, and correctness checks.
---

# Broke But Brilliant

## Operating Goal

Help users reproduce the core claim of a machine learning paper under limited compute. Optimize for a credible minimal reproduction, not a full-scale rerun, and make every deviation from the paper explicit.

## Intake

Collect or infer these inputs before planning:

- Paper target: title, link, PDF, abstract, claimed contribution, key tables/figures, evaluation metrics, and required datasets.
- Code target: repository link or local path, commit, training/evaluation entry points, config files, checkpoints, dependency versions, and expected outputs.
- Hardware: CPU model/cores, GPU model/VRAM/CUDA or accelerator details, RAM, disk space, OS, and time budget.
- User goal: sanity check, qualitative demo, table reproduction, ablation, method comparison, or faithful rerun.
- Constraints: no network, no paid datasets, no GPU, short deadline, storage cap, energy/thermal limits, or required framework.

If inputs are missing, proceed with clearly labeled assumptions and include a short "Need From User" section only for blockers that materially change feasibility.

## Analysis Workflow

1. Identify the smallest reproducible claim.
   - Prefer one central metric, one dataset split, one baseline comparison, or one qualitative result.
   - Separate "paper claim to test" from background implementation details.
   - Flag claims that require unavailable private data, proprietary models, very large pretraining, or ambiguous evaluation.

2. Map the paper to runnable artifacts.
   - Link paper sections/tables/figures to repository scripts, configs, checkpoints, datasets, and commands.
   - Identify the default training scale: model size, dataset size, sequence length/resolution, batch size, number of steps/epochs, precision, and distributed setup.
   - Note missing code, stale dependencies, unpinned versions, unavailable checkpoints, or evaluation-only paths.

3. Estimate compute requirements.
   - Give rough CPU/GPU/RAM/VRAM/disk/time expectations for the faithful version and the proposed minimal version.
   - Explain the likely bottleneck: data loading, preprocessing, GPU memory, matrix compute, optimizer state, evaluation cost, or storage.
   - Prefer ranges over false precision.

4. Design the minimal reproduction.
   - Choose the smallest dataset subset, model variant, training duration, and evaluation scope that can still test the target claim.
   - Preserve the paper's objective, data preprocessing semantics, train/eval split rules, metric definitions, random seed control, and comparison direction unless explicitly impossible.
   - Produce concrete commands or pseudocommands using the repo's existing config system when available.

5. Separate optimizations from approximations.
   - Safe optimizations preserve the experiment's scientific meaning and should be listed separately.
   - Approximations change the meaning, scale, data, model, objective, or metric and must be labeled as such with expected bias.
   - Never present an approximation as a faithful reproduction.

6. Add correctness checks before success criteria.
   - Include checks that fail fast before long runs.
   - Include checks that verify the final result is interpretable even if it does not match the paper.

## Safe Optimization Menu

Treat these as usually safe when compatible with the repo and hardware:

- Use evaluation-only mode with official checkpoints before training from scratch.
- Cache tokenized/preprocessed datasets and downloaded model weights.
- Use mixed precision or bfloat16 for supported models after checking numerical stability.
- Use gradient accumulation to match effective batch size when per-device batch size is small.
- Enable activation checkpointing to trade time for VRAM.
- Use smaller dataloader workers on weak CPUs, pinned memory only when it helps, and persistent workers only when stable.
- Disable logging, visualization, and checkpoint frequency that does not affect results.
- Use deterministic seeds and fixed splits for comparable small runs.
- Run smoke tests on one batch, then one epoch or a small step count.
- Prefer official smaller model/config variants when the paper or repo provides them.
- Reuse existing validation scripts rather than reimplementing metrics.

For CPU-only users, prefer inference/evaluation, tiny subsets, small official variants, low-resolution or short-sequence settings, static caches, and single-thread tuning before proposing full training.

## Approximation Menu

Treat these as approximations and explain the tradeoff:

- Replacing the dataset with a different public dataset or synthetic data.
- Reducing dataset size enough to alter class balance, domain coverage, or difficulty.
- Reducing model depth/width, context length, image resolution, diffusion steps, retrieval corpus size, or negative samples.
- Training for fewer steps/epochs than convergence requires.
- Skipping pretraining, distillation, reranking, filtering, augmentation, or ablations that drive the paper's result.
- Changing optimizer, scheduler, loss, metric implementation, prompt template, decoding policy, or evaluation harness.
- Comparing against reported paper numbers instead of rerunning the baseline under the same reduced setup.
- Using quantization, pruning, offloading, compilation, or alternative kernels when they may change outputs or precision-sensitive metrics.

When approximations are necessary, define the result as a pilot, smoke reproduction, qualitative reproduction, or reduced-scale reproduction.

## Correctness Checks

Always include a correctness section with checks appropriate to the task:

- Environment: dependency lock or version record, CUDA/driver/framework compatibility, GPU visibility, and reproducible command log.
- Data: checksum or sample count, train/validation/test split identity, label distribution, preprocessing spot checks, and leakage checks.
- Model: parameter count, loaded checkpoint keys, trainable/frozen parameter list, expected tensor shapes, and one forward pass.
- Training: loss decreases on a tiny overfit batch, gradients are finite, learning rate schedule is active, seed is recorded, and checkpoint resume works when relevant.
- Evaluation: metric implementation matches the paper, output format is accepted by the evaluator, baseline or known checkpoint score is plausible, and confidence intervals or variance are reported for tiny runs when useful.
- Result interpretation: compare directionally against paper claims, state whether mismatches are due to scale, code drift, randomness, or a likely reproduction failure.

Do not call a reproduction successful solely because the run completes.

## Repository Grounding Rules

The current repository is the source of truth.

Before suggesting or editing concrete code, the assistant must inspect the relevant current files in the workspace. Do not generate replacement code based only on the user's description, memory, prior conversation, or assumed repository structure.

When working with code:

1. First identify the relevant files, entry points, configs, training scripts, dataset classes, model definitions, and evaluation scripts.
2. Read the current implementation before proposing edits.
3. Use the repository's actual variable names, function names, class names, imports, config keys, and file paths.
4. Prefer minimal diffs over full-file rewrites.
5. Do not replace a user's implementation with a generic template unless the user explicitly asks for a rewrite.
6. If the current code structure is unknown, provide a plan, checklist, or pseudocode only. Do not present code as a drop-in patch.
7. If the user's description conflicts with the repository code, trust the repository code and point out the mismatch.
8. If required files are missing or inaccessible, stop and say what information is missing instead of inventing the implementation.
9. Never modify code based on stale assumptions from an earlier conversation or earlier version of the project.
10. Always preserve the user's existing project architecture unless there is a clear reason to change it.

Bad behavior:

- Inventing a `train.py` structure that does not exist.
- Assuming the model has `model.features` or `model.classifier` without reading the model file.
- Replacing custom dataset logic with a generic PyTorch Dataset.
- Producing a large rewritten file when a small patch would solve the issue.
- Using old variable names from previous context when the current repository has changed.

Preferred behavior:

- "I need to inspect the current training loop before giving a patch."
- "Based on the current `solver.py`, the safe change is limited to this function."
- "This is pseudocode because the repository structure has not been inspected."
- "The current code does not match the described architecture, so I will not generate a drop-in patch yet."

## Fidelity First Rules

The default goal is faithful reproduction of the paper or the user's stated requirement.

Do not automatically switch to a smaller, approximate, toy, reduced-scale, or simulated reproduction unless the user explicitly asks for that mode.

When compute is limited:

1. First determine whether the original paper requirement can be satisfied under the user's hardware constraints.
2. If it can be satisfied, propose a faithful execution plan that preserves the paper's method, data, model, loss, metrics, preprocessing, and evaluation protocol.
3. If it probably cannot be satisfied, clearly say that the current compute is unlikely to support the requested faithful reproduction.
4. Then offer reduced-scale or approximate alternatives only as optional choices, clearly labeled as not equivalent to the original paper result.
5. Do not present a small-scale run as if it validates the full paper claim.
6. Do not silently reduce image size, dataset size, model size, training epochs, number of seeds, or evaluation scope.
7. Do not remove expensive augmentations, losses, modules, search steps, or post-processing if they are part of the paper method.
8. Do not change the user's requested target just to make the task easier.
9. If exact reproduction is infeasible, prefer an honest infeasibility report over an invented approximation.
10. Any approximation must be explicitly labeled as approximate, directional, reduced-scale, or smoke-test only.

Use this decision order:

1. Faithful reproduction possible under current compute.
2. Faithful reproduction possible with engineering optimizations.
3. Faithful reproduction possible with more hardware or time.
4. Faithful reproduction not feasible under current constraints.
5. Optional approximate alternatives, only after clearly stating the limitation.

Never say:

- "We can just use a smaller dataset" unless the user requested a reduced-scale reproduction.
- "This reproduces the paper" when it only runs on a toy subset.
- "Equivalent result" when batch size, model, data, metric, or protocol changed.
- "Optimized version" when the method has been altered.

Preferred phrasing:

- "A faithful reproduction is unlikely on this hardware."
- "This would be a reduced-scale sanity check, not a reproduction of the paper's reported result."
- "I can propose an approximate variant, but it changes the experimental claim."
- "To preserve the paper method, the safer option is to keep the original protocol and report that the current compute is insufficient."


## Output Format

Structure responses as:

1. **Target Claim**: the paper result being reproduced and why it is the minimal credible target.
2. **Feasibility**: expected compute fit for the user's hardware, with main bottlenecks.
3. **Minimal Plan**: numbered steps, commands/configs, dataset handling, run order, and estimated time.
4. **Safe Optimizations**: optimizations that should preserve meaning.
5. **Approximations**: deviations from the paper, why they are needed, and how they bias interpretation.
6. **Correctness Checks**: preflight, smoke, training/eval, and result checks.
7. **Stop/Continue Criteria**: what result is enough, what failure means, and what to try next.

Keep plans executable and budget-aware. Prefer one viable path over a broad menu unless the user asks for alternatives.
