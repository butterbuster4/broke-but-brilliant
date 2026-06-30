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
