# Broke but Brilliant Test Examples

This file records three example prompts used to test the `broke-but-brilliant` skill and the expected style of response.

## Test 1: Laptop CPU Reproduction Plan

### User Prompt

```text
[$broke-but-brilliant](C:\Users\Administrator\.codex\skills\broke-but-brilliant\SKILL.md) I want to reproduce a machine learning paper, but I only have a laptop CPU and 16GB RAM. The paper trains a CNN on a large image dataset. Help me make a low-compute reproduction plan.
```

### Expected Result

The response should frame the task as a reduced-scale directional reproduction, not a faithful full rerun.

Key points to include:

- Target a minimal claim such as showing that the paper's CNN/config improves validation accuracy or loss over a baseline on a fixed small subset.
- Explain that full training is likely infeasible on laptop CPU and 16GB RAM.
- Recommend a small stratified subset, for example `1k-10k` images.
- Recommend lower but still defensible image sizes such as `128x128` or `160x160`, unless the model requires `224x224`.
- Use small batches such as `8-32`.
- Run in stages: smoke run, tiny overfit run, then reduced reproduction.
- Compare against one baseline under the same reduced setup.
- Separate safe optimizations from approximations.
- Include correctness checks for environment, data, model, training, evaluation, and result interpretation.

Example output structure:

```text
Target Claim
Feasibility
Minimal Plan
Safe Optimizations
Approximations
Correctness Checks
Stop/Continue Criteria
```

Important interpretation:

- Do not call the result a faithful paper reproduction.
- Call it a reduced-scale reproduction, pilot reproduction, or directional reproduction.

## Test 2: RTX 3060 12GB Out-of-Memory Adaptation

### User Prompt

```text
[$broke-but-brilliant](C:\Users\Administrator\.codex\skills\broke-but-brilliant\SKILL.md) I have an RTX 3060 12GB. The paper code runs out of memory during training. It uses PyTorch, batch size 64, image size 512. Help me adapt it without changing the paper method.
```

### Expected Result

The response should preserve the paper method and recommend execution-level memory optimizations first.

Key points to include:

- Keep the same CNN architecture, loss, optimizer, augmentation, image size `512`, metric, and effective batch size `64`.
- Explain that batch size 64 at 512px is likely too large for 12GB VRAM.
- First try AMP mixed precision.
- Then reduce per-device microbatch and use gradient accumulation:
  - `batch_size=16, grad_accum=4`
  - `batch_size=8, grad_accum=8`
  - `batch_size=4, grad_accum=16`
- Ensure loss is divided by `grad_accum_steps`.
- Step the optimizer and scheduler only at optimizer-update boundaries when the original schedule expects that.
- Add activation checkpointing if still out of memory.
- Use `channels_last`, `zero_grad(set_to_none=True)`, and reduced logging/checkpoint frequency.
- Warn about BatchNorm: gradient accumulation preserves effective optimizer batch size, but BatchNorm still sees the microbatch.

Safe optimizations:

- AMP mixed precision if stable.
- Gradient accumulation preserving effective batch size.
- Activation checkpointing.
- `channels_last`.
- `optimizer.zero_grad(set_to_none=True)`.
- Less frequent logging and checkpointing.

Approximations to avoid first:

- Reducing image size.
- Changing model width/depth.
- Removing augmentations.
- Replacing BatchNorm.
- Changing optimizer.
- Using 8-bit optimizer states.

Correctness checks:

- Run one forward/backward pass at increasing batch sizes.
- Print peak GPU memory.
- Verify `microbatch * grad_accum == 64`.
- Check AMP gradients are finite.
- Overfit `16-32` images and confirm loss drops.
- Record microbatch, accumulation, AMP, image size, and peak VRAM.

## Test 3: AutoDL RTX 4090 Low Utilization Dataloader Bottleneck

### User Prompt

```text
[$broke-but-brilliant](C:\Users\Administrator\.codex\skills\broke-but-brilliant\SKILL.md) I am using AutoDL with RTX 4090, but GPU utilization is low and training is slow. The dataloader may be the bottleneck. Help me diagnose and optimize.
```

### Expected Result

The response should diagnose whether the input pipeline is starving the GPU and recommend optimizations that preserve the experiment.

Key points to include:

- Keep paper method unchanged: same data, preprocessing semantics, augmentations, model, optimizer, and metrics.
- Identify likely bottlenecks:
  - Slow disk or network-mounted dataset.
  - Too few or too many `DataLoader` workers.
  - Expensive CPU transforms or augmentations.
  - Small batch size.
  - Blocking host-to-GPU copies.
  - Per-sample Python overhead, image decode, JSON parsing, or repeated file discovery.
- Measure data wait time versus GPU compute time with `torch.cuda.synchronize()`.
- Watch GPU metrics with `nvidia-smi dmon -s pucm`.
- Watch CPU and disk with tools like `htop` and `iostat -xz 1`.
- Benchmark the dataloader alone.
- If AutoDL storage is slow, copy the dataset to local fast storage such as an `autodl-tmp` path when available.

Safe optimizations:

- Tune `DataLoader` with sweeps:
  - `num_workers`: `4, 8, 12, 16`
  - `prefetch_factor`: `2, 4, 8`
- Use `pin_memory=True`.
- Use `persistent_workers=True`.
- Use `non_blocking=True` for GPU transfers.
- Cache decoded/resized data only when preprocessing semantics remain unchanged.
- Avoid recomputing file lists or parsed annotations each epoch.
- Use faster image decode/transforms if output semantics match.
- Increase batch size only when it does not change the paper's intended optimization setup.

Approximations to avoid first:

- Reducing image size.
- Removing expensive augmentations.
- Changing interpolation or crop policy.
- Changing effective batch size when optimizer schedule depends on it.
- Precomputing random augmentations that should vary each epoch.
- Changing dataset format in a way that changes sampling order or class balance.

Correctness checks:

- Confirm train/validation sample counts are identical.
- Compare transformed samples before and after optimization.
- Confirm labels are unchanged.
- Confirm random augmentations still vary across epochs.
- Run 100 batches and verify finite loss.
- Compare a short loss curve against the original pipeline.
- Record images/sec, batch wait time, GPU utilization, peak VRAM, worker count, prefetch factor, and storage path.

Stop when:

- GPU utilization is high during training steps.
- Batch wait time is much smaller than compute time.
- Images/sec improves without changing samples, labels, transforms, or metrics.
- A short training run gives a similar loss curve to the original pipeline.
