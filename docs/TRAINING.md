# Training Contract

The algorithm is in `train_scopd.py`. The shared trainer under
`scopd/visionzip_aokvqa/` handles checkpointing, optimization, DDP and EMA updates.

For every example, generate one greedy response at retention b=0.10. All scoring
branches condition on exactly that response. The EMA teacher receives the full
image and the same question, without a reference answer. The forward objective is
`KL(q_full_EMA || p_b)`. Only generated response positions participate, including
EOS when generated. Prompt, image and padding positions do not participate.

SCOPD averages this loss over all generated positions. SCOPD+ evaluates one
no-gradient student probe at b+0.01 on the same prefix and computes:

```text
m_t = (p_b,t + p_plus,t) / 2
B_t = 0.5 * KL(p_b,t || m_t) + 0.5 * KL(p_plus,t || m_t)
selected = top ceil(0.10 * response_length) positions by B_t
loss_i = mean(KL(q_full_EMA,t || p_b,t) for t in selected)
batch_loss = mean_i(loss_i)
```

JSD is full-vocabulary, FP32 and stop-gradient. Ties favor earlier positions.
There is no KL floor, projection fraction, batch competition or auxiliary JSD
loss. The top fraction is a fraction of response tokens, not image tokens.
Native VisionZip selection and merging are recomputed independently at each
budget; masks need not be nested.

Only 392 LLM LoRA tensors (40,370,176 parameters for Qwen2.5-VL-7B) are trainable.
The vision encoder/projector remain frozen. Dropout is zero. AdamW uses constant
2e-5 learning rate and zero weight decay; EMA decay is 0.9999 with lazy
initialization after the first optimizer update.

The ordered LLaVA-CoT training JSONL must have `sample_id`, `image`, `question`
and an answer/target field for the shared data schema. That answer field is NOT
provided to the teacher or used as the SCOPD/SCOPD+ target. Supply images separately.
Do not re-sort or re-shuffle the file. Neither benchmark examples nor judge output
are inputs to training. Dataset decontamination is not performed by this entry;
retain and verify the dataset's own manifests.

Image budget defaults to 1280 nominal visual tokens: both pixel bounds are
1003520. Response cap is 1024, independent of image budget. Native smart-resize
rounding determines the actual resized grid.

The default 10240 examples mean 320 optimizer updates at batch32. One GPU uses
accumulation32; four GPUs use accumulation8 per rank. Resume requires the same
world size, data hash and configuration. It restores adapter FP32 bits after
PEFT construction and checks optimizer, EMA, cursor and RNG. A changed image or
response budget is not silently accepted.
