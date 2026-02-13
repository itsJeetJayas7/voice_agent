# Recovered vLLM Configuration Backup (2026-02-13)

## Source evidence used
- `/root/start_vllm.sh`
- `/root/vllm_server.log` (successful run on 2026-02-12)
- `/root/vllm_startup.log` (failed run on 2026-02-13 when compile cache behavior diverged)
- `/root/RUNBOOK.md`
- `/root/.config/vllm/usage_stats.json`

## Known-good launch command (recovered)
```bash
source /root/voxtral-env/bin/activate
export VLLM_DISABLE_COMPILE_CACHE=1
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768
```

## Recovered runtime config details
- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- Host/Port: `0.0.0.0:8000`
- API mode: OpenAI-compatible server with realtime route `/v1/realtime`
- Dtype: `torch.bfloat16` (downcast from float32)
- Quantization: `None`
- Device: `cuda`
- Tensor parallel size: `1`
- Pipeline parallel size: `1`
- Data parallel size: `1`
- Max model length: `32768`
- Compilation mode: PIECEWISE cudagraphs
- Compile cache env override: `VLLM_DISABLE_COMPILE_CACHE=1`
- Chunked prefill: enabled (`max_num_batched_tokens=2048`)
- Prefix caching: enabled
- KV cache dtype: `auto`

## Why `VLLM_DISABLE_COMPILE_CACHE=1` is required (inference)
- Successful run (`/root/vllm_server.log`) explicitly reports compile cache disabled and reaches `Application startup complete`.
- Failed run (`/root/vllm_startup.log`) fails with `RuntimeError: The compiled artifact is not serializable` and points to disabling compile cache as mitigation.
- Therefore, prior known-good startup is inferred to require `VLLM_DISABLE_COMPILE_CACHE=1`.

## Observed successful API readiness signals
- `Route: /v1/realtime, Endpoint: realtime_endpoint`
- `Application startup complete`
- `GET /health ... 200 OK`
