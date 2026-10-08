# fake-sglang

A CPU-only **fake SGLang engine** that simulates prefill, decode, and KV-cache
transfer, so you can develop and test a distributed inference framework on
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) + SGLang without GPUs.

It works by shipping an importable `sglang` package that implements the subset
of SGLang's Python API that Dynamo's SGLang backend consumes. **Dynamo runs
completely unmodified** — `python -m dynamo.sglang` loads this instead of real
SGLang and never knows the difference.

> This is **not** SGLang and does no inference. Output tokens are deterministic
> noise; latencies come from a model, not a forward pass. Every worker prints a
> `SIMULATED engine` banner at startup for exactly this reason.

## Why not something that already exists

| Existing | What it gives | Why it wasn't enough |
|---|---|---|
| `python -m dynamo.mocker` | Dynamo's Rust scheduler simulator: KV blocks, chunked prefill, preemption, bandwidth-modelled transfer | Plugs in as `EngineType.Mocker`. The `dynamo.sglang` component never loads, so none of the SGLang integration is exercised |
| `dynamo.common.backend.sample_engine` | A CPU fake engine on Dynamo's Unified Backend SDK | Same: it bypasses `dynamo.sglang` entirely |
| `sglang/tools/sglang-simulator` | Real SGLang scheduler with a mocked `ModelRunner.forward`, GPU-free in CI | **No PD-disaggregation support at all**, and it still needs the full CUDA-only `sglang` + `torch` install |

## What it simulates

- **Prefill** — chunked against a token budget, cost proportional to tokens
  actually recomputed (prefix-cache hits and transferred KV are free).
- **Decode** — one shared step per batch whose cost grows with batch size, so
  queueing shows up as ITL growth the way it does on a GPU.
- **KV cache** — a finite pool of fixed-size blocks, cumulative-hash prefix
  reuse, and LIFO preemption under pressure.
- **KV transfer** — a **real** cross-process HTTP bootstrap handshake between
  the prefill and decode workers, with the payload cost simulated as
  `tokens x kv_bytes_per_token / bandwidth`. `kv_bytes_per_token` is derived
  from the model's real `config.json`.
- **KV events** — block stores and evictions published over ZMQ in SGLang's
  exact wire format, so Dynamo's KV-aware router builds a real prefix tree and
  routes on cache hits.
- **Scheduler metrics** — pushed over ZMQ to the socket Dynamo binds, so
  load-based routing sees real occupancy.

Aggregated and disaggregated modes produce **byte-identical output** for the
same prompt, and both are reproducible run to run.

## Setup

Clone this repo next to a Dynamo checkout — the bootstrap script expects
`../dynamo` and creates the shared virtualenv at `../.venv`:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git clone <this repo> fake-sglang
```

Then one command, ~5 minutes on a clean machine:

```bash
cd fake-sglang
./scripts/bootstrap_macos.sh
```

It installs cmake/protobuf, Rust, `uv`, creates `../.venv` (Python 3.10),
builds Dynamo's PyO3 bindings from source (there is no macOS wheel), installs
`ai-dynamo` and this package, and verifies the result.

**Safe to re-run.** Every step checks before it acts, so a second run reuses
the existing toolchain and virtualenv and reinstalls nothing already
satisfied (~19s, almost all of it the incremental Rust rebuild). The binding
build is repeated on purpose — it is the only way a change under `dynamo/lib`
reaches the wheel, and cargo's cache keeps it at ~15s rather than ~2.5min.

| Override | Effect |
|---|---|
| `SKIP_BINDINGS=1` | Skip the Rust rebuild when `dynamo._core` already imports (~2s total) |
| `FORCE_RECREATE=1` | Delete and rebuild the virtualenv from scratch |
| `PY_VERSION=3.12` | Use a different Python (see the `asyncio.timeout` note under *Known gaps*) |

If `../.venv` exists but is a different Python than expected, the script stops
with an explicit message rather than half-provisioning against it.

Model metadata ships in the repo: `models/qwen3-0.6b/` holds Qwen3-0.6B's
**config + tokenizer only, no weights** (~15 MB instead of ~1.5 GB), pinned to
a Hugging Face revision so every checkout is byte-identical (see
`models/qwen3-0.6b/SOURCE.md`). The launch scripts use it by default. Pointing
`--model-path` at a local directory also makes Dynamo skip its own model
download entirely, so runs are offline and instant.

To use a different model, or to move to a newer Qwen3 revision, fetch its
metadata and point `MODEL` at it:

```bash
source ../.venv/bin/activate
python scripts/fetch_model_metadata.py <hf-repo-id> models/<name> [REVISION]
MODEL=$PWD/models/<name> ./launch/agg.sh
```

`fetch_model_metadata.py` needs `huggingface_hub` and network access. When
bumping the Qwen3 revision, update `QWEN3_0_6B_REVISION` in the script and
`SOURCE.md` alongside the regenerated files.

## Run

No etcd, no NATS, no Docker — the launch scripts use Dynamo's file-based
discovery backend (`DYN_DISCOVERY_BACKEND=file`) and a TCP request plane.

```bash
./launch/agg.sh            # frontend :8000 + one aggregated worker
./launch/disagg.sh         # frontend :8000 + prefill worker + decode worker
./launch/agg_kv_router.sh  # frontend :8000 + 2 workers behind the KV-aware router
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake-model","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

Any flag is forwarded to the workers, so you can dial the simulation:

```bash
./launch/disagg.sh --fake-kv-bandwidth-gb-s 1 --fake-itl-ms 25
```

## Clusters and network topology

`launch/cluster.sh` runs any mix of workers from a YAML file: how many prefill,
decode and aggregated workers, where each group sits in the network, and what
a KV transfer costs over each kind of link.

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --dry-run   # print the plan only
./launch/cluster.sh launch/clusters/two-zones.yaml             # run it
./launch/cluster.sh launch/clusters/mixed.yaml -- --fake-itl-ms 20   # args after -- go to every worker
```

```yaml
frontend: {router_mode: kv}          # kv also turns on per-worker KV events
topology:
  levels: [zone, rack, node]         # a location has one component per level
  links:                             # named by the deepest level both ends share
    node:    {bandwidth_gb_s: 450,  latency_ms: 0.01}   # same node
    rack:    {bandwidth_gb_s: 50,   latency_ms: 0.05}   # same rack, other node
    zone:    {bandwidth_gb_s: 12.5, latency_ms: 0.5}    # same zone, other rack
    default: {bandwidth_gb_s: 1.25, latency_ms: 5}      # nothing shared
workers:
  - {role: prefill, count: 2, location: zone-a/rack-1/node-1}
  - {role: decode,  count: 4, location: zone-b/rack-1/node-1, args: [--fake-itl-ms, 12]}
  - {role: agg,     count: 1, location: zone-a/rack-2/node-1}
```

The prefill worker publishes its location with each KV room. The decode worker
looks up the link between the two locations and prices the transfer as
`latency_ms + bytes / bandwidth_gb_s`. It logs every handoff at INFO, so you
can count how far KV traveled in a run:

```
decode-2 | fake-sglang: KV room 5439... zone-a/rack-1/node-1 -> zone-b/rack-1/node-1 over 'default' link (1.25 GB/s): 2011 tokens, 219.95 MiB, 189.51 ms
```

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --log-dir runs/1
grep -h "KV room" runs/1/decode-*.log | grep -o "over '[a-z]*'" | sort | uniq -c
```

At startup the launcher prints the placement and the link that each
prefill → decode pair would use. Ports are assigned automatically. Use
`--port-offset N` to run a cluster beside another deployment. Each run gets
its own discovery registry (`DYN_FILE_KV`), so concurrent runs never discover
each other's workers.

**What Dynamo does with this today: nothing.** In the current Dynamo,
`PrefillRouter` picks the prefill worker by KV overlap plus load and the
decode worker by load alone. Nothing links the two choices by distance.
Measured on `two-zones.yaml` with 20 requests: 11 handoffs crossed zones,
6 crossed racks within a zone, and 3 stayed on one node. The topology is
there so you can measure that cost, and measure any topology-aware pairing
you add to Dynamo against it.

Aggregated workers serving the same model form their own worker set. The
frontend splits requests between that set and the prefill → decode set
instead of adding the aggregated workers to the decode pool.

## Simulation knobs

| Flag | Default | Effect |
|---|---|---|
| `--fake-prefill-ms-per-1k-tokens` | 55.0 | Prefill cost per 1k recomputed tokens |
| `--fake-prefill-overhead-ms` | 4.0 | Fixed per-prefill-step cost |
| `--fake-itl-ms` | 9.0 | Base inter-token latency |
| `--fake-itl-ms-per-running-req` | 0.35 | ITL added per extra request in the batch |
| `--fake-kv-bandwidth-gb-s` | 64.0 | Simulated KV-transfer bandwidth |
| `--fake-kv-transfer-overhead-ms` | 1.0 | Fixed per-transfer cost |
| `--fake-speedup-ratio` | 1.0 | Divides every simulated sleep (time compression) |
| `--fake-num-kv-blocks` | 8192 | KV pool size; lower it to reach preemption |
| `--fake-bootstrap-timeout-s` | 30.0 | How long decode waits for prefill's KV |
| `--fake-location` | unset | This worker's place in the network, e.g. `zone-a/rack-1/node-2` |
| `--fake-topology` | unset | JSON link table; with locations on both ends it replaces the two `--fake-kv-*` knobs per transfer |

KV event publishing is enabled with SGLang's own flag, not a `--fake-` one:

```bash
--page-size 16 --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:25557"}'
```

Dynamo derives `use_kv_events` from that same config, so one flag turns on both
the engine's publishing and Dynamo's subscription. Give each worker its own
port.

## How disaggregation flows

Dynamo's Rust `PrefillRouter` orchestrates this; the two workers only do the
KV handoff.

```
frontend --> PrefillRouter --> prefill worker   (max_tokens forced to 1)
                                    |
                    yields disaggregated_params {bootstrap_host, port, room}
                    and publishes the room on its bootstrap HTTP server
                                    |
             PrefillRouter stamps it onto the decode leg as bootstrap_info
                                    v
                              decode worker
                    GET /room/{id}  -> Bootstrapping   (real HTTP wait)
                    sleep(payload)  -> Transferring    (simulated bytes)
                                    -> Success
                    decodes from a "prebuilt" batch, leading with the
                    token prefill actually produced
```

`bootstrap_host == "2.2.2.2"` (SGLang's `FAKE_BOOTSTRAP_HOST`) short-circuits
the whole handshake. Dynamo's health-check canary and prefill warmup both use
it; without that path, warmup would block for its 1800 s timeout.

## KV-aware routing

Each worker publishes `BlockStored` / `BlockRemoved` events as its prefix cache
changes. Dynamo's router subscribes, builds a radix tree per worker, and sends
a request to whichever worker already holds the longest matching prefix.

The subtlety worth knowing: Dynamo treats the published `block_hashes` as
**opaque engine ids** and recomputes its own prefix key from the published
`token_ids` (`lib/kv-router/src/zmq_wire/convert.rs`). So the token payload has
to be exactly right — wrong or misaligned tokens produce hashes that silently
never match the frontend's, and routing degrades to round-robin with no error
anywhere. `tests/test_kv_events.py` asserts the published tokens rehash to what
`compute_block_hash_for_seq` gives for the same prompt.

Measured on this stack with two workers (`./launch/agg_kv_router.sh`), a
~6700-token prompt sent twice:

```
pass   overlap_blocks
cold   0
warm   422          <- full prefix hit, routed back to the worker holding it
```

With `--kv-events-config` omitted the same test gives `0` on both passes, which
is what proves the match comes from the published events rather than the
router's own bookkeeping.

> **Known flakiness.** Occasionally the warm pass still reports
> `overlap_blocks=0` on a freshly started stack. The engine is emitting
> correctly in those runs (verified with a raw ZMQ subscriber), so this is a
> delivery or indexer-timing race on the Dynamo side, not in the fake engine.
> A block is published once and never re-announced, so an event lost in that
> window is not recovered. Restarting the stack clears it.

## Layout

```
src/sglang/       the shim: SGLang-shaped facade, no simulation logic
                    (incl. the KV-event wire format, matched to SGLang's)
src/fakeengine/   the simulator: scheduler, KV pool, timing, bootstrap,
                    transfer, KV-event policy
launch/           agg.sh, disagg.sh, agg_kv_router.sh,
                    cluster.sh (+ clusters/*.yaml)
scripts/          bootstrap_macos.sh, fetch_model_metadata.py
models/           pinned model metadata (config + tokenizer, no weights)
tests/            contract tests + simulator behavior + disagg + KV events
                    + topology + cluster launcher
```

The split is the point: `src/sglang/` translates, `src/fakeengine/` behaves.
Swapping in real SGLang, or driving the simulator from a different facade, only
touches one side.

## Tests

```bash
pytest tests/                                                    # 73 tests
pytest ../dynamo/components/src/dynamo/sglang/tests -m unit      # 556 pass
```

`tests/test_shim_contract.py` is the drift guard. It re-derives the contract
from Dynamo's own source — every `sglang.*` import, every `server_args.<field>`
and `tokenizer_manager.<attr>` read — and fails by name when Dynamo starts
using something the shim lacks. It understands Dynamo's `_compat.py` N/N-1
fallback groups, requiring only that *one* alternative in each group resolves.

### Known gaps

Three of Dynamo's own SGLang tests still fail, none of them core:

- `test_video_encode_real`, `test_compat_supports_tensor_image_sizes` —
  multimodal image/video paths, deliberately stubbed out.
- `test_cancelled_sync_engine_route_keeps_engine_routes_serialized` — uses
  `asyncio.timeout`, which is Python 3.11+. Unrelated to the shim; the
  bootstrap pins 3.10 because `ai-dynamo`'s `aisimulate` dependency is marked
  `python_version >= '3.11'` and ships only an sdist. Build on 3.12 (Rust is
  already installed by then) if you need that test.

## Not simulated yet

- **Forward-pass metrics (FPM).** The wire schema is implemented and matched to
  Dynamo's (its contract test passes), but the engine publishes no FPM stream;
  scheduler metrics go over the simpler `metrics_ipc_name` socket instead.
- **Multi-rank KV events.** One DP rank per worker. The per-rank port offset is
  implemented, but nothing drives `dp_size > 1`.
- Multimodal, embedding/rerank, diffusion, LoRA, and weight updates — these
  raise a clear `NotImplementedError` rather than pretending.
