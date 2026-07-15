# Matmul N-dim span padding (issue #1918)

## Problem

`F.linear(x[1, K], W[N, K])` (e.g. lm_head, `N=49216`, `K=4096`) SIGABRTs in
`dxp_standalone`. The weight's device layout is `[N/64, K, 64]`. Per-core span in
`work_division` is `outer_stick_range × Π(full inner device_size) × itemsize`, so
only splitting the **outer stick dim** (`N/64`) reduces span. That split must
divide the stick count, and `49216/64 = 769` is **prime** → the only splits are
`{1, 769}`, neither ≤ 32 cores. Span stays 384.5 MB > 256 MB → EAR overflow.

Empirically verified (latest `main`, torch-spyre `e4cd21e7`):
`49216` (769 sticks, prime) FAILS; `49280` (770 = 2·5·7·11) PASSES with no span
warnings (span reduction splits `d0` by 2 → 385 sticks → 192.5 MB).

## Fix

Pad the matmul weight's **N** dim up by whole sticks so the outer stick count is
**composite enough to split** under the 256 MB limit, grow the matmul output to
the padded N, then **narrow the output back to true N** (a stick-aligned slice,
since we pad by whole sticks). Padded output columns are independent of the true
columns, so no zero-masking of logits is needed — the extra stick is simply
dropped. Everything else (span reduction, work division) is unchanged and already
handles the composite case.

This is the compiler's job, not HF adapters': it's a Spyre stick/span constraint,
the reported repro is raw `torch.compile` (no adapter), and the compiler already
owns matmul-weight padding (`padding.py::insert_bmm_padding` pads y's K dim).

## Approach: extend `padding.py` with an N-pad path

Add `insert_matmul_n_padding(graph)` in `torch_spyre/_inductor/padding.py`, and
register it in `passes.py` **immediately after `insert_bmm_padding`** (line ~337),
before `dedup_and_promote_constants` and `span_reduction`. At that point layouts
are `FixedTiledLayout` (exact device_size / stick info) and it runs before span
reduction, so the padded weight is what work division sees.

### Per BATCH_MATMUL_OP:

1. **Identify y (weight) and its N stick dim.** Reuse `identify_matmul_inputs()`
   / `host_coordinates()` (as K-pad does). N is the output-carried symbol absent
   from x. Confirm y's layout puts N on the outer stick dim; if not, bail (no
   pad, current behavior).

2. **Decide whether to pad (span-driven).** Reuse `work_division.get_per_core_span`
   / `MAX_SPAN_BYTES`:
   - `n_sticks = device_size[stick_dim]`
   - `stick_span_bytes = Π(device_size[stick_dim+1:]) × itemsize` (bytes per stick)
   - `max_sticks_per_core = MAX_SPAN_BYTES // stick_span_bytes`
   - `required_split = ceil(n_sticks / max_sticks_per_core)`
   - If `n_sticks` already has a divisor `d` with `required_split ≤ d ≤ sencores`
     and `ceil(n_sticks/d) ≤ max_sticks_per_core` → **no pad needed** (skip).

3. **Compute pad target.** Smallest `n_sticks_padded ≥ n_sticks` that HAS such a
   divisor. (769 → 770 via `d=2`.) Search upward; in practice the next even value
   works. `pad_elems = (n_sticks_padded − n_sticks) × elems_per_stick`.

4. **Pad y along N.** Reuse `lower_pad_sequence(y_fx_node, padded_size, ..., dim=y_N_host_dim, insert_before=matmul_fx_node, orig_stl=...)` — same helper K-pad uses.
   (Zero-fill is not required for correctness here, but reusing the zero-fill
   sequence is simplest and harmless.)

5. **Grow the matmul output N + narrow back.** This is the one piece K-pad does
   NOT do (K is reduced away; N is a live output dim):
   - Grow the `Reduction.ranges[N]` (and the output buffer's `FixedTiledLayout`)
     to `N_padded`, and rewire y's loader to the padded buffer (extend
     `_rebuild_matmul`).
   - Insert a stick-aligned narrow `[.., 0:N]` (ReinterpretView / slice
     ComputedBuffer) after the matmul and repoint the matmul's original consumers
     to it. Because padding is by whole sticks, the narrow drops whole trailing
     stick(s) — clean on the tiled layout.

### Alternative considered (fallback)

Do the pad at **lowering** (`lower_mm`): pad y via `constant_pad_nd`, build the
reduction at `N_padded`, wrap the result in a `slice` to `N`. Pros: layout / WSR /
span reduction all flow from `N_padded` naturally — no post-layout output surgery.
Cons: span decision uses an estimate before real layout (accurate for the standard
`[N/64, K, 64]` weight layout, since we know N, K, itemsize, stick=64) and it sits
on the hot lowering path. Keep this as the fallback if step 5's post-layout output
surgery proves too invasive.

## Correctness

- Padded N columns are computed from padded weight rows and are independent of the
  true N columns (matmul output columns don't interact) → dropping them is exact.
  No logit masking needed (unlike HF-side vocab padding).
- Narrow is stick-aligned (whole-stick pad) → no partial-stick view on device.
- Non-overflowing matmuls and composite-stick matmuls hit the step-2 early-out →
  zero behavior change.

## Testing

- New test in `tests/inductor/` (use `write-spyre-op-test` framework):
  - `F.linear` with prime-stick vocab `49216` → compiles, `out[:, :49216]`
    matches CPU reference within fp16 tol.
  - `49280` (already composite) and real Granite `49152` → still pass, and assert
    NO pad is inserted for the composite cases (step-2 early-out).
  - A small non-overflow matmul → assert no pad / no output narrow.
- Regression: `tests/inductor/test_inductor_ops.py`, `test_building_blocks.py`,
  and coarse-tile e2e suite unchanged.
- Empirical: rerun `spyre-launcher/scripts/repro_issue_1918.py` on the pod →
  expect both 49216 and 49280 PASS.

## Risks / open questions

- **Output-grow surgery (step 5)** is the main risk: correctly re-laying-out the
  matmul output at `N_padded` after layouts are assigned, and rewiring consumers
  through the narrow. If messy, switch to the lowering-level fallback.
- **Constant weight perf:** padding a parameter emits a runtime pad each call.
  `dedup_and_promote_constants` already dedups pad constants; constant-folding the
  N-pad (pad the parameter once at materialization) is a follow-up optimization,
  not required for correctness.
- **Composition with K-pad:** both pad y; N-pad runs first, K-pad then pads the
  already-N-padded buffer's K. Verify the two pad sequences compose (expected to,
  since they touch orthogonal dims).
- **Layout assumption:** only the standard `N`-outer-stick weight layout is
  handled; other layouts bail (no regression, just unfixed — file follow-up if
  they occur, e.g. umacdevi's conv2d-via-unfold shapes).
```
