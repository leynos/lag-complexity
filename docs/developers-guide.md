# Developer guide

This guide records development practices that are specific to maintaining this
repository. Follow the project-wide guidance in `AGENTS.md` first, then use
this guide for workflow automation details.

## Spelling policy

Run `make spelling` to enforce en-GB-oxendict prose spelling. Every run
regenerates `typos.toml` from the live shared dictionary and the
`typos.local.toml` overlay, so `typos.toml` is never drift checked in CI. Put
narrow repository-specific exceptions in `typos.local.toml`; never edit
generated entries by hand.

## Continuous Integration workflow

The Continuous Integration (CI) workflow lives in `.github/workflows/ci.yml`.
The `build-test` job is the required check for pull requests and runs
formatting, linting, and tests through the Makefile targets used locally.

Pull requests never contact CodeScene; `.github/workflows/coverage-main.yml`
alone uploads coverage, as described under "Coverage publication" below. GitHub
Actions does not permit the `secrets` context inside `if:` expressions, so using
`if: ${{ secrets.CS_ACCESS_TOKEN }}` makes the workflow invalid before any job
can start. The publisher therefore gates the upload on the output of a check
step whose command evaluates the secret's presence:

```yaml
- name: Check for the CodeScene token
  id: codescene-token
  run: echo "available=${{ secrets.CS_ACCESS_TOKEN != '' }}" >> "$GITHUB_OUTPUT"
- name: Upload coverage data to CodeScene
  if: steps.codescene-token.outputs.available == 'true' && github.ref == 'refs/heads/main'
```

The workflow validation tests in `tests/workflows.rs` and the shared CV-005
contract library that `make test-workflow-contracts` runs assert this shape so
future workflow edits fail in `make test` or `make test-workflow-contracts`
before they break CI.

## Dependabot auto-merge workflow

The Dependabot auto-merge caller lives in
`.github/workflows/dependabot-automerge.yml`. It uses `pull_request_target`
because the workflow needs write permissions to approve and enable auto-merge
on eligible Dependabot pull requests. The called reusable workflow does not
check out or execute pull request code; it reads event metadata and uses the
GitHub API.

The workflow must keep top-level permissions set to `{}` so jobs do not inherit
repository defaults. The `automerge` job then grants only the scopes required
by the pinned reusable workflow:

- `contents: write` to approve and enable auto-merge.
- `pull-requests: write` to update pull request review and merge state.
- `checks: read` and `statuses: read` to inspect required checks.
- `id-token: write` so the reusable workflow can resolve its own pinned
  workflow reference through GitHub OpenID Connect (OIDC) metadata.

For `pull_request_target` events, the workflow gate must check
`github.event.pull_request.user.login == 'dependabot[bot]'`. Do not gate on
`github.actor`: a maintainer can rerun or otherwise trigger a
`pull_request_target` workflow for a Dependabot-authored pull request, and the
actor then differs from the pull request author. `workflow_dispatch` remains
allowed for manual operation.

## Coverage publication

Pull-request continuous integration (CI) generates LCOV coverage and ratchets
it against the baseline written by `coverage-main.yml`. The pull-request lane
publishes no coverage artefact, never contacts CodeScene, and never receives
`CS_ACCESS_TOKEN`, so a change in CodeScene's application programming interface
(API) cannot hold a pull request.

`coverage-main.yml` is the only publisher. On each push to `main` it refreshes
the ratchet baseline and uploads the report to CodeScene. It also runs on
demand through `workflow_dispatch`, for merges that fire no push event: a
dispatch on `main` uploads a fresh report, but the shared action advances the
baseline only on a push, so the ratchet catches up at the next push to `main`.
No `env` binds `CS_ACCESS_TOKEN`: a check step writes whether the secret is
set, from an expression evaluated before its shell runs, and the upload step
receives the token only as its `access-token` input, because the uploader is a
composite action that would pass its step's `env` to its nested steps. The
upload runs only when the token is present and the ref is `refs/heads/main`, so
a dispatch from a branch cannot publish that branch's coverage as the trunk's.

The publisher's concurrency group is keyed on the ref alone and never cancels a
run in progress, so runs on `main` never overlap, and a newer trigger replaces
an older pending run rather than queueing behind it. GitHub does not promise to
start runs in trigger order, so this does not guarantee commit order: an older
run that starts late can publish its commit's coverage after a newer one, and
the next push supersedes it. A manual re-run of an older run keeps its SHA and
its run id: it republishes that commit's coverage to CodeScene, but replaces no
ratchet baseline while the original run's cache entry survives, because the
shared action saves each baseline under a key that includes the run id. If that
entry is gone, never saved or since evicted, a re-run of a push saves the older
commit's baseline again, the shared action restores the newest entry under the
key prefix, and later ratchets read the older baseline until the next push
saves a newer one. That stale-order risk is accepted. A re-run of a dispatch
saves no baseline, since the shared action saves one only on a push.

Two gaps are known and accepted, and both are tracked in
[shared-actions issue 518](https://github.com/leynos/shared-actions/issues/518):

- Merges made by the Dependabot automerge workflow use `GITHUB_TOKEN` and fire
  no push event, so they are measured only at the next push to `main` or a
  manual dispatch.
- A dispatch that replaces a pending push writes no baseline, since the shared
  action saves one only on a push, so the ratchet baseline stays behind until
  the next push. A dispatch made before a push can also reach the concurrency
  group after it, replace it and upload the older commit; that is part of the
  stale-order risk accepted above.

No other workflow a push starts, directly or through a local call, may generate
coverage outside the pull-request guard, and none, the publisher included, may
run a local action, whose `action.yml` the contract does not read, so the
publisher is the only baseline writer. Both coverage steps select the same
inputs at the same `shared-actions` pin because the pull-request ratchet is
only meaningful against a baseline measured the same way.

`make test-workflow-contracts` holds this shape by running
`cv005-contracts check`, the shared contract library in `leynos/shared-actions`
(`packages/cv005-contracts`), from a full commit named by `CV005_CONTRACTS_REF`
in the Makefile, and CI runs it as its own step. A fix to the rules is
therefore a pin bump. The target needs `uv`, which fetches the Python 3.13 the
library runs under. The repository's parameters are in `.github/cv005.toml`: its
`repository` name and the `[selection]` inputs the baseline measures, which
the publisher's generator must carry and every pull-request lane must match.
The library's own suite proves each rule refuses the shape it exists to refuse,
so this repository keeps no copy of the readers or the refusal cases. Its rules
read every workflow a pull request can start, from its own events, reviews and
comments, a merge queue, or a push not confined to `main` or tags, following
local reusable-workflow calls, `workflow_run` chains and local composite
actions, and refuse any mention of the CodeScene host, uploader, client, or
token there. They also refuse `continue-on-error` wherever it would turn a
failed ratchet or upload green, and they read workflows strictly, so a
duplicate key is refused rather than silently resolved.

## Workflow pins and Dependabot

Dependabot owns the upgrade of GitHub Actions and reusable workflows, including
calls into `leynos/shared-actions`. Contract tests that assert a caller's exact
commit SHA create a lockstep dependency: every time Dependabot opens a bump PR,
the test fails until a human edits the pinned constant to match. That defeats
the purpose of automated dependency updates and turns a routine bump into a
manual chore.

Contract tests may still verify the *shape* of a reusable-workflow caller. They
must not verify the specific SHA value.

- Do assert the workflow references the correct reusable workflow path.
- Do assert the ref is pinned to a full 40-character commit SHA, not a
  mutable branch such as `main` or `rolling`.
- Do assert the expected `on:` triggers, least-privilege `permissions:`, and
  the inputs the caller relies on.
- Do not hard-code the current SHA value as an expected string. Match it with
  a pattern instead.
- Do not fail a test purely because Dependabot bumped the pinned SHA.

```python
import re

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

def test_uses_pinned_full_sha(caller_step):
    ref = caller_step["uses"].split("@")[-1]
    assert SHA_RE.match(ref), f"expected a 40-hex commit SHA, got {ref!r}"
```

If a workflow's behaviour genuinely depends on a feature only present from a
particular commit onwards, express that as a comment or a changelog note, not
as a test assertion on the SHA string.

## The build standard

Development, test, lint, and typecheck builds use the parallel `rustc` frontend
(`-Zthreads=8`) and, on Linux, the `mold` linker (`-Clink-arg=-fuse-ld=mold`).
These are defaults in `.cargo/config.toml`, which Cargo discovers on its own,
so a bare `cargo build` gets them. `mold` ships for Linux only, so the linker
flag lives in a Linux-only table and macOS and Windows keep their platform
linker. Cargo selects one `rustflags` source rather than merging them, so every
source repeats the same flags apart from the linker.

An assigned `RUSTFLAGS` replaces the configuration's flags, so the Makefile
recipes that set it compose the standard's flags onto any inherited value (CI's
`setup-rust` exports one). Two builds are deliberately excluded: coverage
assigns `RUSTFLAGS` without the fast flags, because a measurement should not
depend on them, and the release recipe and workflow keep the platform linker,
because they assign `RUSTFLAGS` (even an empty value displaces the
configuration). Cargo has no per-profile `rustflags`, so a direct
`cargo build --release` takes the configuration's flags unless `RUSTFLAGS` is
assigned too.

On Linux, install `mold` before building: the configuration names it, so a
build without it fails at link time. CI installs it through `setup-rust`'s
`install-mold` input. `tests/build_standard_contract.rs` holds the standard. It
reads the configuration sources, the commands `make -n` prints for each
development target on a Linux host and a macOS host (each keeping the caller's
own `RUSTFLAGS`) and for each coverage and release target on a Linux host, and
the `setup-rust` steps of the CI workflows (each must pass `install-mold`), so
a flag lost through a recipe or workflow edit fails there. The decision is
recorded in [ADR 001](adr-001-rust-build-standard.md).

### Cranelift

Cranelift is the development-profile codegen backend. The full suite was
measured under it on the pinned `nightly-2025-06-26` on 2026-09-29: all 182
tests across the 13 test binaries and the doctests pass. Coverage selects LLVM
explicitly (`CARGO_PROFILE_DEV_CODEGEN_BACKEND=llvm`), because instrumentation
needs it, and release builds use the release profile, which Cranelift does not
touch. Re-measure the whole suite on the next toolchain bump; if it fails,
record the failing tests here as an exception and remove the backend from
`.cargo/config.toml`.
