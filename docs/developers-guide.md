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

The workflow validation tests in `tests/workflows.rs` and the contract tests in
`tests/workflow_contracts/` assert this shape so future workflow edits fail in
`make test` or `make test-workflow-contracts` before they break CI.

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

Two gaps are known and accepted. Merges made by the Dependabot automerge
workflow use `GITHUB_TOKEN` and fire no push event, so they are measured only
at the next push to `main` or a manual dispatch. And the publisher's
concurrency group is keyed on the ref alone and never cancels a run in
progress, so runs on `main` never overlap and a newer run replaces any pending
one; a dispatch that replaces a pending push uploads the same or a newer
commit, but leaves the ratchet baseline one commit behind until the next push.
Both are tracked in leynos/shared-actions#518.

No other workflow a push starts, directly or through a local call, may generate
coverage outside the pull-request guard, so the publisher is the only baseline
writer. Both coverage steps select the same inputs at the same `shared-actions`
pin because the pull-request ratchet is only meaningful against a baseline
measured the same way.

`make test-workflow-contracts` holds this shape. The contract tests are
`codescene_pull_request_test.py`, `codescene_publisher_test.py` and
`codescene_token_test.py` under `tests/workflow_contracts/`, with the rules in
the `codescene_*_rules.py` modules beside them and the strict workflow reader in
`codescene_workflow_reader.py`. The rules read every workflow a pull request
can start, from its own events, reviews and comments, a merge queue, or a push
not confined to `main` or tags, following local reusable-workflow calls and
`workflow_run` chains, and refuse any mention of the CodeScene host, uploader,
client, or token there. They also refuse `continue-on-error` wherever it would
turn a failed ratchet or upload green. The upload guard is compared as an exact
set of conjuncts, so an `||` hidden inside an extra conjunct fails the
comparison without a separate scan. Each clause has a test that mutates the
workflows and expects the clause to refuse the result.

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
