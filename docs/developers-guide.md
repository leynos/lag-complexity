# Developer guide

This guide records development practices that are specific to maintaining this
repository. Follow the project-wide guidance in `AGENTS.md` first, then use
this guide for workflow automation details.

## Continuous Integration workflow

The Continuous Integration (CI) workflow lives in `.github/workflows/ci.yml`.
The `build-test` job is the required check for pull requests and runs
formatting, linting, and tests through the Makefile targets used locally.

The CodeScene upload step must route `secrets.CS_ACCESS_TOKEN` through the step
`env` block and gate on `env.CS_ACCESS_TOKEN`. GitHub Actions does not permit
the `secrets` context inside `if:` expressions, so using
`if: ${{ secrets.CS_ACCESS_TOKEN }}` makes the workflow invalid before any job
can start. Keep the gate in this form:

```yaml
env:
  CS_ACCESS_TOKEN: ${{ secrets.CS_ACCESS_TOKEN }}
if: ${{ env.CS_ACCESS_TOKEN }}
```

The workflow validation tests in `tests/workflows.rs` assert this contract so
future workflow edits fail in `make test` before they break CI.

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
