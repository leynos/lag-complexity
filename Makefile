.PHONY: help all clean test test-workflow-contracts build release lint typecheck \
	fmt check-fmt markdownlint nixie spelling bless

APP ?= lagc
CARGO ?= cargo
BUILD_JOBS ?=
CLIPPY_FLAGS ?= --all-targets --all-features -- -D warnings
MDLINT ?= markdownlint-cli2
# `make fmt` and `make check-fmt` call mdtablefix directly. `--git` selects the
# Markdown files Git tracks and `--include-untracked` adds the untracked files
# Git does not ignore, so a new document is formatted before it is staged.
# Both modes need mdtablefix 0.6.0 or later; CI pins the version at the
# install-mdtablefix step.
MDTABLEFIX ?= mdtablefix
MDTABLEFIX_SELECT = --git --include-untracked
MDTABLEFIX_RULES = --wrap --renumber --breaks --ellipsis --fences
NIXIE ?= nixie
WHITAKER ?= whitaker
# Every uv call but the spelling builder's goes through the vendored helper,
# which cleans the environment, uses the global uv cache, runs offline first and
# goes online at most once. The builder runs directly at its release tag,
# because the spelling baseline requires that form and the helper accepts only
# a full commit. See
# https://github.com/leynos/shared-actions/tree/main/uv_gate for the contract.
# `scripts/uv_gate.py` is the shared-actions helper, vendored byte for byte, so
# update it by copying the canonical file, never by editing it here. The helper
# finds uv on PATH (outside ~/.lody, which it removes), so there is no UV
# override: put the uv you want first. It needs python3 3.9 or newer. A uv
# that reports only `Git operation failed` for a Git source missing from the
# cache is not retried online: warm the cache once with `uv tool run`.
UV_GATE ?= python3 scripts/uv_gate.py

# The CV-005 CodeScene contracts live in shared-actions and run from a full
# commit, so a fix is a pin bump. `.github/cv005.toml` holds this repository's
# only parameters.
CV005_CONTRACTS_REF ?= 88977798a5c3bae1549afb99642529488c665276
CV005_CONTRACTS = $(UV_GATE) tool --python 3.13 \
	--from 'git+https://github.com/leynos/shared-actions@$(CV005_CONTRACTS_REF)\#subdirectory=packages/cv005-contracts' \
	-- cv005-contracts

TYPOS_CONFIG_BUILDER_VERSION ?= v0.1.3
TYPOS_CONFIG_BUILDER = uv tool run --python 3.14 --from \
	"git+https://github.com/leynos/typos-config-builder.git@$(TYPOS_CONFIG_BUILDER_VERSION)" \
	typos-config-builder

# The development build standard (concordat rule `rust-build-defaults`):
# the parallel rustc frontend and, on Linux, the mold linker. An assigned
# RUSTFLAGS replaces every `rustflags` table in .cargo/config.toml, so each
# recipe that sets it composes these onto any inherited value (CI's
# setup-rust exports one), except coverage, which stays on LLVM and the
# platform linker.
BUILD_HOST_OS := $(shell uname -s)
STANDARD_RUSTFLAGS := -Zthreads=8$(if $(filter Linux,$(BUILD_HOST_OS)), -Clink-arg=-fuse-ld=mold)

build: target/debug/$(APP) ## Build debug binary
release: target/release/$(APP) ## Build release binary

all: release spelling test-workflow-contracts ## Build the release binary and enforce spelling

clean: ## Remove build artifacts
	$(CARGO) clean

test: ## Run tests with warnings treated as errors
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) test --all-targets --all-features $(BUILD_JOBS)
	# --all-targets skips doctests, so run them explicitly to keep CI
	# aligned with the plain `cargo test` used by cargo-mutants.
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) test --doc --all-features $(BUILD_JOBS)

test-workflow-contracts: ## Validate the mutation-testing and CodeScene coverage workflow contracts
	$(CV005_CONTRACTS) check --repository .
	$(UV_GATE) tool --python 3.13 --from 'pytest==9.2.0' --with 'pyyaml==6.0.3' -- pytest tests/workflow_contracts -q

target/%/$(APP): ## Build binary in debug or release mode
	$(if $(findstring release,$(@)),RUSTFLAGS="$${RUSTFLAGS-}",RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)") $(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

# Whitaker runs under the toolchain its installer manages, so it takes none of
# the development standard: its literal `RUSTFLAGS` displaces the `rustflags`
# tables (no frontend flag, no `mold`) and inherits nothing from the caller. An
# inherited `CARGO_ENCODED_RUSTFLAGS` outranks `RUSTFLAGS`, so it is cleared first.
# The repository's only Python is the mutation-testing workflow contract test
# under tests/. Interrogate scans each target recursively, subject to its
# default exclusions.
INTERROGATE_VERSION ?= 1.7.0
INTERROGATE_TARGETS ?= scripts tests
# Level 2 prints a MISSED row per undocumented definition, which the
# `interrogate` target fails on: the rounded percentage alone can pass a lone gap.
INTERROGATE_FLAGS ?= -vv --fail-under 100
INTERROGATE ?= $(UV_GATE) tool --python 3.13 --from 'interrogate==$(INTERROGATE_VERSION)' -- interrogate

# Interrogate rounds its percentage to one decimal, so `--fail-under 100` alone
# can pass a lone gap in a large repository; fail on any MISSED row as well.
.PHONY: interrogate
interrogate: ## Enforce complete Python docstring coverage
	@report="$$($(INTERROGATE) $(INTERROGATE_FLAGS) $(INTERROGATE_TARGETS))"; \
	  status=$$?; \
	  printf '%s\n' "$$report"; \
	  test "$$status" -eq 0 || exit "$$status"; \
	  if printf '%s\n' "$$report" | grep -q '| *MISSED *|'; then \
	    printf '%s\n' 'interrogate: undocumented definitions remain; see the MISSED rows above' >&2; \
	    exit 1; \
	  fi

lint: yamllint actionlint ## Run Clippy, the Whitaker Dylint suite and the docstring gate
	$(MAKE) --no-print-directory interrogate
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)" $(CARGO) clippy $(CLIPPY_FLAGS)
	unset CARGO_ENCODED_RUSTFLAGS && RUSTFLAGS="-D warnings" $(WHITAKER) --all -- --all-targets --all-features

typecheck: ## Check all targets and features with warnings denied
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) check --all-targets --all-features $(BUILD_JOBS)

fmt: ## Format Rust and Markdown sources
	$(CARGO) fmt --all
	$(MDTABLEFIX) --in-place $(MDTABLEFIX_SELECT) $(MDTABLEFIX_RULES)
	@unset FORCE_COLOR; $(MDLINT) --fix "**/*.md"

check-fmt: ## Verify formatting
	$(CARGO) fmt --all -- --check
	$(MDTABLEFIX) --check $(MDTABLEFIX_SELECT) $(MDTABLEFIX_RULES)

markdownlint: spelling ## Lint Markdown files and enforce spelling
	find . -type f -name '*.md' -not -path './target/*' -not -path './.uv-cache/*' -not -path './.uv-tools/*' -print0 | xargs -0 $(MDLINT)

spelling: ## Enforce en-GB-oxendict spelling
	$(TYPOS_CONFIG_BUILDER) gate --repository .

nixie: ## Validate Mermaid diagrams
	# CI currently requires --no-sandbox; remove once nixie supports
	# environment variable control for this option
	nixie --no-sandbox

bless: ## Regenerate golden test snapshots
	$(CARGO) run --no-default-features --bin bless_traces $(if $(SNAPSHOT),-- "$(SNAPSHOT)",)

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":"; printf "Available targets:\n"} {printf "  %-20s %s\n", $$1, $$2}'

YAMLLINT ?= yamllint
ACTIONLINT ?= actionlint

.PHONY: yamllint actionlint

# yamllint reads `.yamllint.yml`, which extends `.yamllint-estate.yml`, and
# resolves that path against the current directory, so run it from the root.
yamllint: SHELL := bash
yamllint: .SHELLFLAGS := -eo pipefail -c
yamllint: ## Lint every tracked YAML file strictly
	git ls-files -z '*.yml' '*.yaml' | xargs -0 -r $(YAMLLINT) --strict --

actionlint: ## Lint the GitHub Actions workflows, including their shell scripts
	$(ACTIONLINT)
