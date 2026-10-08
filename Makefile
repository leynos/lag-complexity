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
UV ?= uv
UV_ENV = UV_CACHE_DIR=.uv-cache UV_TOOL_DIR=.uv-tools

# The CV-005 CodeScene contracts live in shared-actions and run from a full
# commit, so a fix is a pin bump. `.github/cv005.toml` holds this repository's
# only parameters.
CV005_CONTRACTS_REF ?= 88977798a5c3bae1549afb99642529488c665276
CV005_CONTRACTS = $(UV_ENV) $(UV) tool run --python 3.13 \
	--from 'git+https://github.com/leynos/shared-actions@$(CV005_CONTRACTS_REF)\#subdirectory=packages/cv005-contracts' \
	cv005-contracts

TYPOS_CONFIG_BUILDER_VERSION ?= v0.1.1
TYPOS_CONFIG_BUILDER = $(UV_ENV) $(UV) tool run --python 3.14 --from \
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
	uv run --with 'pytest>=8' --with 'pyyaml>=6' pytest tests/workflow_contracts -q

target/%/$(APP): ## Build binary in debug or release mode
	$(if $(findstring release,$(@)),RUSTFLAGS="$${RUSTFLAGS-}",RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)") $(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

lint: ## Run Clippy and the Whitaker Dylint suite with warnings denied
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)" $(CARGO) clippy $(CLIPPY_FLAGS)
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(WHITAKER) --all -- --all-targets --all-features

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
