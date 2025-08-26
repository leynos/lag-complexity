Feature: lagc CLI stub

  Scenario: run in dry-run mode
    Given the lagc binary
    When running with "--dry-run=true"
    Then it exits successfully

  Scenario: run with invalid flag value
    Given the lagc binary
    When running with "--dry-run=maybe"
    Then it exits with an error

  Scenario: run in dry-run mode via environment variable
    Given the lagc binary
    And these environment variables
      | LAGC_DRY_RUN | true |
    When running with no args
    Then it exits successfully
