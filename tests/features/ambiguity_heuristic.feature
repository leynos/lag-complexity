Feature: Ambiguity heuristic

  Background:
    Given an ambiguity heuristic

  Scenario: ambiguous query
    When evaluating "It is about Mercury and some others."
    Then the ambiguity score is 5.0

  Scenario: empty input
    When evaluating empty input
    Then an ambiguity error is returned
