Feature: Heuristic tracing - simple query

  Background:
    Given a heuristic complexity scorer

  Scenario: tracing a simple query
    When tracing "Plain question"
    Then the trace query is "Plain question"
    And the traced complexity components are 0.0, 0.0, 1.0
    And the traced total is 1.0
