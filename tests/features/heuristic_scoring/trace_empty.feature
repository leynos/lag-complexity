Feature: Heuristic tracing - empty query

  Background:
    Given a heuristic complexity scorer

  Scenario: tracing empty query
    When tracing empty query
    Then a depth error is returned from trace
