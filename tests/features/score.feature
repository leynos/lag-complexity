Feature: Heuristic scoring

  Background:
    Given a heuristic complexity scorer

  Scenario: scoring a multi clause question
    When scoring "If Alice and Bob play chess, who wins?"
    Then the scored complexity components are 0.0, 4.0, 1.0
    And the scored total is 5.0

  Scenario: scoring an ambiguous phrasing
    When scoring "A few cats chase a dog; what happens?"
    Then the scored complexity components are 0.0, 0.0, 2.0
    And the scored total is 2.0

  Scenario: scoring empty query
    When scoring empty query
    Then a depth error is returned

  Scenario: tracing a simple query
    When tracing "Plain question"
    Then the trace query is "Plain question"
    And the traced complexity components are 0.0, 0.0, 1.0
    And the traced total is 1.0

  Scenario: tracing empty query
    When tracing empty query
    Then a depth error is returned from trace
