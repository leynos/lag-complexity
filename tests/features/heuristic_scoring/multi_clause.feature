Feature: Heuristic scoring - multi-clause query

  Background:
    Given a heuristic complexity scorer

  Scenario: scoring a multi-clause question
    When scoring "If Alice and Bob play chess, who wins?"
    Then the scored complexity components are 0.0, 4.0, 1.0
    And the scored total is 5.0
