Feature: Depth heuristic

  Background:
    Given a depth heuristic

  Scenario: complex query
    When evaluating "If Alice and Bob play chess, who wins?"
    Then the depth score is 4.0

  Scenario: empty input
    When evaluating empty input
    Then a depth error is returned
