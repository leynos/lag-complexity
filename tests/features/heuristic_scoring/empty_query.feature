Feature: Heuristic scoring - empty query

  Background:
    Given a heuristic complexity scorer

  Scenario: scoring empty query
    When scoring empty query
    Then a depth error is returned
