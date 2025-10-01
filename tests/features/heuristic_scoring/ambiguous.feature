Feature: Heuristic scoring - ambiguous phrasing

  Background:
    Given a heuristic complexity scorer

  Scenario: scoring an ambiguous phrasing
    When scoring "A few cats chase a dog; what happens?"
    Then the scored complexity components are 0.0, 0.0, 2.0
    And the scored total is 2.0

  Scenario: scoring curated ambiguous entity lexicon
    When scoring "Mercury-based alloys and Amazon logistics?"
    Then the scored complexity components are 0.0, 1.0, 5.0
    And the scored total is 6.0
