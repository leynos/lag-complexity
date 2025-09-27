Feature: Heuristic scoring - pronoun context

  Background:
    Given a heuristic complexity scorer

  Scenario: scoring a pronoun anchored by context
    When scoring "Alice fixed the radio. It works now?"
    Then the scored complexity components are 0.0, 0.0, 2.0
    And the scored total is 2.0

  Scenario: scoring a pronoun without an antecedent
    When scoring "It broke last night."
    Then the scored complexity components are 0.0, 0.0, 3.0
    And the scored total is 3.0
