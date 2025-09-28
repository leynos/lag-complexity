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

  Scenario: scoring multiple pronouns without an antecedent
    When scoring "It broke. They panicked."
    Then the scored complexity components are 0.0, 0.0, 5.0
    And the scored total is 5.0

  Scenario: scoring a capitalised adverb before a pronoun
    When scoring "However, it broke."
    Then the scored complexity components are 0.0, 0.0, 3.0
    And the scored total is 3.0

  Scenario: scoring idiomatic apostrophe pronoun usage
    When scoring "It's raining."
    Then the scored complexity components are 0.0, 0.0, 3.0
    And the scored total is 3.0

  Scenario: scoring idiomatic apostrophe pronoun usage with curly apostrophe
    When scoring "It’s raining."
    Then the scored complexity components are 0.0, 0.0, 3.0
    And the scored total is 3.0

  Scenario: scoring antecedent across a sentence boundary
    When scoring "Alice fixed the radio. However, it works."
    Then the scored complexity components are 0.0, 0.0, 2.0
    And the scored total is 2.0
