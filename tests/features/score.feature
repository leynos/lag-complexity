Feature: Score queries

  Background:
    Given a dummy complexity function

  Scenario: valid query
    When scoring "hello"
    Then the total score is 3.0

  Scenario: empty query
    When scoring empty query
    Then an error is returned
