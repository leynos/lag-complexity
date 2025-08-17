Feature: Score queries

  Scenario: valid query
    Given a dummy complexity function
    When scoring "hello"
    Then the total score is 3.0

  Scenario: empty query
    Given a dummy complexity function
    When scoring empty query
    Then an error is returned
