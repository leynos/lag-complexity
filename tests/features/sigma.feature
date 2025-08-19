Feature: Sigma normalisation

  Scenario: minmax inside range
    Given a minmax sigma
    When normalising 5.0
    Then the result is 0.5

  Scenario: minmax clamped below
    Given a minmax sigma
    When normalising -5.0
    Then the result is 0.0

  Scenario: zscore zero std
    Given a zscore sigma with zero std
    When normalising 1.0
    Then the result is 0.5

  Scenario: robust standard
    Given a robust sigma
    When normalising 1.0
    Then the result is 0.6625

  Scenario: robust zero mad
    Given a robust sigma with zero mad
    When normalising 1.0
    Then the result is 0.5

