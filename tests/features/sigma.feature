Feature: Sigma normalisation

  Scenario: minmax inside range
    Given a minmax sigma
    When normalising 5.0
    Then the result is 0.5

  Scenario: minmax clamped below
    Given a minmax sigma
    When normalising -5.0
    Then the result is 0.0

  Scenario: minmax clamped above
    Given a minmax sigma
    When normalising 15.0
    Then the result is 1.0

  Scenario: zscore zero std
    Given a zscore sigma with zero std
    When normalising 1.0
    Then the result is 0.5

  Scenario: robust standard
    Given a robust sigma
    When normalising 1.0
    Then the result is 0.6625
    # Note: comparison uses a tolerance of 1e-4

  Scenario: robust zero mad
    Given a robust sigma with zero mad
    When normalising 1.0
    Then the result is 0.5

