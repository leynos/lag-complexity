Feature: API embedding provider

  Background:
    Given an API embedding provider

  Scenario: successful request
    When embedding "hello"
    Then an embedding of length 2

  Scenario: server failure
    When embedding fails
    Then an embedding error is returned
