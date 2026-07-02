//! Behavioural checks for GitHub Actions workflow contracts.

use std::{error::Error, fs};

use serde::Deserialize;
use serde_yaml::Value;

const CI_WORKFLOW: &str = ".github/workflows/ci.yml";
const DEPENDABOT_AUTOMERGE_WORKFLOW: &str = ".github/workflows/dependabot-automerge.yml";

#[derive(Debug, Deserialize)]
struct Workflow {
    #[serde(rename = "on")]
    triggers: Value,
    #[serde(default)]
    permissions: Value,
    jobs: Value,
}

#[test]
fn ci_codescene_upload_gates_on_env_secret() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(CI_WORKFLOW)?;
    let build_test = mapping_value(&workflow.jobs, "build-test")?;
    let steps = sequence_value(mapping_value(build_test, "steps")?)?;
    let upload_step = named_step(steps, "Upload coverage data to CodeScene")?;

    let env = mapping_value(upload_step, "env")?;
    assert_scalar_eq(
        mapping_value(env, "CS_ACCESS_TOKEN")?,
        "${{ secrets.CS_ACCESS_TOKEN }}",
    );
    assert_scalar_eq(
        mapping_value(upload_step, "if")?,
        "${{ env.CS_ACCESS_TOKEN }}",
    );

    Ok(())
}

#[test]
fn dependabot_automerge_uses_deny_all_default_permissions() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(DEPENDABOT_AUTOMERGE_WORKFLOW)?;

    assert!(mapping(&workflow.permissions)?.is_empty());

    Ok(())
}

#[test]
fn dependabot_automerge_gates_pull_request_target_by_author() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(DEPENDABOT_AUTOMERGE_WORKFLOW)?;
    let automerge = mapping_value(&workflow.jobs, "automerge")?;

    assert_scalar_eq(
        mapping_value(automerge, "if")?,
        concat!(
            "${{ github.event_name == 'workflow_dispatch' || ",
            "github.event.pull_request.user.login == 'dependabot[bot]' }}"
        ),
    );

    Ok(())
}

#[test]
fn dependabot_automerge_grants_only_required_job_permissions() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(DEPENDABOT_AUTOMERGE_WORKFLOW)?;
    let automerge = mapping_value(&workflow.jobs, "automerge")?;
    let permissions = mapping_value(automerge, "permissions")?;

    assert_scalar_eq(mapping_value(permissions, "contents")?, "write");
    assert_scalar_eq(mapping_value(permissions, "pull-requests")?, "write");
    assert_scalar_eq(mapping_value(permissions, "checks")?, "read");
    assert_scalar_eq(mapping_value(permissions, "statuses")?, "read");
    assert_scalar_eq(mapping_value(permissions, "id-token")?, "write");

    Ok(())
}

#[test]
fn dependabot_automerge_keeps_expected_triggers() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(DEPENDABOT_AUTOMERGE_WORKFLOW)?;
    let triggers = mapping(&workflow.triggers)?;
    let pull_request_target = mapping_value(&workflow.triggers, "pull_request_target")?;
    let workflow_dispatch = mapping_value(&workflow.triggers, "workflow_dispatch")?;

    assert!(triggers.contains_key(Value::String("pull_request_target".to_owned())));
    assert!(triggers.contains_key(Value::String("workflow_dispatch".to_owned())));
    assert_sequence_eq(mapping_value(pull_request_target, "branches")?, &["main"])?;
    assert_sequence_eq(
        mapping_value(pull_request_target, "types")?,
        &[
            "opened",
            "reopened",
            "synchronize",
            "labeled",
            "ready_for_review",
        ],
    )?;
    assert!(mapping_value(workflow_dispatch, "inputs").is_ok());

    Ok(())
}

fn read_workflow(path: &str) -> Result<Workflow, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let workflow = serde_yaml::from_str(&contents)?;

    Ok(workflow)
}

fn named_step<'a>(steps: &'a [Value], name: &str) -> Result<&'a Value, Box<dyn Error>> {
    steps
        .iter()
        .find(|step| scalar_value(mapping_value(step, "name").ok()) == Some(name))
        .ok_or_else(|| format!("missing workflow step named {name:?}").into())
}

fn mapping(value: &Value) -> Result<&serde_yaml::Mapping, Box<dyn Error>> {
    value
        .as_mapping()
        .ok_or_else(|| format!("expected mapping, found {value:?}").into())
}

fn mapping_value<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Box<dyn Error>> {
    mapping(value)?
        .get(Value::String(key.to_owned()))
        .ok_or_else(|| format!("missing mapping key {key:?}").into())
}

fn sequence_value(value: &Value) -> Result<&[Value], Box<dyn Error>> {
    value
        .as_sequence()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("expected sequence, found {value:?}").into())
}

fn assert_sequence_eq(value: &Value, expected: &[&str]) -> Result<(), Box<dyn Error>> {
    let actual = sequence_value(value)?
        .iter()
        .map(|entry| {
            scalar_value(Some(entry))
                .map(str::to_owned)
                .ok_or_else(|| format!("expected scalar entry, found {entry:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    assert_eq!(actual, expected);

    Ok(())
}

fn assert_scalar_eq(value: &Value, expected: &str) {
    assert_eq!(scalar_value(Some(value)), Some(expected));
}

fn scalar_value(value: Option<&Value>) -> Option<&str> {
    value.and_then(Value::as_str)
}
