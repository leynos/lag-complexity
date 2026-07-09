//! Behavioural checks for GitHub Actions workflow contracts.

use std::{error::Error, fs};

use serde::Deserialize;
use serde_yaml::Value;

#[derive(Clone, Copy)]
enum WorkflowFile {
    CoverageMain,
    DependabotAutomerge,
}

impl WorkflowFile {
    const fn path(self) -> &'static str {
        match self {
            Self::CoverageMain => ".github/workflows/coverage-main.yml",
            Self::DependabotAutomerge => ".github/workflows/dependabot-automerge.yml",
        }
    }
}

#[derive(Clone, Copy)]
struct StepName<'a>(&'a str);

#[derive(Debug, Deserialize)]
struct Workflow {
    #[serde(rename = "on")]
    triggers: Value,
    #[serde(default)]
    permissions: Value,
    jobs: Value,
}

#[test]
fn coverage_main_upload_gates_on_env_secret() -> Result<(), Box<dyn Error>> {
    // The CodeScene upload lives in coverage-main.yml (push to main), not
    // the pull-request CI job: `cs-coverage upload` is accepted only for
    // analysed branches. The job-level env supplies the token (empty when
    // the secret is unset — e.g. forks), and the step guards on it so a
    // token-less run skips the upload rather than failing.
    let workflow = read_workflow(WorkflowFile::CoverageMain)?;
    let job = mapping_value_path(&workflow.jobs, &["coverage-upload"])?;

    let env = mapping_value(job, "env")?;
    assert_scalar_eq(
        mapping_value(env, "CS_ACCESS_TOKEN")?,
        "${{ secrets.CS_ACCESS_TOKEN || '' }}",
    );

    let steps = sequence_value(mapping_value(job, "steps")?)?;
    let upload_step = named_step(steps, StepName("Upload coverage data to CodeScene"))?;
    assert_scalar_eq(mapping_value(upload_step, "if")?, "env.CS_ACCESS_TOKEN");

    Ok(())
}

#[test]
fn dependabot_automerge_uses_deny_all_default_permissions() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(WorkflowFile::DependabotAutomerge)?;

    assert!(mapping(&workflow.permissions)?.is_empty());

    Ok(())
}

#[test]
fn dependabot_automerge_gates_pull_request_target_by_author() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(WorkflowFile::DependabotAutomerge)?;
    let automerge = mapping_value_path(&workflow.jobs, &["automerge"])?;

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
    let workflow = read_workflow(WorkflowFile::DependabotAutomerge)?;
    let permissions = mapping_value_path(&workflow.jobs, &["automerge", "permissions"])?;
    let permissions_map = mapping(permissions)?;

    assert_eq!(permissions_map.len(), 5);
    assert_scalar_eq(mapping_value(permissions, "contents")?, "write");
    assert_scalar_eq(mapping_value(permissions, "pull-requests")?, "write");
    assert_scalar_eq(mapping_value(permissions, "checks")?, "read");
    assert_scalar_eq(mapping_value(permissions, "statuses")?, "read");
    assert_scalar_eq(mapping_value(permissions, "id-token")?, "write");

    Ok(())
}

#[test]
fn dependabot_automerge_keeps_expected_triggers() -> Result<(), Box<dyn Error>> {
    let workflow = read_workflow(WorkflowFile::DependabotAutomerge)?;
    let pull_request_target = mapping_value_path(&workflow.triggers, &["pull_request_target"])?;
    let workflow_dispatch = mapping_value_path(&workflow.triggers, &["workflow_dispatch"])?;

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

fn read_workflow(file: WorkflowFile) -> Result<Workflow, Box<dyn Error>> {
    let contents = fs::read_to_string(file.path())?;
    let workflow = serde_yaml::from_str(&contents)?;

    Ok(workflow)
}

fn named_step<'a>(steps: &'a [Value], name: StepName<'_>) -> Result<&'a Value, Box<dyn Error>> {
    steps
        .iter()
        .find(|step| scalar_value(mapping_value(step, "name").ok()) == Some(name.0))
        .ok_or_else(|| format!("missing workflow step named {:?}", name.0).into())
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

fn mapping_value_path<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, Box<dyn Error>> {
    path.iter()
        .try_fold(value, |acc, key| mapping_value(acc, key))
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
