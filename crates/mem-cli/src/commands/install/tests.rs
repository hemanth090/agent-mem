//! Unit tests for client registration.
use super::*;

const EXE: &str = "/opt/bin/agent-mem";

fn opts() -> Options {
    Options {
        dry_run: false,
        uninstall: false,
        force: false,
        timestamp: 111,
    }
}

fn setup() -> (tempfile::TempDir, PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let home = d.path().to_path_buf();
    (d, home)
}

fn run_all(home: &Path, o: &Options) -> Vec<Outcome> {
    run(home, Path::new(EXE), &ClientKind::ALL, o)
}

fn read_json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn skips_undetected_clients() {
    let (_d, home) = setup();
    // PATH may contain real binaries on a dev machine; only assert on file effects.
    let out = run_all(&home, &opts());
    for o in out {
        if let Status::Skipped(_) = o.status {
            continue;
        }
        // detected through PATH -> must have been created under the temp home
        assert!(o.path.unwrap().starts_with(&home));
    }
}

#[test]
fn claude_fresh_and_existing_preserves_keys_and_order() {
    let (_d, home) = setup();
    let p = home.join(".claude.json");
    std::fs::write(
        &p,
        r#"{"zeta":1,"mcpServers":{"other":{"command":"x"}},"alpha":{"k":[1,2]}}"#,
    )
    .unwrap();
    let out = run(&home, Path::new(EXE), &[ClientKind::Claude], &opts());
    assert_eq!(out[0].status, Status::Added);
    let v = read_json(&p);
    assert_eq!(v["mcpServers"]["other"]["command"], "x");
    assert_eq!(v["mcpServers"]["agent-mem"]["command"], EXE);
    assert_eq!(v["mcpServers"]["agent-mem"]["args"], json!(["mcp"]));
    assert_eq!(v["mcpServers"]["agent-mem"]["type"], "stdio");
    assert_eq!(v["alpha"]["k"], json!([1, 2]));
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["zeta", "mcpServers", "alpha"]);
    assert!(home.join(".claude.json.bak.111").exists());
}

#[test]
fn idempotent_second_run_writes_nothing() {
    let (_d, home) = setup();
    std::fs::write(home.join(".claude.json"), "{}").unwrap();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    std::fs::create_dir_all(home.join(".config/opencode")).unwrap();
    std::fs::create_dir_all(home.join(".gemini/antigravity")).unwrap();
    let first = run_all(&home, &opts());
    assert!(first.iter().all(|o| o.status == Status::Added), "{first:?}");
    let before: Vec<String> = [
        ".claude.json",
        ".codex/config.toml",
        ".config/opencode/opencode.json",
        ".gemini/antigravity/mcp_config.json",
    ]
    .iter()
    .map(|f| std::fs::read_to_string(home.join(f)).unwrap())
    .collect();
    let second = run(
        &home,
        Path::new(EXE),
        &ClientKind::ALL,
        &Options {
            timestamp: 222,
            ..opts()
        },
    );
    assert!(
        second.iter().all(|o| o.status == Status::Unchanged),
        "{second:?}"
    );
    assert!(!home.join(".claude.json.bak.222").exists());
    let after: Vec<String> = [
        ".claude.json",
        ".codex/config.toml",
        ".config/opencode/opencode.json",
        ".gemini/antigravity/mcp_config.json",
    ]
    .iter()
    .map(|f| std::fs::read_to_string(home.join(f)).unwrap())
    .collect();
    assert_eq!(before, after);
}

#[test]
fn opencode_shape_and_schema_on_new_file() {
    let (_d, home) = setup();
    std::fs::create_dir_all(home.join(".config/opencode")).unwrap();
    run(&home, Path::new(EXE), &[ClientKind::Opencode], &opts());
    let v = read_json(&home.join(".config/opencode/opencode.json"));
    assert_eq!(v["$schema"], "https://opencode.ai/config.json");
    assert_eq!(v["mcp"]["agent-mem"]["type"], "local");
    assert_eq!(v["mcp"]["agent-mem"]["command"], json!([EXE, "mcp"]));
    assert_eq!(v["mcp"]["agent-mem"]["enabled"], true);
}

#[test]
fn codex_toml_preserves_comments_and_other_servers() {
    let (_d, home) = setup();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    let p = home.join(".codex/config.toml");
    std::fs::write(
        &p,
        "# my codex config\nmodel = \"gpt-5\"\n\n[mcp_servers.other]\ncommand = \"npx\"\nargs = [\"-y\", \"foo\"]\n",
    )
    .unwrap();
    let out = run(&home, Path::new(EXE), &[ClientKind::Codex], &opts());
    assert_eq!(out[0].status, Status::Added);
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(text.contains("# my codex config"));
    assert!(text.contains("model = \"gpt-5\""));
    let doc: toml_edit::DocumentMut = text.parse().unwrap();
    assert_eq!(doc["mcp_servers"]["other"]["command"].as_str(), Some("npx"));
    assert_eq!(
        doc["mcp_servers"]["agent-mem"]["command"].as_str(),
        Some(EXE)
    );
    assert_eq!(
        doc["mcp_servers"]["agent-mem"]["args"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    // re-run is a no-op; changed exe path is an update
    let again = run(&home, Path::new(EXE), &[ClientKind::Codex], &opts());
    assert_eq!(again[0].status, Status::Unchanged);
    let moved = run(
        &home,
        Path::new("/new/agent-mem"),
        &[ClientKind::Codex],
        &opts(),
    );
    assert_eq!(moved[0].status, Status::Updated);
    let doc: toml_edit::DocumentMut = std::fs::read_to_string(&p).unwrap().parse().unwrap();
    assert_eq!(
        doc["mcp_servers"]["agent-mem"]["command"].as_str(),
        Some("/new/agent-mem")
    );
}

#[test]
fn codex_creates_file_when_dir_exists() {
    let (_d, home) = setup();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    run(&home, Path::new(EXE), &[ClientKind::Codex], &opts());
    let text = std::fs::read_to_string(home.join(".codex/config.toml")).unwrap();
    assert!(text.contains("[mcp_servers.agent-mem]"), "{text}");
}

#[test]
fn antigravity_prefers_existing_file_and_updates_all_existing() {
    let (_d, home) = setup();
    std::fs::create_dir_all(home.join(".gemini/config")).unwrap();
    std::fs::create_dir_all(home.join(".gemini/antigravity")).unwrap();
    std::fs::write(
        home.join(".gemini/antigravity/mcp_config.json"),
        r#"{"mcpServers":{}}"#,
    )
    .unwrap();
    let out = run(&home, Path::new(EXE), &[ClientKind::Antigravity], &opts());
    assert_eq!(out.len(), 1);
    assert!(!home.join(".gemini/config/mcp_config.json").exists());
    let v = read_json(&home.join(".gemini/antigravity/mcp_config.json"));
    assert_eq!(v["mcpServers"]["agent-mem"]["command"], EXE);

    // both exist -> both updated
    std::fs::write(home.join(".gemini/config/mcp_config.json"), "{}").unwrap();
    let out = run(
        &home,
        Path::new("/other/exe"),
        &[ClientKind::Antigravity],
        &opts(),
    );
    assert_eq!(out.len(), 2);
}

#[test]
fn antigravity_creates_requested_path_when_only_gemini_dir_exists() {
    let (_d, home) = setup();
    std::fs::create_dir_all(home.join(".gemini")).unwrap();
    let out = run(&home, Path::new(EXE), &[ClientKind::Antigravity], &opts());
    assert_eq!(out[0].status, Status::Added);
    assert_eq!(
        out[0].path.as_deref(),
        Some(home.join(".gemini/config/mcp_config.json").as_path())
    );
    assert!(home.join(".gemini/config/mcp_config.json").exists());
}

#[test]
fn invalid_files_are_left_untouched() {
    let (_d, home) = setup();
    std::fs::write(home.join(".claude.json"), "{ not json").unwrap();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    std::fs::write(home.join(".codex/config.toml"), "this = = bad").unwrap();
    let out = run(
        &home,
        Path::new(EXE),
        &[ClientKind::Claude, ClientKind::Codex],
        &opts(),
    );
    assert!(out.iter().all(|o| o.status.is_failure()), "{out:?}");
    assert_eq!(
        std::fs::read_to_string(home.join(".claude.json")).unwrap(),
        "{ not json"
    );
    assert_eq!(
        std::fs::read_to_string(home.join(".codex/config.toml")).unwrap(),
        "this = = bad"
    );
    assert!(!home.join(".claude.json.bak.111").exists());
}

#[test]
fn wrong_shaped_section_is_refused() {
    let (_d, home) = setup();
    std::fs::write(home.join(".claude.json"), r#"{"mcpServers":[1]}"#).unwrap();
    let out = run(&home, Path::new(EXE), &[ClientKind::Claude], &opts());
    assert!(out[0].status.is_failure());
}

#[test]
fn dry_run_writes_nothing() {
    let (_d, home) = setup();
    std::fs::write(home.join(".claude.json"), "{}").unwrap();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    let o = Options {
        dry_run: true,
        ..opts()
    };
    let out = run(
        &home,
        Path::new(EXE),
        &[ClientKind::Claude, ClientKind::Codex],
        &o,
    );
    assert!(matches!(out[0].status, Status::WouldAdd(_)));
    assert!(matches!(out[1].status, Status::WouldAdd(_)));
    assert_eq!(
        std::fs::read_to_string(home.join(".claude.json")).unwrap(),
        "{}"
    );
    assert!(!home.join(".codex/config.toml").exists());
    assert!(!home.join(".claude.json.bak.111").exists());
    assert!(describe(&out[0]).contains("agent-mem"));
}

#[test]
fn uninstall_removes_only_our_entry() {
    let (_d, home) = setup();
    std::fs::write(
        home.join(".claude.json"),
        r#"{"mcpServers":{"other":{"command":"x"},"agent-mem":{"command":"y"}}}"#,
    )
    .unwrap();
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    std::fs::write(
        home.join(".codex/config.toml"),
        "# keep\n[mcp_servers.other]\ncommand = \"a\"\n[mcp_servers.agent-mem]\ncommand = \"b\"\n",
    )
    .unwrap();
    let o = Options {
        uninstall: true,
        ..opts()
    };
    let out = run(
        &home,
        Path::new(EXE),
        &[ClientKind::Claude, ClientKind::Codex],
        &o,
    );
    assert_eq!(out[0].status, Status::Removed);
    assert_eq!(out[1].status, Status::Removed);
    let v = read_json(&home.join(".claude.json"));
    assert!(v["mcpServers"].get("agent-mem").is_none());
    assert_eq!(v["mcpServers"]["other"]["command"], "x");
    let t = std::fs::read_to_string(home.join(".codex/config.toml")).unwrap();
    assert!(t.contains("# keep") && t.contains("[mcp_servers.other]") && !t.contains("agent-mem"));
    let again = run(
        &home,
        Path::new(EXE),
        &[ClientKind::Claude, ClientKind::Codex],
        &o,
    );
    assert!(again.iter().all(|o| o.status == Status::NotInstalled));
}

#[test]
fn uninstall_with_no_files_is_skipped() {
    let (_d, home) = setup();
    let o = Options {
        uninstall: true,
        ..opts()
    };
    let out = run(&home, Path::new(EXE), &ClientKind::ALL, &o);
    assert!(out.iter().all(|o| matches!(o.status, Status::Skipped(_))));
}

#[test]
fn instruction_block_written_idempotently() {
    let (_d, home) = setup();
    let p = home.join("AGENTS.md");
    std::fs::write(&p, "# Existing\n").unwrap();
    assert!(write_instructions(&p, true, 1).unwrap().contains("would"));
    assert_eq!(std::fs::read_to_string(&p).unwrap(), "# Existing\n");
    write_instructions(&p, false, 1).unwrap();
    let first = std::fs::read_to_string(&p).unwrap();
    assert!(first.starts_with("# Existing") && first.contains("agent-mem:begin"));
    assert!(write_instructions(&p, false, 2)
        .unwrap()
        .contains("up to date"));
    assert_eq!(std::fs::read_to_string(&p).unwrap(), first);
}

#[test]
fn force_configures_undetected_clients() {
    let (_d, home) = setup();
    let o = Options {
        force: true,
        ..opts()
    };
    let out = run(
        &home,
        Path::new(EXE),
        &[ClientKind::Claude, ClientKind::Codex, ClientKind::Opencode],
        &o,
    );
    assert!(out.iter().all(|x| x.status == Status::Added), "{out:?}");
    assert!(home.join(".codex/config.toml").exists());
    assert!(home.join(".config/opencode/opencode.json").exists());
}

#[test]
fn binary_copy_is_atomic_idempotent_and_executable() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("downloaded-agent-mem");
    std::fs::write(&src, b"#!/bin/sh\necho v1\n").unwrap();
    let dst = stable_exe(&d.path().join("data"));
    assert_eq!(
        install_binary(&src, &dst, true).unwrap(),
        CopyStatus::WouldCopy
    );
    assert!(!dst.exists());
    assert_eq!(
        install_binary(&src, &dst, false).unwrap(),
        CopyStatus::Copied
    );
    assert_eq!(std::fs::read(&dst).unwrap(), std::fs::read(&src).unwrap());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&dst).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
    assert_eq!(install_binary(&src, &dst, false).unwrap(), CopyStatus::Same);
    // running from the destination itself is a no-op
    assert_eq!(install_binary(&dst, &dst, false).unwrap(), CopyStatus::Same);
    // an upgrade replaces it
    std::fs::write(&src, b"#!/bin/sh\necho v2\n").unwrap();
    assert_eq!(
        install_binary(&src, &dst, false).unwrap(),
        CopyStatus::Copied
    );
    assert_eq!(std::fs::read(&dst).unwrap(), b"#!/bin/sh\necho v2\n");
    assert!(!dst.with_extension("new").exists());
}
