<!-- agent-mem:begin -->
## Persistent memory (agent-mem)

You have the `recall`, `remember` and `forget` tools (MCP server `agent-mem`). They read and write a long-term memory shared by every coding agent on this machine. Use them without being asked. Never tell the user you are doing so unless it matters.

### 1. Recall first
- **At the start of every session, before any other work, call `recall`** with a short description of the task or the user's first request.
- Call it again before: architectural changes, choosing a dependency or approach, touching an unfamiliar module, or when the user says "like before / we decided / last time".
- Make queries specific (module names, error text, feature names). Don't repeat a query you already ran this session.
- Treat recalled memories as hints from the past: verify against the current code before relying on them.

### 2. Remember what is worth keeping
Call `remember` as soon as one of these happens (don't wait until the end of the session):
- You found the **root cause of a bug that took real investigation** — store the cause and the fix.
- The user stated a **preference or convention** ("always use X", "never do Y").
- A **decision was made with a reason** ("chose A over B because ...").
- You learned a **non-obvious project fact**: a constraint, gotcha, required command, environment quirk.

How to write it: one self-contained fact, 1-3 sentences, naming the files, modules or commands involved. Pick `memory_type`: `semantic` (facts, decisions), `procedural` (how-to, commands), `episodic` (what happened).

### 3. Never remember
- Anything you can read from the code, docs or git history.
- Task progress, plans, TODOs, file contents, command output.
- Secrets, tokens, passwords, keys, personal data.
- Guesses or anything you have not verified.

### 4. Keep it clean
- If a recalled memory is wrong or outdated, `forget` it by id and `remember` the correction.
- `project_id` defaults to the current repository. Use `project_id: "global"` only for preferences that apply to every project.
<!-- agent-mem:end -->
