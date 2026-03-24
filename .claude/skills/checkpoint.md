# Checkpoint Skill

You are the **Progress Tracker** for the RAG Framework project. After a task is implemented and tested, you summarize the work, update the DEV_SPEC progress tracking, and optionally commit.

> **Single source of truth:** `dev_spec_v8.md` in the project root.

---

## Step 1: Generate Work Summary

Collect information from the implementation stage and present it to the user:

```
═══════════════════════════════════════
 WORK SUMMARY
═══════════════════════════════════════
 Task:       [X.Y] — Task Name
 Phase:      Phase N — Phase Title

 Files changed:
   + rag/infra/stores/docstore_sqlite.py (created)
   + tests/test_docstore_sqlite.py (created)
   ~ pyproject.toml (modified)

 Tests:
   - 8/8 passed
   - Iterations: 1

 Notes:
   [any relevant observations]
═══════════════════════════════════════
 Is this summary accurate? (confirm / revise)
```

**Wait for user confirmation.** If the user says "revise", ask what needs correction and regenerate.

---

## Step 2: Update DEV_SPEC Progress

After user confirms the summary, update `dev_spec_v8.md`:

### 2a. Update the task row in the Progress Tracking table

Find the task row in **Section 19** and update:

Before:
```
| X.Y | Task Name | [ ] | - | |
```

After:
```
| X.Y | Task Name | [x] | YYYY-MM-DD | brief note about what was done |
```

Rules:
- Use today's date for the completion date
- Keep the note concise (under 60 characters)
- Match the existing format — do not add extra columns or change alignment

### 2b. Update the Overall Progress table

Find the row for the task's phase and increment the "Completed" count by 1. Recalculate the percentage.

Before:
```
| Phase N | 6 | 2 | 33% |
```

After:
```
| Phase N | 6 | 3 | 50% |
```

Also update the **Total** row at the bottom.

### 2c. If the task was in-progress, mark it complete

If the task was previously `[~]`, change it to `[x]`.

---

## Step 3: Prepare Commit

Generate a structured commit message:

```
<type>(<scope>): [Phase X.Y] brief description

Completed Task X.Y: Task Name

Changes:
- Added <component> implementation
- Added unit tests

Testing:
- Command: pytest tests/test_xxx.py -v
- Results: N/N passed

Refs: dev_spec_v8.md Task X.Y
```

Commit types:
- `feat` — new feature or capability
- `test` — adding or updating tests
- `refactor` — restructuring without behavior change
- `docs` — documentation only
- `chore` — project setup, config, tooling

Scope examples: `core`, `infra`, `pipeline`, `cli`, `studio`, `mcp`, `eval`

Present to user:

```
═══════════════════════════════════════
 COMMIT READY
═══════════════════════════════════════
 Message:
   feat(infra): [Phase 2.1] implement DocStore schema

   Completed Task 2.1: Implement SQLite DocStore schema creation

   Changes:
   - Added docstore_sqlite.py with schema init
   - Added test_docstore_sqlite.py

   Testing:
   - Command: pytest tests/test_docstore_sqlite.py -v
   - Results: 8/8 passed

   Refs: dev_spec_v8.md Task 2.1

 Staged files:
   rag/infra/stores/docstore_sqlite.py
   tests/test_docstore_sqlite.py
   dev_spec_v8.md
═══════════════════════════════════════
 Commit now? (yes / no)
```

**Wait for user confirmation.**

- If "yes" → run `git add` for the relevant files + `dev_spec_v8.md`, then `git commit` with the message.
- If "no" → end the workflow. User can commit manually later.

---

## Important Rules

1. **Always update `dev_spec_v8.md`.** This happens regardless of whether the user chooses to commit. The progress update is separate from the git commit.
2. **Two confirmation points.** Summary confirmation (Step 1) and commit confirmation (Step 3). Never skip either.
3. **Atomic updates.** Update exactly one task per checkpoint. Never batch-update multiple tasks.
4. **Include `dev_spec_v8.md` in the commit.** The progress tracking update is part of the deliverable.
5. **Do not modify task descriptions.** Only modify the progress tracking table rows and the overall progress table. Never change acceptance criteria, dependencies, or file lists.
