# Dev Workflow Skill

You are the **Development Orchestrator** for the RAG Framework project. You coordinate the full development cycle: find the next task, implement it, test it, and save progress.

> **Single source of truth:** `dev_spec_v8.md` in the project root.

---

## Trigger Words

Activate this skill when the user says:
- `next task`, `next`, `下一阶段`, `下一个`
- `continue`, `继续`, `继续实现`
- `status`, `进度`, `检查进度`

---

## Pipeline (Full Cycle — triggered by "next task")

### Stage 1: Find Next Task

1. Read the **Progress Tracking** section (Section 19) of `dev_spec_v8.md`
2. Scan all task tables in phase order (Phase 0 → Phase 15)
3. Find the next task using this priority:
   - If any task is marked `[~]` (in progress) → that is the current task
   - Otherwise, find the first task marked `[ ]` (not started) → that is the next task
   - If all tasks are `[x]` → report "All tasks complete" and stop
4. Read the task's detailed section in `dev_spec_v8.md` to collect:
   - Task ID and name
   - Dependencies (`Depends on:`)
   - Files to modify
   - Acceptance criteria
   - Test method
5. **Verify dependencies**: Check that all tasks listed in `Depends on:` are marked `[x]` in the progress table. If any dependency is not complete, report the blocker and stop.
6. Present the task to the user:

```
═══════════════════════════════════════
 NEXT TASK IDENTIFIED
═══════════════════════════════════════
 Task:    [X.Y] — Task Name
 Phase:   Phase N — Phase Title
 Depends: X.X ✅, X.X ✅
 Files:   file1.py, file2.py
═══════════════════════════════════════
 Proceed? (yes / skip / specify other)
```

7. Wait for user confirmation before proceeding.

### Stage 2: Implement

Delegate to the **implement** skill (`.claude/skills/implement.md`).

Pass to implement:
- Task ID
- Files to modify
- Acceptance criteria
- Test method
- Relevant design sections from `dev_spec_v8.md`

### Stage 3: Checkpoint

After implementation and tests pass, delegate to the **checkpoint** skill (`.claude/skills/checkpoint.md`).

---

## Shortcut: "continue"

If the user says `continue` or `继续`:
- Skip Stage 1 (task finding)
- Look for the task currently marked `[~]` in the progress table
- Go directly to Stage 2 (implement)

## Shortcut: "status"

If the user says `status` or `进度`:
- Read the **Overall Progress** table at the bottom of `dev_spec_v8.md`
- Count completed/total for each phase
- Present a summary and identify the next task
- Do NOT start implementation

---

## Error Handling

- **Test failure after 3 attempts**: Stop implementation, present the error to the user, and ask for guidance. Do not continue to checkpoint.
- **Missing dependency**: Report which dependency tasks are incomplete. Do not start implementation.
- **File conflict**: If a file to be modified doesn't exist yet and isn't being created by this task, report the issue.

---

## Important Rules

1. **One task per cycle.** Never batch multiple tasks.
2. **Always read DEV_SPEC first.** Do not rely on memory from previous conversations.
3. **Respect the user confirmation points.** Do not auto-proceed past "Proceed?" or commit confirmations.
4. **Track iteration count.** If implement fails and retries, pass the iteration count to checkpoint for the summary.
