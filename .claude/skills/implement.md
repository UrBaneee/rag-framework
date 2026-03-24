# Implement Skill

You are the **Developer** for the RAG Framework project. Your job is to read the spec, write code, and make tests pass.

> **Single source of truth:** `dev_spec_v8.md` in the project root.

---

## Step 1: Read Spec and Plan

1. Read the task's detailed description in `dev_spec_v8.md` (search for `## Task X.Y`)
2. Read the relevant design sections:
   - Tasks in Phase 0–1 → Read Sections 13–15 (Architecture, Storage, Config)
   - Tasks in Phase 2 → Read Section 14 (Storage Design)
   - Tasks in Phase 3 → Read Sections 4–7 (Ingestion, IR, Cleaning, Chunking)
   - Tasks in Phase 4 → Read Section 9 (Incremental Ingestion, hashing)
   - Tasks in Phase 5–6 → Read Section 10 (Retrieval Pipeline)
   - Tasks in Phase 7 → Read Section 11 (Answer Generation)
   - Tasks in Phase 8 → Read Section 16 (Streamlit Studio)
   - Tasks in Phase 9 → Read Section 13 (System Architecture, MCP)
   - Tasks in Phase 10 → Read Section 12 (Evaluation Framework)
   - Tasks in Phase 11 → Read Section 9 (Incremental Ingestion)
   - Tasks in Phase 12 → Read Sections 7–8 (Chunking, Metadata Enrichment)
   - Tasks in Phase 13 → Read Section 4 (Ingestion, OCR formats)
   - Tasks in Phase 14 → Read Section 12.4 (Answer Quality)
   - Tasks in Phase 15 → Read Section 4.1 (External Connectors)
3. Extract from the task description:
   - **Files to modify** — exactly which files to create or change
   - **Acceptance criteria** — what "done" looks like
   - **Test method** — how to verify
   - **Depends on** — what already exists that you can build on
   - **Notes** — any warnings or special instructions
4. Present the plan to the user:

```
═══════════════════════════════════════
 IMPLEMENTATION PLAN
═══════════════════════════════════════
 Task: [X.Y] — Task Name

 Files to create/modify:
   + new_file.py (create)
   ~ existing_file.py (modify)

 Design principles applied:
   - [list relevant principles from spec]

 Approach:
   - [brief description of implementation]
═══════════════════════════════════════
```

---

## Step 2: Write Code

Follow these coding standards:

**Type hints required:**
```python
def embed(self, texts: list[str]) -> list[list[float]]:
```

**Google-style docstrings:**
```python
def embed(self, texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings into vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.

    Raises:
        ValueError: If texts is empty.
    """
```

**Additional rules:**
- No hardcoded values — use config/settings
- Clean imports — standard lib first, then third-party, then project
- Meaningful variable names
- Functions under 40 lines where possible
- All external calls (API, DB, file I/O) wrapped in try/except
- Abstract base classes use `ABC` and `@abstractmethod`
- Factory functions read from config and return the correct implementation

---

## Step 3: Write Tests

If the task's test method specifies pytest:

1. Create the test file in the correct directory:
   - `rag/core/` or `rag/infra/` code → `tests/test_<module>.py`
   - Pipeline code → `tests/test_<pipeline>.py`
   - E2E tests → `tests/e2e/test_<name>.py`

2. Follow test naming convention:
   ```python
   def test_<function>_<scenario>_<expected>():
   ```

3. Mock external dependencies (API calls, DB connections) in unit tests.

4. Every test must be independent — no shared mutable state between tests.

---

## Step 4: Run Tests

1. Run the specific test command from the task's **Test method**:
   ```bash
   pytest tests/test_<module>.py -v
   ```

2. If tests pass → proceed to report.

3. If tests fail → analyze the error:
   - Read the full error message and stack trace
   - Identify root cause
   - Fix the code
   - Re-run tests
   - **Maximum 3 iterations.** After 3 failures, stop and report to user.

---

## Step 5: Report Results

After tests pass (or after 3 failed iterations):

```
═══════════════════════════════════════
 IMPLEMENTATION COMPLETE ✅  (or FAILED ❌)
═══════════════════════════════════════
 Task: [X.Y] — Task Name

 Files modified:
   + rag/infra/stores/docstore_sqlite.py (created, 120 lines)
   + tests/test_docstore_sqlite.py (created, 85 lines)

 Tests:
   - Command: pytest tests/test_docstore_sqlite.py -v
   - Result: 8/8 passed
   - Iterations: 1

 Acceptance criteria check:
   ✅ Creates tables: documents, text_blocks, chunks
   ✅ Adds required indexes
   ✅ Initializes DB on first run
═══════════════════════════════════════
```

If failed:
```
═══════════════════════════════════════
 IMPLEMENTATION FAILED ❌
═══════════════════════════════════════
 Task: [X.Y] — Task Name
 Iterations: 3/3

 Last error:
   [error message and stack trace summary]

 Attempted fixes:
   1. [what was tried]
   2. [what was tried]
   3. [what was tried]

 Suggested next step:
   [recommendation for user]
═══════════════════════════════════════
```

---

## Important Rules

1. **Read the spec first, always.** Never assume you know the task requirements.
2. **Check existing code before writing.** Look at neighboring files for patterns (import style, class structure, naming).
3. **Don't overwrite unrelated code.** Only modify files listed in the task.
4. **Preserve existing tests.** New code must not break existing passing tests.
5. **Handle the `Notes` field.** If the task has a `Note:` section, it contains critical warnings — follow them.
