---
paths: []
---

# Documentation Update Rule

**Whenever you change code, always consider if documentation needs to be updated and do it in the same commit.**

## Process

Before committing any code changes, review and update:

1. **User-facing documentation:**
   - `README.md` - If usage, installation, or features change
   - `docs/cli.md` - If command-line interface changes
   - `docs/recipes.md` - If code examples or API usage changes
   - `docs/dataset-definition.md` - If dataset format changes

2. **Technical documentation:**
   - `docs/yogo-high-level.md` - If architecture or model changes
   - `dataset/CLAUDE.md` - If dataset metadata or statistics change
   - `dataset/README.md` - If annotation format changes

3. **Code documentation:**
   - Inline comments for complex logic
   - Docstrings for modified functions/classes
   - Type hints for function signatures

4. **Examples and tests:**
   - Update examples in `examples/` if they reference changed code
   - Ensure code examples in documentation still work
   - Update test cases if behavior changes

## Commit Together

Documentation updates should always be in the same commit as the code changes they document. The commit message should mention documentation updates.

## Example Commit Message

```
Add support for multi-channel input images

- Extend model to accept 1, 3, or 4 channel images
- Update data loader to handle different channel formats
- Add channel configuration to dataset definition

Documentation updates:
- Update README.md with channel configuration example
- Add multi-channel section to docs/dataset-definition.md
- Update docstrings in yogo/model.py and yogo/data.py

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Why This Matters

- Keeps documentation in sync with code
- Helps future developers (including future you) understand changes
- Makes code reviews easier
- Prevents confusion from outdated documentation
- Maintains project quality and professionalism
