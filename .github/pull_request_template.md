<!-- Title: "[R] R3: short description" or "CONTRACT: ..." -->
**Task:** R?/F? (#issue)
**Lane:** R / F / CONTRACT

**What changed**
-

**How I checked it**
- [ ] `python -m pytest -q`
- [ ] `python scripts/validate_data.py data/seed/seed.jsonl` (+ any new data files)
- [ ] Touched only my lane's files (or this is a `CONTRACT:` PR and the other human approved it)
- [ ] No secrets, weights or files > 5 MB
