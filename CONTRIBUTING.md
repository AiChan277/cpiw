# Contributing to PrivacyCam

First off, thanks for taking the time to contribute!

## Workflow
1. Fork the repo.
2. Create a new branch: `git checkout -b feature/my-new-feature` or `bugfix/issue-number`.
3. Make your changes.
4. Test thoroughly.
5. Commit with descriptive messages.
6. Push to your fork and submit a Pull Request.

## Testing & Linting
Ensure all code passes `ruff`, `mypy`, and `pytest`.
```bash
pip install -r requirements-dev.txt
pytest
ruff check src/
mypy src/
```

## Pull Request Process
- Include context and rationale in your PR description.
- Ensure backwards compatibility.
- Ensure CI checks pass.
- Request reviews from maintainers.
