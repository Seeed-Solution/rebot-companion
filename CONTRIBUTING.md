# Contributing

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.10.

```bash
uv sync --all-extras          # SDK + [agent] (MCP server) + [calib] (OpenCV) + dev tools
```

## Checks

Run these before opening a pull request; CI (`.github/workflows/ci.yml`) runs the same on
Python 3.10 to 3.13.

```bash
uv run ruff check
uv run ruff format --check     # `uv run ruff format` to fix
uv run pytest -q
```

The tests run against in-process fakes (armd: `tests/fake_armd.py`) and recorded fixtures;
they do not need a robot arm or a camera. A change that affects arm motion should also be
tried on hardware; say in the pull request whether it was, and on which rebot-armd version.

If you change the kinematics port, regenerate the JS reference values from the rebot-armd
Web cockpit with `node tools/gen_web_fixtures.mjs <path-to>/rebot-armd/src/api/index.html`.

## Commits

- One logical change per commit; keep unrelated changes apart.
- Subject in English, imperative mood, following the existing history:
  `type(scope): summary`, for example `fix(camera): send a JSON body with the trigger POST`.
  Types in use: `feat`, `fix`, `test`, `docs`, `ci`, `chore`.
- Explain in the body why the change is needed when it is not obvious from the diff.

## License

By contributing you agree that your contributions are licensed under the MIT License
(see `LICENSE`).
