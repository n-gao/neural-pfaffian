# Neural Pfaffian

## Dealing with implementation issues

- Do not silently introduce hacks, temporary workarounds, or test-only bypasses when a failure points to a deeper issue in the codebase.
- If your implementation fails because of behavior elsewhere, stop and report the problem before spending significant time on workarounds.
- Present:
  - the observed failure,
  - the likely root cause,
  - the principled fix,
  - any short-term workaround,
  - the tradeoff between them.
- Let the user decide which path to take before implementing the workaround, especially for JAX/JIT/static-metadata issues, cache-order-dependent failures, numerical instability, or API drift.

## Development hints

- Install deps: `uv sync --dev`
  - Do not create your own cache directory; use mine
  - If possible, simply use the local venv under `.venv`
- Run tools via uv, i.e. `uv run [pyright | ruff | ...]`
- For linting and formatting use `ruff`
  - All code that you produce must pass `ruff` checks.
- For type checking use `pyright`
  - If type issues are present before making changes, do not refactor code to fix them, without being asked to.
  - In code that you have touched, make the type checker happy.
- Also verify whether the code you have produced is correct by using `python compile`.
- In case you run into permission problems executing those tools, try running them with escalated permissions.
- If you want to execute some Python code to test specific things, please write that code into a file called ".agentic_test_code.py" and pipe that into the Python interpreter instead of piping Python code from the command line into the interpreter.

## Codestyle and JAX-specific hints

- Since we usually JIT everything that is possible it is very important to be careful of compile-time static values. When writing code, always try to understand whether it is truly necessary to leave the "numpy-domain", i.e., whether the arrays you are computing change from step to step. If possible, keep things that should not be considered data, but rather structure, compile-time static.
