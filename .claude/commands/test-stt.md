---
description: Run the full realtime-stt test suite and report results
allowed-tools: [Bash]
---

# Test realtime-stt

Run all tests for the realtime-stt project and report results.

## Instructions

1. Make sure Python 3.11 is active via pyenv:
   ```
   pyenv local 3.11.9
   ```

2. Run the full test suite from the project root:
   ```
   python -m pytest tests/ -v
   ```

3. Parse the output and report:
   - Total passed / failed / skipped / error counts
   - Any failures or errors with full tracebacks
   - Any warnings worth noting
   - Final verdict: PASS or FAIL

Keep the report concise. List failures first if any. If all tests pass, confirm with the count and time taken.
