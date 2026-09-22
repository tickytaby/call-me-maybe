# Failure modes not covered by try/except

Audit of `src/__main__.py`. Each item is a way the script crashes (or hangs)
with an unhandled exception in production, even though the file-loading code
around it looks defensive.

## Data loaded from files, used without validation

All items in this section are resolved.

## Model / runtime

All items in this section are resolved.
