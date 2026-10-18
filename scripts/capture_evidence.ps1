# Produces the two terminal outputs the submission template asks you to screenshot (1.5).
# Run each block in a normal PowerShell window and take a screenshot (Win+Shift+S) of the terminal.
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "=== SCREENSHOT 1: end-to-end query (answer + citations + latency + chunk count + token usage) ==="
python -m app.cli ingest --path data/sample_documents | Out-Null
$env:LOG_LEVEL = "INFO"
# logs (JSON lines incl. per-query latency/chunks/tokens) go to stderr and print in the terminal:
python -m app.cli query "What is the maximum hotel cost per night in tier-1 cities?" -k 5

Write-Host "=== SCREENSHOT 2: idempotent re-ingest (vector count before/after) ==="
python scripts/demo_idempotency.py
