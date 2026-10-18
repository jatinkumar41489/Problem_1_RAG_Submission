# Runs everything end-to-end on Windows PowerShell and stores real outputs under results/ and screenshots/.
# Prereq: venv active, `pip install -r requirements.txt`, .env with GROQ_API_KEY (or $env:GROQ_API_KEY set).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m app.cli verify --list-models
python -m app.cli ingest --path data/sample_documents
python -m app.cli calibrate                      # copy 'recommended_min_relevance_score' into .env (MIN_RELEVANCE_SCORE)
python -m app.cli evaluate                       # retrieval + answers + latency + chunking + cost -> results/
python -m pytest tests -v -o addopts="" 2>&1 | Tee-Object -FilePath results/test_results.txt
Write-Host "Done. See results/evaluation_summary.md"
