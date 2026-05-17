param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path (Get-Location) "src"

& $Python -m extractor.main `
    data\raw\3b1b-videos `
    --output-dir data\processed\3b1b-videos `
    --source-repo 3b1b/videos `
    --max-workers 8 `
    --prompt-augmentations 1

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
