# Build the v3-general dataset (if missing) then retrain the general base.
# Run from the repo root. Laptop should be plugged in; uses the local GPU.
$ErrorActionPreference = 'Continue'
$data = 'data/v3_general_gpt2'

if (-not (Test-Path "$data/train.bin")) {
    Write-Host "[run_v3_general] building dataset -> $data (FineWeb 245M / Code 98M / Wiki 74M / SimpleStories 59M)"
    uv run python scripts/prepare_hf_mix.py --out-dir $data `
        --max-fineweb-tokens 245000000 --max-code-tokens 98000000 `
        --max-wikitext-tokens 74000000 --max-simplestories-tokens 59000000 `
        --max-tinystories-docs 0 --max-calliope-docs 0
    if ($LASTEXITCODE -ne 0) { Write-Host '[run_v3_general] prep FAILED; aborting before training'; exit 1 }
} else {
    Write-Host "[run_v3_general] dataset already present at $data, skipping build"
}

# Auto-resume: if a last.pt exists, continue from it (same run-name appends metrics).
$last = 'checkpoints/calliope_100m_v3_general/last.pt'
$trainArgs = @('-m', 'train.train', '--config', 'configs.calliope_100m_v3_general', '--run-name', 'Calliope-100M-v3-general-run001')
if (Test-Path $last) {
    Write-Host "[run_v3_general] found $last - resuming from it"
    $trainArgs += @('--resume', $last)
} else {
    Write-Host '[run_v3_general] no checkpoint - training from scratch'
}
Write-Host '[run_v3_general] training (60k iters, early-stop off, last.pt saved every 100 iters)...'
uv run python @trainArgs
Write-Host '[run_v3_general] done. Stop anytime (Ctrl+C); re-run this script to resume.'
