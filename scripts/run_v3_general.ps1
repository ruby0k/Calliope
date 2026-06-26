# Build the v3-general dataset (if missing) then retrain the general base.
# Run from the repo root. Laptop should be plugged in; uses the local GPU.
# Pass -Fresh to ignore/archive any existing checkpoint and train from scratch.
param([switch]$Fresh)
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'  # stream training output live to the console
$data = 'data/v3_general_bpe32k'

if (-not (Test-Path "$data/train.bin")) {
    Write-Host "[run_v3_general] building dataset -> $data (32K BPE; FineWeb 245M / Code 98M / Wiki 74M / SimpleStories 59M)"
    uv run python scripts/prepare_hf_mix.py --out-dir $data --vocab-size 32000 `
        --max-fineweb-tokens 245000000 --max-code-tokens 98000000 `
        --max-wikitext-tokens 74000000 --max-simplestories-tokens 59000000 `
        --max-tinystories-docs 0 --max-calliope-docs 0
    if ($LASTEXITCODE -ne 0) { Write-Host '[run_v3_general] prep FAILED; aborting before training'; exit 1 }
} else {
    Write-Host "[run_v3_general] dataset already present at $data, skipping build"
}

# Auto-resume: continue from last.pt unless -Fresh (which archives any existing checkpoint).
$ckptDir = 'checkpoints/calliope_100m_v3_general'
$expDir = 'experiments/Calliope-100M-v3-general-run001'
$last = "$ckptDir/last.pt"
if ($Fresh) {
    $stamp = Get-Date -Format yyyyMMdd_HHmmss
    if (Test-Path $ckptDir) { Write-Host "[run_v3_general] -Fresh: archiving checkpoint -> ${ckptDir}_old_$stamp"; Move-Item $ckptDir "${ckptDir}_old_$stamp" }
    if (Test-Path $expDir) { Write-Host "[run_v3_general] -Fresh: archiving metrics -> ${expDir}_old_$stamp"; Move-Item $expDir "${expDir}_old_$stamp" }
}
$trainArgs = @('-u', '-m', 'train.train', '--config', 'configs.calliope_100m_v3_general', '--run-name', 'Calliope-100M-v3-general-run001')
if ((Test-Path $last) -and (-not $Fresh)) {
    Write-Host "[run_v3_general] found $last - resuming from it"
    $trainArgs += @('--resume', $last)
} else {
    Write-Host '[run_v3_general] training from scratch'
}
Write-Host '[run_v3_general] training (60k iters, early-stop off, last.pt saved every 100 iters)...'
uv run python @trainArgs
Write-Host '[run_v3_general] done. Stop anytime (Ctrl+C); re-run this script to resume.'
