# weekly_pipeline.ps1  (item #25 — automated weekly pipeline)
#
# Runs the full post-race pipeline automatically.
# Designed to be triggered by Windows Task Scheduler every Monday morning.
#
# What it does:
#   1. Ingests any new sessions (new race weekend)
#   2. Fetches sector times via OpenF1
#   3. Rebuilds feature tables
#   4. Retrains both models
#   5. Batch evaluates all completed rounds for the current season
#   6. Generates the prediction for the next upcoming round
#   7. Logs everything to logs/pipeline_YYYY-MM-DD.log
#
# Setup (run once):
#   1. Open Task Scheduler (taskschd.msc)
#   2. Create Basic Task → name it "F1 Pipeline"
#   3. Trigger: Weekly, Monday, 09:00
#   4. Action: Start a program
#      Program:    powershell.exe
#      Arguments:  -ExecutionPolicy Bypass -File "C:\Users\raymo\f1-predictor\src\weekly_pipeline.ps1"
#      Start in:   C:\Users\raymo\f1-predictor
#   5. Finish
#
# Manual run:
#   cd C:\Users\raymo\f1-predictor
#   powershell -ExecutionPolicy Bypass -File src\weekly_pipeline.ps1

# ── Config ────────────────────────────────────────────────────────────────────

$ProjectRoot = "C:\Users\raymo\f1-predictor"
$PythonExe   = "C:\Users\raymo\f1env\Scripts\python.exe"
$LogDir      = "$ProjectRoot\logs"
$CurrentYear = (Get-Date).Year

# ── Setup ─────────────────────────────────────────────────────────────────────

Set-Location $ProjectRoot

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$LogFile = "$LogDir\pipeline_$(Get-Date -Format 'yyyy-MM-dd').log"

function Log-Message {
    param([string]$msg)
    $ts  = Get-Date -Format "HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Run-Step {
    param([string]$label, [string[]]$args)
    Log-Message "── $label ──────────────────────────────────────"
    $result = & $PythonExe @args 2>&1
    $result | ForEach-Object { Add-Content -Path $LogFile -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Log-Message "ERROR: $label failed (exit $LASTEXITCODE)"
        Log-Message "Check $LogFile for details."
        exit 1
    }
    Log-Message "$label complete."
}

# ── Pipeline ──────────────────────────────────────────────────────────────────

Log-Message "===== F1 Pipeline starting ====="
Log-Message "Year: $CurrentYear"
Log-Message "Log:  $LogFile"
Log-Message ""

# Step 1: Ingest new sessions
Run-Step "Ingest" "src\ingest.py"

# Step 2: Fetch sector times
Run-Step "Sector fetch (OpenF1)" "src\openf1.py", "--year", "$CurrentYear"

# Step 3: Rebuild features
Run-Step "Feature engineering" "src\features.py"

# Step 4: Retrain models
Run-Step "Model training" "src\train.py"

# Step 5: Batch evaluate current season
Log-Message "── Batch evaluate $CurrentYear ──────────────────────────────────────"
& $PythonExe "src\batch_evaluate.py" "--year" "$CurrentYear" 2>&1 |
    ForEach-Object { Add-Content -Path $LogFile -Value $_; Write-Host $_ }
Log-Message "Batch evaluate complete."

# Step 6: Find next upcoming round and generate prediction
Log-Message "── Finding next round ──────────────────────────────────────"
$next_round_script = @"
import fastf1, json
from datetime import datetime, timezone
from pathlib import Path

fastf1.Cache.enable_cache('cache')
sched = fastf1.get_event_schedule($CurrentYear, include_testing=False)
now   = datetime.now(timezone.utc)

for _, row in sched.iterrows():
    rnd  = int(row['RoundNumber'])
    for col in ('Session5Date', 'EventDate'):
        if col in row.index and str(row[col]) != 'NaT':
            try:
                dt = __import__('pandas').Timestamp(row[col])
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                if dt.to_pydatetime() > now:
                    print(rnd)
                    exit(0)
            except Exception:
                pass

print(-1)
"@

$next_round = & $PythonExe -c $next_round_script
$next_round = $next_round.Trim()

if ($next_round -eq "-1" -or $next_round -eq "") {
    Log-Message "No upcoming rounds found — season may be complete."
} else {
    Log-Message "Next round: R$next_round — generating prediction..."
    & $PythonExe "src\predict.py" "--year" "$CurrentYear" "--round" "$next_round" 2>&1 |
        ForEach-Object { Add-Content -Path $LogFile -Value $_; Write-Host $_ }
    Log-Message "Prediction saved → predictions\${CurrentYear}_R$('{0:D2}' -f [int]$next_round)_prediction.json"
}

# ── Done ──────────────────────────────────────────────────────────────────────

Log-Message ""
Log-Message "===== Pipeline complete ====="
Log-Message "Log saved → $LogFile"