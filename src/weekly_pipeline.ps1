# weekly_pipeline.ps1 (automated weekly pipeline with error recovery)
#
# Runs the full post-race pipeline automatically.
# Designed to be triggered by Windows Task Scheduler every Monday morning.

# -- Config --------------------------------------------------------------------

$ProjectRoot = "C:\Users\raymo\f1-predictor"
$PythonExe   = "python"
$LogDir      = "$ProjectRoot\logs"
$CurrentYear = (Get-Date).Year

# -- Setup ---------------------------------------------------------------------

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
    param(
        [string]$label,
        [string[]]$StepArgs,
        [bool]$critical = $true
    )
    Log-Message "-- $label --------------------------------------"
    # Using splatting with a custom name to avoid reserved variable conflicts
    $result = & $PythonExe $StepArgs 2>&1
    $result | ForEach-Object { Add-Content -Path $LogFile -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Log-Message "ERROR: $label failed (exit $LASTEXITCODE)"
        if ($critical) {
            Log-Message "CRITICAL: Aborting pipeline. Check $LogFile for details."
            exit 1
        } else {
            Log-Message "WARNING: Non-critical step failed - continuing pipeline."
            return $false
        }
    }
    Log-Message "$label complete."
    return $true
}

# -- Pipeline ------------------------------------------------------------------

$start_time = Get-Date
Log-Message "===== F1 Pipeline starting ====="
Log-Message "Year: $CurrentYear"
Log-Message "Log:  $LogFile"
Log-Message ""

# Step 1: Ingest new sessions (CRITICAL)
Run-Step "Ingest" @("src\ingest.py") -critical $true

# Step 2: Fetch sector times (non-critical - OpenF1 may be down)
Run-Step "Sector fetch (OpenF1)" @("src\openf1.py", "--year", "$CurrentYear") -critical $false

# Step 3: Rebuild features (CRITICAL)
Run-Step "Feature engineering" @("src\features.py") -critical $true

# Step 4: Retrain models (CRITICAL)
$TuneFlag = ""
$DayOfMonth = (Get-Date).Day
if ($DayOfMonth -le 7) {
    Log-Message "First week of the month detected - enabling automated hyperparameter tuning."
    $TuneFlag = "--tune"
}
Run-Step "Model training" @("src\train.py", $TuneFlag) -critical $true

# Step 5: Batch evaluate current season (non-critical)
Log-Message "-- Batch evaluate $CurrentYear --------------------------------------"
& $PythonExe "src\batch_evaluate.py" "--year" "$CurrentYear" 2>&1 |
    ForEach-Object { Add-Content -Path $LogFile -Value $_; Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
    Log-Message "WARNING: Batch evaluation failed - predictions still generated."
} else {
    Log-Message "Batch evaluate complete."
}

# Step 6: Find next upcoming round and generate prediction
Log-Message "-- Finding next round --------------------------------------"
# Using a single-quoted here-string to avoid PS interpolation issues
$next_round_script = @'
import fastf1, sys
from datetime import datetime, timezone

try:
    year = int(sys.argv[1])
    fastf1.Cache.enable_cache('cache')
    sched = fastf1.get_event_schedule(year, include_testing=False)
    now   = datetime.now(timezone.utc)

    next_rnd = -1
    for _, row in sched.iterrows():
        rnd  = int(row['RoundNumber'])
        # Check both Session5Date (Race) and EventDate
        for col in ('Session5Date', 'EventDate'):
            if col in row.index and str(row[col]) != 'NaT':
                try:
                    dt = __import__('pandas').Timestamp(row[col])
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                    if dt.to_pydatetime() > now:
                        next_rnd = rnd
                        break
                except Exception:
                    pass
        if next_rnd != -1:
            break
    print(next_rnd)
except Exception as e:
    print(-1)
'@

$next_round = & $PythonExe -c $next_round_script $CurrentYear
$next_round = $next_round.Trim()

if ($next_round -eq "-1" -or $next_round -eq "") {
    Log-Message "No upcoming rounds found - season may be complete."
} else {
    Log-Message "Next round: R$next_round - generating prediction..."
    & $PythonExe "src\predict.py" "--year" "$CurrentYear" "--round" "$next_round" 2>&1 |
        ForEach-Object { Add-Content -Path $LogFile -Value $_; Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        Log-Message "WARNING: Prediction generation failed for R$next_round."
    } else {
        $formatted_round = "{0:D2}" -f [int]$next_round
        Log-Message "Prediction saved -> predictions\${CurrentYear}_R${formatted_round}_prediction.json"
        
        Log-Message "Running Monte Carlo Simulation..."
        & $PythonExe "src\simulate.py" "--year" "$CurrentYear" "--round" "$next_round" 2>&1 |
            ForEach-Object { Add-Content -Path $LogFile -Value $_; Write-Host $_ }
        if ($LASTEXITCODE -ne 0) {
            Log-Message "WARNING: Simulation failed for R$next_round."
        } else {
            Log-Message "Simulations saved -> predictions\${CurrentYear}_R${formatted_round}_simulations.json"
        }
    }
}

# -- Summary -------------------------------------------------------------------

$elapsed = (Get-Date) - $start_time
Log-Message ""
Log-Message "===== Pipeline complete ====="
Log-Message "Duration: $($elapsed.ToString('hh\:mm\:ss'))"
Log-Message "Log saved -> $LogFile"