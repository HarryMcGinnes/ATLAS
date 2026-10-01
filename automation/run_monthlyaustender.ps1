$ErrorActionPreference = "Stop"

# ============================================================
# CONFIG
# ============================================================

$RepoRoot = "C:\Users\harry.mcginnes\ATLAS"

$PythonScript = Join-Path `
    $RepoRoot `
    "pipeline\fetch_financial_year_analysis.py"

$MonthlyDir = Join-Path `
    $RepoRoot `
    "data\austender\monthly"

$LogDir = Join-Path `
    $RepoRoot `
    "automation\logs"

# ============================================================
# SETUP
# ============================================================

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDir `
    | Out-Null

$Now = Get-Date

$LogFile = Join-Path `
    $LogDir `
    ("monthly_refresh_" + $Now.ToString("yyyy-MM-dd_HH-mm-ss") + ".log")

function Write-Log {
    param(
        [string]$Message
    )

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $Line = "[$Timestamp] $Message"

    Write-Host $Line

    Add-Content `
        -Path $LogFile `
        -Value $Line
}

# ============================================================
# CALCULATE PREVIOUS MONTH
# ============================================================

$FirstOfThisMonth = Get-Date `
    -Year $Now.Year `
    -Month $Now.Month `
    -Day 1

$PreviousMonthDate = $FirstOfThisMonth.AddMonths(-1)

$Year = $PreviousMonthDate.Year

$Month = $PreviousMonthDate.Month

$Period = "{0:D4}-{1:D2}" -f $Year, $Month

$ExpectedCsv = Join-Path `
    $MonthlyDir `
    ("financial_year_analysis_" + $Period + ".csv")

Write-Log "Starting ATLAS monthly AusTender refresh."

Write-Log "Detected previous month: $Period"

Write-Log "Expected output: $ExpectedCsv"

# ============================================================
# MOVE TO REPO
# ============================================================

Set-Location $RepoRoot

# ============================================================
# CHECK GIT
# ============================================================

Write-Log "Checking Git repository."

git status --short | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "Git repository check failed."
}

# ============================================================
# PULL LATEST CHANGES
# ============================================================

Write-Log "Pulling latest changes from GitHub."

git pull --rebase origin main

if ($LASTEXITCODE -ne 0) {
    throw "git pull failed."
}

# ============================================================
# RUN AUSTENDER FETCH
# ============================================================

Write-Log "Running AusTender Financial Year Analysis fetch."

python `
    $PythonScript `
    --year $Year `
    --month $Month `
    --headless

if ($LASTEXITCODE -ne 0) {
    throw "AusTender fetch script failed."
}

# ============================================================
# VALIDATE EXPECTED FILE EXISTS
# ============================================================

Write-Log "Checking downloaded CSV."

if (-not (Test-Path $ExpectedCsv)) {
    throw "Expected CSV was not created: $ExpectedCsv"
}

$FileInfo = Get-Item $ExpectedCsv

if ($FileInfo.Length -le 0) {
    throw "Downloaded CSV is empty."
}

Write-Log (
    "CSV created successfully. Size: " +
    [Math]::Round($FileInfo.Length / 1MB, 2) +
    " MB"
)

# ============================================================
# VALIDATE CSV STRUCTURE
# ============================================================

Write-Log "Validating CSV structure."

$ValidationScript = @"
import pandas as pd
from pathlib import Path

path = Path(r"$ExpectedCsv")

df = pd.read_csv(
    path,
    dtype=str,
    low_memory=False,
    encoding="utf-8-sig"
)

required = {
    "01. Agency Name",
    "02. Contract Type",
    "03. Contract Notice ID",
    "15. Category Type",
    "29. Supplier Name",
    "35. Agency Division",
    "36. Agency Branch",
    "48. Value",
}

missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(
        "Missing required columns: " +
        ", ".join(missing)
    )

if len(df.columns) != 48:
    raise SystemExit(
        f"Expected 48 columns, got {len(df.columns)}"
    )

if len(df) == 0:
    raise SystemExit(
        "CSV contains zero rows."
    )

print(
    f"CSV validation PASS: "
    f"{len(df):,} rows, "
    f"{len(df.columns)} columns"
)
"@

$ValidationScript | python -

if ($LASTEXITCODE -ne 0) {
    throw "CSV validation failed."
}

Write-Log "CSV validation passed."

# ============================================================
# CHECK IF FILE CHANGED
# ============================================================

Write-Log "Checking Git status for monthly CSV."

git add -- $ExpectedCsv

$Staged = git diff --cached --name-only

if (-not $Staged) {
    Write-Log "No new monthly CSV changes detected."
    Write-Log "Nothing to commit."
    exit 0
}

# ============================================================
# COMMIT
# ============================================================

$CommitMessage = (
    "Add AusTender FYA monthly export " +
    $Period
)

Write-Log "Creating Git commit: $CommitMessage"

git commit `
    -m $CommitMessage

if ($LASTEXITCODE -ne 0) {
    throw "git commit failed."
}

# ============================================================
# PUSH
# ============================================================

Write-Log "Pushing monthly CSV to GitHub."

git push origin main

if ($LASTEXITCODE -ne 0) {
    throw "git push failed."
}

# ============================================================
# FINISH
# ============================================================

Write-Log "ATLAS monthly AusTender refresh complete."

Write-Log (
    "Uploaded: " +
    "data/austender/monthly/" +
    "financial_year_analysis_" +
    $Period +
    ".csv"
)