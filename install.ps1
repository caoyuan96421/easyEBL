<#
.SYNOPSIS
    Create the easyEBL Python virtual environment (.venv, next to this script)
    and install easyEBL with all of its dependencies.

.DESCRIPTION
    1. Finds a Python 3.10 - 3.13 interpreter (py launcher, PATH, or miniconda).
    2. Creates .venv in this folder. Unless -SyncVenv is given, the folder is
       marked "ignored" for Dropbox so thousands of package files are not synced.
    3. Installs easyEBL in editable mode with extras (default: all,test):
       numpy, scipy, gdstk, PySide6, pyqtgraph, nidaqmx, pytest, pytest-qt.
    4. Verifies the imports and reports whether the NI-DAQmx driver is present.

.PARAMETER Python
    Full path to a specific python.exe to build the venv from.
.PARAMETER Extras
    pip extras to install (default "all,test"; use "gui,gds" for a lean install).
.PARAMETER Recreate
    Delete an existing .venv and build it again.
.PARAMETER RunTests
    Run the test suite after installing.
.PARAMETER InstallNIDriver
    Download and install the NI-DAQmx driver via "python -m nidaqmx installdriver"
    (needs administrator rights; only required on the PC wired to the DAQ).
.PARAMETER SyncVenv
    Do not mark .venv as ignored for Dropbox.

.EXAMPLE
    .\install.ps1
.EXAMPLE
    .\install.ps1 -Recreate -RunTests
.EXAMPLE
    .\install.ps1 -Python "C:\Users\me\miniconda3\python.exe"
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Extras = "all,test",
    [switch]$Recreate,
    [switch]$RunTests,
    [switch]$InstallNIDriver,
    [switch]$SyncVenv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$VPy  = Join-Path $Venv "Scripts\python.exe"
$MinVer = [version]"3.10"
$MaxVer = [version]"3.13"

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Get-PythonVersion {
    param([string]$Exe, [string[]]$PreArgs)
    try {
        $out = & $Exe @PreArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return [version]($out | Select-Object -Last 1).Trim() }
    } catch { }
    return $null
}

function Find-Python {
    $candidates = @()
    if ($Python) {
        $candidates += ,@{ Exe = $Python; Pre = @() }
    } else {
        foreach ($v in @("3.12", "3.11", "3.13", "3.10")) { $candidates += ,@{ Exe = "py"; Pre = @("-$v") } }
        $candidates += ,@{ Exe = "python"; Pre = @() }
        foreach ($p in @("$env:USERPROFILE\miniconda3\python.exe", "$env:USERPROFILE\anaconda3\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
            if (Test-Path $p) { $candidates += ,@{ Exe = $p; Pre = @() } }
        }
    }
    foreach ($c in $candidates) {
        $ver = Get-PythonVersion -Exe $c.Exe -PreArgs $c.Pre
        if ($null -ne $ver) {
            if ($ver -ge $MinVer -and $ver -le $MaxVer) {
                Write-Host "Using Python $ver : $($c.Exe) $($c.Pre -join ' ')"
                return $c
            }
            Write-Host "Skipping Python $ver at $($c.Exe) $($c.Pre -join ' ') (need $MinVer - $MaxVer)"
        }
    }
    throw "No Python $MinVer - $MaxVer found. Install one from python.org (tick 'Add to PATH'), or pass -Python <path to python.exe>."
}

Push-Location $Root
try {
    Write-Step "easyEBL folder: $Root"

    # ------------------------------------------------------------ venv
    if ($Recreate -and (Test-Path $Venv)) {
        Write-Step "Removing existing .venv"
        Remove-Item -Recurse -Force $Venv
    }
    if (-not (Test-Path $VPy)) {
        $py = Find-Python
        Write-Step "Creating virtual environment in .venv"
        New-Item -ItemType Directory -Force -Path $Venv | Out-Null
        if (-not $SyncVenv) {
            # Dropbox: exclude the venv from syncing (harmless when Dropbox is not used)
            try {
                Set-Content -Path $Venv -Stream com.dropbox.ignored -Value 1
                Write-Host ".venv marked as ignored for Dropbox"
            } catch { Write-Host "(could not set Dropbox ignore flag: $($_.Exception.Message))" }
        }
        Invoke-Checked $py.Exe ($py.Pre + @("-m", "venv", $Venv))
    } else {
        Write-Step "Reusing existing .venv (use -Recreate to rebuild)"
    }

    # ------------------------------------------------------------ packages
    Write-Step "Upgrading pip / setuptools / wheel"
    Invoke-Checked $VPy @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    Write-Step "Installing easyEBL (editable) with extras [$Extras]"
    Invoke-Checked $VPy @("-m", "pip", "install", "-e", "${Root}[$Extras]")

    # ------------------------------------------------------------ checks
    Write-Step "Checking imports"
    $check = @"
import importlib, sys
import easyebl
print('easyEBL', easyebl.__version__, 'on Python', sys.version.split()[0])
for m in ['numpy', 'scipy', 'gdstk', 'PySide6', 'pyqtgraph', 'nidaqmx', 'pytest']:
    try:
        mod = importlib.import_module(m)
        print('  %-10s %s' % (m, getattr(mod, '__version__', 'ok')))
    except ImportError:
        print('  %-10s not installed (extras: $Extras)' % m)
try:
    import nidaqmx.system
    print('NI-DAQmx driver', nidaqmx.system.System.local().driver_version)
except Exception as e:
    print('NI-DAQmx driver not found - fine for the simulator; needed on the DAQ PC (%s)' % type(e).__name__)
"@
    Invoke-Checked $VPy @("-c", $check)

    if ($InstallNIDriver) {
        Write-Step "Installing the NI-DAQmx driver (administrator rights required)"
        Invoke-Checked $VPy @("-m", "nidaqmx", "installdriver")
    }

    if ($RunTests) {
        Write-Step "Running the test suite"
        $env:QT_QPA_PLATFORM = "offscreen"
        Invoke-Checked $VPy @("-m", "pytest", "-q")
        Remove-Item Env:\QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }

    Write-Step "Done"
    Write-Host @"
Activate the environment:   .\.venv\Scripts\Activate.ps1     (cmd: .venv\Scripts\activate.bat)
Start the demo (simulator):  .\run_demo.bat
Start on hardware:           .\run_demo.bat --backend nidaqmx --device Dev1
Run the tests:               .\.venv\Scripts\python.exe -m pytest -q
"@
}
finally {
    Pop-Location
}
