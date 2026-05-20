$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"
$LauncherPath = Join-Path $ProjectRoot "run_labelmate.bat"
$IconPng = Join-Path $ProjectRoot "docs\labelmate_icon.png"
$IconIco = Join-Path $ProjectRoot "docs\labelmate_icon.ico"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-Python {
    $commands = @(
        @("py", "-3"),
        @("python", "")
    )

    foreach ($cmd in $commands) {
        $name = $cmd[0]
        $arg = $cmd[1]
        $exists = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $exists) {
            if ([string]::IsNullOrWhiteSpace($arg)) {
                return $name
            }
            return "$name $arg"
        }
    }

    throw "Python was not found. Install Python 3.9+ from https://www.python.org/downloads/ and enable 'Add python.exe to PATH'."
}

function Convert-PngToIco {
    param(
        [string]$PngPath,
        [string]$IcoPath
    )

    if (!(Test-Path $PngPath)) {
        return $false
    }

    try {
        Add-Type -AssemblyName System.Drawing
        $source = [System.Drawing.Image]::FromFile($PngPath)
        $bitmap = New-Object System.Drawing.Bitmap 256, 256
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.Clear([System.Drawing.Color]::Transparent)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.DrawImage($source, 0, 0, 256, 256)
        $icon = [System.Drawing.Icon]::FromHandle($bitmap.GetHicon())
        $stream = [System.IO.File]::Open($IcoPath, [System.IO.FileMode]::Create)
        $icon.Save($stream)
        $stream.Close()
        $graphics.Dispose()
        $source.Dispose()
        $bitmap.Dispose()
        $icon.Dispose()
        return $true
    }
    catch {
        Write-Host "Could not convert PNG icon to ICO: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

Write-Step "Preparing LabelMate in $ProjectRoot"

if (!(Test-Path $VenvDir)) {
    Write-Step "Creating Python virtual environment"
    $PythonCommand = Find-Python
    Invoke-Expression "$PythonCommand -m venv `"$VenvDir`""
}
else {
    Write-Step "Using existing virtual environment"
}

Write-Step "Upgrading pip"
& $PythonExe -m pip install --upgrade pip

Write-Step "Installing LabelMate dependencies"
& $PipExe install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Step "Creating launcher"
$launcher = @"
@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python main.py
if errorlevel 1 pause
"@
Set-Content -Path $LauncherPath -Value $launcher -Encoding ASCII

Write-Step "Creating desktop shortcut"
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "LabelMate.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $LauncherPath
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.Description = "Launch LabelMate"

if (!(Test-Path $IconIco)) {
    [void](Convert-PngToIco -PngPath $IconPng -IcoPath $IconIco)
}

if (Test-Path $IconIco) {
    $shortcut.IconLocation = $IconIco
}

$shortcut.Save()

Write-Step "Starting LabelMate"
Start-Process -FilePath $LauncherPath -WorkingDirectory $ProjectRoot

Write-Host ""
Write-Host "Done. You can now launch LabelMate from the desktop shortcut." -ForegroundColor Green
