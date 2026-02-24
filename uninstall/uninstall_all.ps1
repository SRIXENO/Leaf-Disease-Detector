# uninstall_all.ps1
# Uninstalls ALL Python packages from the current environment

Write-Host "🔍 Collecting installed packages..."

# Detect Python command
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} else {
    Write-Host "❌ Neither 'python' nor 'py' is found. Install Python first."
    Pause
    exit
}

# Export installed packages
& $pythonCmd -m pip freeze > packages.txt

if (-Not (Test-Path packages.txt)) {
    Write-Host "❌ Failed to create packages.txt (pip might not be installed)."
    Pause
    exit
}

Write-Host "📦 Uninstalling packages..."
Get-Content packages.txt | ForEach-Object {
    if ($_ -ne "") {
        Write-Host "⏳ Uninstalling $_ ..."
        & $pythonCmd -m pip uninstall -y $_
    }
}

Remove-Item packages.txt
Write-Host "✅ All packages have been uninstalled successfully!"
Pause
