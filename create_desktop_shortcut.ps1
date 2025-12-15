# PowerShell script to create desktop shortcut for dev environment launcher

$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "PhysioMetrics Dev Environment.lnk"
$TargetPath = "C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\open_dev_environment.bat"
$WorkingDirectory = "C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6"

$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $WorkingDirectory
$Shortcut.Description = "Launch PhysioMetrics development environment (Terminal, Claude, VS Code, Explorer)"
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully at: $ShortcutPath" -ForegroundColor Green
