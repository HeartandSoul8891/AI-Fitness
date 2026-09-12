function Get-Tree($Path = ".", $Exclude = @("venv", "__pycache__", ".git"), $Indent = "") {
    Get-ChildItem $Path | Where-Object { $Exclude -notcontains $_.Name } | ForEach-Object {
        Write-Host "$Indent├── $($_.Name)"
        if ($_.PSIsContainer) {
            Get-Tree -Path $_.FullName -Exclude $Exclude -Indent "$Indent│   "
        }
    }
}

Get-Tree