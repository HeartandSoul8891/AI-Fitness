param(
    [string]$InputFile = "tree.txt",
    [string]$OutputFile = "tree_clean.txt"
)

if (-not (Test-Path $InputFile)) {
    Write-Error "ERROR: '$InputFile' was not found."
    exit 1
}

$lines = Get-Content -LiteralPath $InputFile
$output = [System.Collections.Generic.List[string]]::new()
$output.Add('root')

$skipVenv = $false
$skipPyCache = $false

# Unicode box-drawing characters via hex code points to avoid encoding errors
$cTee = [string][char]0x251C + [string][char]0x2500 + [string][char]0x2500
$cCorner = [string][char]0x2514 + [string][char]0x2500 + [string][char]0x2500
$cVert = [string][char]0x2502

foreach ($line in $lines) {
    if ($line -match '^\s*\+---venv\s*$') { break }
    if ($line -match 'pycache') {
        $skipPyCache = $true
        continue
    }
    if ($skipPyCache) {
        if ($line -match '\.pyc$') { continue }
        if ($line -match '^\s*(\+---|\\---)') {
            $skipPyCache = $false
        } else {
            continue
        }
    }
    if ($line -match '^Folder PATH listing') { continue }
    if ($line -match '^Volume serial number') { continue }
    if ($line -match '^\s*[A-Z]:\\') { continue }
    if ($line -match '\.pyc\s*$') { continue }
    
    $line = $line -replace '\+---', $cTee
    $line = $line -replace '\\---', $cCorner
    $line = $line -replace '\|', $cVert
    $output.Add($line)
}

[System.IO.File]::WriteAllLines($OutputFile, $output, [System.Text.UTF8Encoding]::new($false))

Write-Host "`nDone!`nInput : $InputFile`nOutput: $OutputFile`n"