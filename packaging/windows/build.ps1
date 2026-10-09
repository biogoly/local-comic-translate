param(
    [string]$Python = '3.12',
    [string]$Environment = '',
    [string]$OutputDirectory = '',
    [switch]$SkipTests,
    [switch]$RequireCuda
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $Environment) { $Environment = Join-Path $repoRoot '.venv-win-build' }
$Environment = [IO.Path]::GetFullPath($Environment)
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repoRoot 'release-artifacts' }
$buildPython = Join-Path $Environment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $buildPython)) {
    & uv venv --python $Python $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated build environment.' }
}
& uv pip sync (Join-Path $PSScriptRoot 'requirements.lock') --require-hashes --python $buildPython
if ($LASTEXITCODE -ne 0) { throw 'Could not install locked build dependencies.' }
$buildArguments = @((Join-Path $PSScriptRoot 'build.py'), '--output-dir', $OutputDirectory)
if ($SkipTests) { $buildArguments += '--skip-tests' }
if ($RequireCuda) { $buildArguments += '--require-cuda' }
& $buildPython @buildArguments
if ($LASTEXITCODE -ne 0) { throw 'Windows package build or verification failed.' }
