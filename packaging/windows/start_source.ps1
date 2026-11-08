[CmdletBinding()]
param(
    [Parameter(Position=0)][string]$Project,
    [switch]$SetupOnly,
    [switch]$Repair,
    [switch]$UsePip,
    [string]$RuntimeRoot,
    [string]$SmokeReport,
    [switch]$RequireCuda
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($Project) { $Project = (Resolve-Path -LiteralPath $Project).Path }
if ($SmokeReport) { $SmokeReport = [IO.Path]::GetFullPath($SmokeReport) }
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $env:LOCALAPPDATA 'LocalComicTranslate\SourceRuntime' }
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
[IO.Directory]::CreateDirectory($RuntimeRoot) | Out-Null

# All setup changes are local to this process or the application's runtime.
Remove-Item Env:PYTHONHOME,Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE='1'
$env:UV_NO_CONFIG='1'
$env:UV_PYTHON_DOWNLOADS='never'
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR=Join-Path $RuntimeRoot 'uv-cache' }
$env:PIP_CACHE_DIR=Join-Path $RuntimeRoot 'pip-cache'
$env:PIP_CONFIG_FILE='NUL'
foreach ($setting in 'UV_DEFAULT_INDEX','UV_INDEX','UV_INDEX_URL','UV_EXTRA_INDEX_URL','UV_FIND_LINKS',
                       'PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_FIND_LINKS') {
    Remove-Item -LiteralPath "Env:$setting" -ErrorAction SilentlyContinue
}
$uv = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue
$finder = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
$direct = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
$candidates = @()
try {
    # A failed discovery probe is expected on machines with only one interpreter manager.
    $ErrorActionPreference = 'Continue'
    if ($uv) {
        $found = & $uv.Source python find --system --no-project --no-python-downloads cpython-3.12-windows-x86_64-none 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $found }
    }
    if (-not $candidates -and $finder) {
        $found = & $finder.Source -3.12-64 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $found }
    }
} finally {
    $ErrorActionPreference = 'Stop'
}
if ($direct -and $direct.Source -notlike '*\Microsoft\WindowsApps\python.exe') { $candidates += $direct.Source }
$basePython = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $identity = & $candidate -c 'import platform,sys; print(sys.version_info.major,sys.version_info.minor,platform.machine().lower(),sys.maxsize>2**32)'
        if ($LASTEXITCODE -eq 0 -and $identity -match '^3 12 (amd64|x86_64) True$') {
            $basePython = $candidate
            break
        }
    }
}
if (-not $basePython) {
    throw 'Install Python 3.12 for Windows x64 from python.org (current-user install with the Python Launcher), or use uv python install 3.12, then rerun run.bat.'
}

function Get-RuntimeHash([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        $digest = $algorithm.ComputeHash($stream)
        return [BitConverter]::ToString($digest).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

$lock = Join-Path $PSScriptRoot 'runtime.lock'
if (-not (Test-Path -LiteralPath $lock -PathType Leaf)) {
    # A source checkout keeps one authoritative lock, including its build tools.
    $sourceLock=Join-Path $PSScriptRoot 'requirements.lock'
    $lock=Join-Path $RuntimeRoot ((Get-RuntimeHash $sourceLock)+'.lock')
    & $basePython (Join-Path $PSScriptRoot 'build_source_asset.py') runtime-lock $sourceLock $lock
    if ($LASTEXITCODE -ne 0) { throw 'Could not derive the runtime dependencies from the Windows lock.' }
}
$hash=Get-RuntimeHash $lock
$slot=Join-Path $RuntimeRoot ('py312-'+$hash.Substring(0,16))
$owner=Join-Path $slot 'source-runtime.id'
try {
    $setupGuard = [IO.File]::Open("$slot.setup.lock", 'OpenOrCreate', 'ReadWrite', 'None')
} catch [IO.IOException] {
    throw 'Another setup is preparing this runtime. Wait for it to finish and rerun run.bat.'
}
try {
    if (Test-Path -LiteralPath $slot) {
        if (-not (Test-Path -LiteralPath $owner -PathType Leaf) -or [IO.File]::ReadAllText($owner) -ne $hash) {
            throw "Runtime directory belongs to another installation: $slot"
        }
    } else {
        [IO.Directory]::CreateDirectory($slot) | Out-Null
        [IO.File]::WriteAllText($owner,$hash)
    }
    $venv=Join-Path $slot 'venv'
    $python=Join-Path $venv 'Scripts\python.exe'
    $ready=Join-Path $slot 'dependencies.ready'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        Write-Host 'Creating the isolated Python runtime...'
        if ($uv -and -not $UsePip) {
            & $uv.Source venv --allow-existing --no-project --no-python-downloads --python $basePython $venv
        } else {
            & $basePython -m venv $venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python runtime.' }
        $Repair=$true
    }
    if ($Repair -or -not (Test-Path -LiteralPath $ready -PathType Leaf)) {
        if (Test-Path -LiteralPath $ready) { Remove-Item -LiteralPath $ready }
        Write-Host 'Installing the pinned editor dependencies from PyPI. The GPU runtime can take several GB.'
        if ($uv -and -not $UsePip) {
            & $uv.Source pip sync --python $python --default-index https://pypi.org/simple --require-hashes --strict $lock
            if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the network and disk space, then rerun run.bat.' }
            & $uv.Source pip check --python $python
        } else {
            & $python -m ensurepip --default-pip
            if ($LASTEXITCODE -ne 0) { throw 'The installed Python must include venv and ensurepip.' }
            & $python -m pip --isolated --disable-pip-version-check install --index-url https://pypi.org/simple --require-hashes --no-deps -r $lock
            if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the network and disk space, then rerun run.bat.' }
            & $python -m pip --isolated --disable-pip-version-check check
        }
        if ($LASTEXITCODE -ne 0) { throw 'The isolated runtime has incompatible or missing dependencies.' }
        [IO.File]::WriteAllText($ready,$hash)
    }
} finally {
    $setupGuard.Dispose()
}
Write-Host "Editor runtime: $python"
if ($SetupOnly) { exit 0 }
Push-Location -LiteralPath $repo
try {
    if ($SmokeReport) {
        $env:QT_QPA_PLATFORM='offscreen'
        & $python -c 'import sys; from app.package_smoke import run_smoke_test; sys.exit(run_smoke_test(sys.argv[1],require_cuda=bool(int(sys.argv[2]))))' $SmokeReport ([int]$RequireCuda.IsPresent)
    } elseif ($Project) { & $python (Join-Path $repo 'comic.py') $Project }
    else { & $python (Join-Path $repo 'comic.py') }
    $exitCode=$LASTEXITCODE
} finally { Pop-Location }
exit $exitCode
