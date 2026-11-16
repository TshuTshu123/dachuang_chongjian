#requires -Version 5.1
<#
.SYNOPSIS
Export one MVX frame as paired RAW camera assets and a merged, colored PLY.
.EXAMPLE
.\export_capture.ps1 -MvxPath C:\storage\take\take_joined.mvx -Frame 0 -Name export_00006
.EXAMPLE
.\export_capture.ps1 -MvxPath D:\take\take_joined.mvx -RingRoot C:\ring_it -OutputRoot D:\exports
#>
[CmdletBinding()]
param(
    [string]$MvxPath,
    [ValidateRange(0, 2147483646)][int]$Frame = 0,
    [string]$Name,
    [string]$RingRoot = 'C:\ring_it',
    [string]$BuildPath,
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'exports'),
    [ValidateRange(0, 10000)][int]$ExpectedStreams = 0,
    [switch]$NoZip,
    [switch]$KeepIntermediate,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Json($Value, [string]$Path) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-Exporter([string]$Exe, [string[]]$Arguments, [string]$LogPath) {
    Write-Host "Running $([IO.Path]::GetFileName($Exe)); log: $LogPath"
    # The call operator passes each argument separately, including paths with spaces.
    # Run in our private work directory: vendor logs/debug files cannot alter the installation.
    Push-Location $workDir
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes native stderr (including harmless DLL probes)
        # to NativeCommandError. The process exit code determines success here.
        $ErrorActionPreference = 'Continue'
        & $Exe @Arguments *> $LogPath
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }
    if ($code -ne 0) { throw "Exporter exited with code $code. See $LogPath" }
}

function Get-PlyVertexCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing PLY: $Path" }
    $reader = [IO.StreamReader]::new($Path, [Text.Encoding]::ASCII)
    try {
        if ($reader.ReadLine() -ne 'ply') { throw "Invalid PLY header: $Path" }
        $count = 0L
        for ($i = 0; $i -lt 200; $i++) {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            if ($line -match '^element vertex (\d+)\s*$') { $count = [long]$Matches[1] }
            if ($line -eq 'end_header') {
                if ($count -le 0) { throw "PLY has no vertices: $Path" }
                return $count
            }
        }
        throw "Missing/oversized PLY header: $Path"
    } finally { $reader.Dispose() }
}

if (-not $MvxPath) { $MvxPath = (Read-Host 'MVX path (use the multi-camera *_joined.mvx)').Trim('"') }
$source = Get-Item -LiteralPath $MvxPath
if ($source.PSIsContainer -or $source.Extension -ine '.mvx') { throw 'Input must be an existing .mvx file.' }
if (-not $BuildPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $RingRoot 'builds'))) {
        $nested = Join-Path $RingRoot 'ring_it'
        if (Test-Path -LiteralPath (Join-Path $nested 'builds')) { $RingRoot = $nested }
    }
    $builds = @(Get-ChildItem -LiteralPath (Join-Path $RingRoot 'builds') -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'XmlGraphRunner.exe') })
    if ($builds.Count -ne 1) { throw 'Specify -BuildPath: expected exactly one build containing XmlGraphRunner.exe.' }
    $BuildPath = $builds[0].FullName
}
$BuildPath = (Get-Item -LiteralPath $BuildPath).FullName
$runner = Join-Path $BuildPath 'XmlGraphRunner.exe'
$cropper = Join-Path $BuildPath 'Mvx2FileCropper.exe'
foreach ($p in @($runner, $cropper, (Join-Path $PSScriptRoot 'export_graph.xml'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing dependency: $p" }
}
if (-not $Name) { $Name = '{0}_f{1:D5}_{2}' -f $source.BaseName, $Frame, (Get-Date -Format 'yyyyMMdd_HHmmss_fff') }
if ($Name -in @('.', '..') -or $Name -match '[<>:"/\\|?*\x00-\x1f]' -or $Name -match '[. ]$') {
    throw 'Name must be a single valid folder/file name.'
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$bundle = Join-Path $OutputRoot $Name
$zipPath = "$bundle.zip"
foreach ($vendorPath in @($source.FullName, $BuildPath, $bundle)) {
    if ($vendorPath -match '[^\x00-\x7f]') {
        throw 'This RING version mixes ANSI command-line paths and UTF-8 XML paths. Use English/ASCII input, build and output paths (e.g. -OutputRoot D:\exports). Spaces are supported.'
    }
}
if ((Test-Path -LiteralPath $bundle) -or (Test-Path -LiteralPath $zipPath)) {
    throw "Output exists; choose another -Name: $bundle"
}
$workDir = Join-Path $bundle '_work'
$logsDir = Join-Path $bundle 'logs'
$rawRoot = Join-Path $bundle "data_raw\$Name"
$dataRoot = Join-Path $bundle 'data'
$cropped = Join-Path $workDir 'selected.mvx'
$plyPath = Join-Path $dataRoot "$Name.ply"
$plan = [ordered]@{
    source_mvx = $source.FullName; source_frame_index = $Frame
    crop_end_exclusive = $Frame + 1; build = $BuildPath
    output = $bundle; merged_ply = $plyPath; raw = $rawRoot
    archive = $(if ($NoZip) { $null } else { $zipPath })
}
if ($DryRun) { $plan | ConvertTo-Json; return }
foreach ($directory in @($workDir, $logsDir, $rawRoot, $dataRoot)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$manifest = [ordered]@{ schema_version = 1; status = 'running'; created_utc = [DateTime]::UtcNow.ToString('o'); plan = $plan }
$manifestPath = Join-Path $bundle 'manifest.json'
Write-Json $manifest $manifestPath
try {
    # Cropper uses an exclusive end: --start N --end N+1 is exactly one frame.
    Invoke-Exporter $cropper @('--in', $source.FullName, '--start', "$Frame", '--end', "$($Frame + 1)", '--out', $cropped) (Join-Path $logsDir 'crop.log')
    if (-not (Test-Path -LiteralPath $cropped)) { throw 'Cropper produced no MVX; frame may be out of range.' }
    $cropXmlPath = [IO.Path]::ChangeExtension($cropped, '.xml')
    [xml]$cropXml = Get-Content -LiteralPath $cropXmlPath -Raw
    $frames = @($cropXml.SelectNodes('//ATOM[@TYPE="Frame"]'))
    if ($frames.Count -eq 0) { throw 'Selected MVX contains no frames.' }
    $streams = @($frames | Group-Object STREAMID)
    if (@($streams | Where-Object Count -ne 1).Count) { throw 'Crop did not contain exactly one frame per stream.' }
    if ($ExpectedStreams -gt 0 -and $streams.Count -ne $ExpectedStreams) {
        throw "Expected $ExpectedStreams streams, selected MVX contains $($streams.Count)."
    }
    [xml]$graph = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'export_graph.xml') -Raw
    $graph.SelectSingleNode('//parameter[@name="MVX File Path"]').SetAttribute('value', $cropped)
    $graph.SelectSingleNode('//parameter[@name="Ouptut directory"]').SetAttribute('value', $rawRoot)
    $graph.SelectSingleNode('//parameter[@name="File name"]').SetAttribute('value', $plyPath)
    $graphPath = Join-Path $workDir 'export.xml'
    $graph.Save($graphPath)
    Invoke-Exporter $runner @($graphPath, '-d') (Join-Path $logsDir 'export.log')
    $mergedCount = Get-PlyVertexCount $plyPath
    $rawCount = 0L
    $inventory = @()
    foreach ($stream in $streams) {
        $id = [string][int]$stream.Name
        $directory = Join-Path $rawRoot $id
        $cameraPlys = @(Get-ChildItem -LiteralPath $directory -Filter '*.ply' -File |
            Where-Object BaseName -Match '^\d+$')
        if ($cameraPlys.Count -ne 1) { throw "Stream $id must contain exactly one PLY." }
        $frameName = $cameraPlys[0].BaseName
        # Vendor emits calibration without a frame prefix. Normalize for run.py/raw_capture.py.
        foreach ($metadata in @('CameraParams_Primary.json', 'CameraParams_Secondary.json', 'transform.txt', 'PhotogrammetryInfo.txt')) {
            $old = Join-Path $directory $metadata
            if (Test-Path -LiteralPath $old) { Move-Item -LiteralPath $old -Destination (Join-Path $directory "${frameName}_$metadata") }
        }
        foreach ($required in @("$frameName.ply", "${frameName}_RGB_01.png", "${frameName}_CameraParams_Primary.json", "${frameName}_transform.txt", 'CylinderCropParams.cfg')) {
            $asset = Join-Path $directory $required
            if (-not (Test-Path -LiteralPath $asset -PathType Leaf) -or (Get-Item -LiteralPath $asset).Length -eq 0) {
                throw "Missing/empty RAW asset: $asset. Use a multi-camera MVX retaining photos and calibration."
            }
        }
        $count = Get-PlyVertexCount $cameraPlys[0].FullName
        $rawCount += $count
        $inventory += [ordered]@{ stream = $id; exported_frame = $frameName; cropped_frame = [string]$stream.Group[0].FRAME; timestamp_ns = [string]$stream.Group[0].TS; vertices = $count }
    }
    if ($rawCount -ne $mergedCount) { throw "RAW total $rawCount differs from merged PLY $mergedCount." }
    $manifest.status = 'complete'
    $manifest['completed_utc'] = [DateTime]::UtcNow.ToString('o')
    $manifest['streams'] = $inventory
    $manifest['stream_count'] = $streams.Count
    $manifest['merged_vertices'] = $mergedCount
    $manifest['unit'] = 'mm (RING export convention; no rescaling applied)'
    $manifest['merged_ply_sha256'] = (Get-FileHash -LiteralPath $plyPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifest['source_size_bytes'] = $source.Length
    $manifest['source_last_write_utc'] = $source.LastWriteTimeUtc.ToString('o')
    $manifest['source_hash_note'] = 'Full source MVX is not hashed; cropped-frame SHA256 identifies the exported input.'
    $manifest['cropped_mvx_sha256'] = (Get-FileHash -LiteralPath $cropped -Algorithm SHA256).Hash.ToLowerInvariant()
    Copy-Item -LiteralPath $cropXmlPath -Destination (Join-Path $logsDir 'selected_frame.xml')
    Copy-Item -LiteralPath $graphPath -Destination (Join-Path $logsDir 'export_graph.xml')
    Write-Json $manifest (Join-Path $rawRoot 'capture_info.json')
    Write-Json $manifest $manifestPath
    if (-not $KeepIntermediate) {
        # Only this invocation's private directory can be removed.
        $resolvedWork = [IO.Path]::GetFullPath($workDir)
        $expectedWork = [IO.Path]::GetFullPath((Join-Path $bundle '_work'))
        if ($resolvedWork -ne $expectedWork -or -not $resolvedWork.StartsWith("$bundle\", [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe intermediate path.' }
        Remove-Item -LiteralPath $resolvedWork -Recurse -Force
    }
    $hashes = @(Get-ChildItem -LiteralPath $bundle -Recurse -File | ForEach-Object {
        '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.FullName.Substring($bundle.Length + 1).Replace('\', '/')
    })
    $hashes | Set-Content -LiteralPath (Join-Path $bundle 'SHA256SUMS.txt') -Encoding UTF8
    if (-not $NoZip) {
        Write-Host 'Creating ZIP (camera PNGs may take a while)...'
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::CreateFromDirectory($bundle, $zipPath, [IO.Compression.CompressionLevel]::Fastest, $true)
        '{0}  {1}' -f (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant(), [IO.Path]::GetFileName($zipPath) |
            Set-Content -LiteralPath "$zipPath.sha256" -Encoding ASCII
        Write-Host "DOWNLOAD: $zipPath" -ForegroundColor Green
    }
    Write-Host "PLY: $plyPath"
    Write-Host "RAW: $rawRoot"
    Write-Host "OK: $($streams.Count) streams, $mergedCount points."
} catch {
    $manifest.status = 'failed'
    $manifest['error'] = $_.Exception.Message
    Write-Json $manifest $manifestPath
    # Retain failed outputs and logs for diagnosis; never label them ready to download.
    throw
}
