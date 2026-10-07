[CmdletBinding()]
param(
    [ValidateSet('Debug', 'RelWithDebInfo')]
    [string]$Configuration = 'RelWithDebInfo',
    [ValidateRange(1, 1024)]
    [int]$Jobs = 8,
    [switch]$ShadersOnly
)

$ErrorActionPreference = 'Stop'
$project_root = Split-Path -Parent $PSScriptRoot
$build_directory = Join-Path $project_root 'build'
$cache_file = Join-Path $build_directory 'CMakeCache.txt'
if ((Test-Path -LiteralPath $cache_file) -eq $false) {
    throw 'Configure the project in build/ with CMake before running the development build.'
}

$cache_values = Get-Content -LiteralPath $cache_file
$multi_configuration = @($cache_values -match '^CMAKE_CONFIGURATION_TYPES:[^=]+=.+$').Count -gt 0
$configuration_matches = @($cache_values -match "^CMAKE_BUILD_TYPE:[^=]+=$Configuration$").Count -gt 0
if (($ShadersOnly -eq $false) -and ($multi_configuration -eq $false) -and ($configuration_matches -eq $false)) {
    & cmake -S $project_root -B $build_directory "-DCMAKE_BUILD_TYPE=$Configuration"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$target = if ($ShadersOnly) { 'raytracer_runtime_shaders' } else { 'raytracer' }
& cmake --build $build_directory --config $Configuration --target $target --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($ShadersOnly) {
    Write-Host 'Shader sources updated. Use Reload Shaders in the running development application.'
} else {
    $executable = if ($Configuration -eq 'Debug') { 'raytracer_debug' } else { 'raytracer_dev' }
    if ($IsWindows -or ($env:OS -eq 'Windows_NT')) {
        $executable += '.exe'
    }
    Write-Host "Development executable: $(Join-Path $project_root "bin/$executable")"
}
