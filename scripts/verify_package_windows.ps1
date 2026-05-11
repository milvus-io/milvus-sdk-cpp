param(
    [string]$PackageDir = 'build/Pack'
)

$ErrorActionPreference = 'Stop'

$package = Get-ChildItem -Path $PackageDir -File -Filter 'libmilvus-*' | Select-Object -First 1
if (-not $package) {
    Write-Error "No package found under $PackageDir"
}

$smokeDir = Join-Path $env:RUNNER_TEMP 'milvus-sdk-package-smoke'
Remove-Item -Recurse -Force $smokeDir -ErrorAction SilentlyContinue
$extractedPackageDir = Join-Path $smokeDir 'package'
New-Item -ItemType Directory -Force -Path $extractedPackageDir | Out-Null

if ($package.Name.EndsWith('.zip')) {
    Expand-Archive -Path $package.FullName -DestinationPath $extractedPackageDir
} elseif ($package.Name.EndsWith('.tar.gz')) {
    tar -xzf $package.FullName -C $extractedPackageDir
} else {
    Write-Error "Unsupported package format: $($package.FullName)"
}

$includeMilvus = Get-ChildItem -Path $extractedPackageDir -Recurse -Directory -Filter milvus | Where-Object { $_.FullName -match '[\\/]include[\\/]milvus$' } | Select-Object -First 1
$libFile = Get-ChildItem -Path $extractedPackageDir -Recurse -File -Include milvus_sdk.lib,libmilvus_sdk.lib | Select-Object -First 1
$binFile = Get-ChildItem -Path $extractedPackageDir -Recurse -File -Include milvus_sdk.dll,libmilvus_sdk.dll | Select-Object -First 1
if (-not $includeMilvus -or -not $libFile -or -not $binFile) {
    Get-ChildItem -Path $extractedPackageDir -Recurse -File | Select-Object -First 200 | ForEach-Object { $_.FullName }
    Write-Error 'Package does not contain expected include/lib layout'
}

$includeDir = Split-Path -Parent $includeMilvus.FullName
$smokeCpp = Join-Path $smokeDir 'smoke.cpp'
$smokeObj = Join-Path $smokeDir 'smoke.obj'
$exportsFile = Join-Path $smokeDir 'milvus_sdk.exports.txt'
@'
#include <string>

#include "milvus/MilvusClientV2.h"

int
main() {
    if (milvus::INDEX_TYPE == nullptr || std::string(milvus::INDEX_TYPE) != "index_type") {
        return 1;
    }
    if (milvus::METRIC_TYPE == nullptr || std::string(milvus::METRIC_TYPE) != "metric_type") {
        return 1;
    }

    auto client = milvus::MilvusClientV2::Create();
    return client == nullptr ? 1 : 0;
}
'@ | Set-Content -Path $smokeCpp

$buildCmd = Join-Path $smokeDir 'build-smoke.cmd'
# Compile headers and inspect DLL exports instead of linking the oversized Windows import library.
@"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cl /std:c++14 /EHsc /DMILVUS_SDK_SHARED /I"$includeDir" /c /Fo"$smokeObj" "$smokeCpp"
if errorlevel 1 exit /b %errorlevel%
dumpbin /exports "$($binFile.FullName)" > "$exportsFile"
if errorlevel 1 exit /b %errorlevel%
"@ | Set-Content -Path $buildCmd
cmd /c $buildCmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$exports = Get-Content -Raw -Path $exportsFile
foreach ($symbol in @('INDEX_TYPE', 'METRIC_TYPE')) {
    if ($exports -notmatch $symbol) {
        Write-Error "Package DLL does not export $symbol"
    }
}

$env:PATH = "$(Split-Path -Parent $binFile.FullName);$env:PATH"
Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError=true, CharSet=System.Runtime.InteropServices.CharSet.Unicode)]
public static extern System.IntPtr LoadLibrary(string lpFileName);

[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError=true)]
public static extern bool FreeLibrary(System.IntPtr hModule);
'@
$handle = [Win32.NativeMethods]::LoadLibrary($binFile.FullName)
if ($handle -eq [IntPtr]::Zero) {
    $errorCode = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Error "Failed to load $($binFile.FullName), GetLastError=$errorCode"
}
[Win32.NativeMethods]::FreeLibrary($handle) | Out-Null
