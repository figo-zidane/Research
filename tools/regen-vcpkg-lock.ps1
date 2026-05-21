# 現在の .vcpkg レジストリと build\vs2026\vcpkg_installed\vcpkg\status から
# vcpkg.lock.txt の候補を stdout に出力する。
# 自動上書きはしない — 手動でリダイレクトして git diff してから採用すること。
#
# 使い方:
#   .\tools\regen-vcpkg-lock.ps1 > vcpkg.lock.txt.new
#   git diff --no-index vcpkg.lock.txt vcpkg.lock.txt.new
#   Move-Item -Force vcpkg.lock.txt.new vcpkg.lock.txt

$repo  = "c:\Users\figo1\Documents\Work\Research"
$reg   = Join-Path $repo ".vcpkg"
$inst  = Join-Path $repo "build\vs2026\vcpkg_installed\vcpkg\status"

if (-not (Test-Path $inst)) {
    Write-Error "vcpkg status file not found: $inst`nRun 'cmake --preset vs2026-x64' first."
    exit 1
}

$baselineSha = (& git -C $reg rev-parse HEAD).Trim()
$exeHash     = (Get-FileHash (Join-Path $reg "vcpkg.exe") -Algorithm SHA256).Hash
$toolMeta    = (Get-Content (Join-Path $reg "scripts\vcpkg-tool-metadata.txt") -Raw).TrimEnd()

$records = (Get-Content $inst -Raw) -split "\r?\n\r?\n" | ForEach-Object {
    $kv = @{}
    foreach ($line in ($_ -split "\r?\n")) {
        if ($line -match '^([^:]+):\s*(.*)$') { $kv[$matches[1]] = $matches[2] }
    }
    if ($kv.ContainsKey("Version") -and -not $kv.ContainsKey("Feature")) {
        $pkg = $kv["Package"]
        $ver = $kv["Version"]
        if ($kv.ContainsKey("Port-Version") -and $kv["Port-Version"] -ne "0") {
            $ver = "$ver#$($kv['Port-Version'])"
        }
        $portfile = Join-Path $reg "ports\$pkg\portfile.cmake"
        $sha = if (Test-Path $portfile) {
            (Get-FileHash $portfile -Algorithm SHA512).Hash.ToLower()
        } else { "MISSING" }
        "{0,-24} {1,-14} {2}" -f $pkg, $ver, $sha
    }
} | Where-Object { $_ } | Sort-Object

@"
# vcpkg.lock.txt -- research-renderer の解決済みバージョン
# Generated $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')
# 列: <package> <version>[#<port-version>] <portfile-sha512>
#
# baseline-sha     = $baselineSha
# vcpkg.exe-sha256 = $exeHash
# --- .vcpkg/scripts/vcpkg-tool-metadata.txt verbatim ---
$toolMeta
# -------------------------------------------------------

"@
$records -join "`r`n"
