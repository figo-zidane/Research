# Python ビルド依存（nanobind）をハッシュ検証付きでプロジェクト内に隔離インストールする。
#
# vcpkg のローカルレジストリ固定と同じ思想:
#   - バージョンと artifact SHA256 を tools/python-requirements.txt で固定
#   - --require-hashes で改ざん・差し替えを拒否（ハッシュ不一致なら pip が停止）
#   - --only-binary=:all: でソースからのビルド（任意コード実行）を禁止
#   - --no-deps で暗黙のトランジティブ依存を遮断（nanobind はランタイム依存なし）
#   - グローバル site-packages を汚さず .python-deps/ に隔離（再現性）
#
# 使い方:
#   .\tools\install-python-deps.ps1
#
# 更新手順は documents/vcpkg_supply_chain_hardening.md の
# 「Python ビルド依存（pip）」節を参照。

$ErrorActionPreference = "Stop"

$repo   = Split-Path -Parent $PSScriptRoot
$req    = Join-Path $repo "tools\python-requirements.txt"
$target = Join-Path $repo ".python-deps"

if (-not (Test-Path $req)) {
    Write-Error "requirements file not found: $req"
    exit 1
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Error "python が PATH にありません。Python 3.10+ をインストールしてください。"
    exit 1
}
Write-Host "Using python: $python"
& $python --version

# クリーンインストール（前回分が残っていると --require-hashes の検証が曖昧になるため）
if (Test-Path $target) {
    Remove-Item -Recurse -Force $target
}

& $python -m pip install `
    --require-hashes `
    --only-binary=:all: `
    --no-deps `
    --target $target `
    -r $req

if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed (ハッシュ不一致または取得失敗の可能性)。"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Installed into: $target"
Write-Host "nanobind cmake_dir:"
$env:PYTHONPATH = $target
& $python -m nanobind --cmake_dir
