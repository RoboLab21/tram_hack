param(
    [string]$Image = "check-code:humble"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

docker build --tag $Image $ProjectDir
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось собрать Docker-образ '$Image'."
}
