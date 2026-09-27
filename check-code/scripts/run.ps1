param(
    [string]$Image = "check-code:humble",
    [string]$Container = "check-code-humble"
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$ExistingId = docker container ls --all --quiet --filter "name=^/$Container$"

if ($LASTEXITCODE -ne 0) {
    throw "Не удалось проверить состояние Docker-контейнеров."
}

if ($ExistingId) {
    $RunningId = docker container ls --quiet --filter "name=^/$Container$"
    if ($RunningId) {
        Write-Host "Контейнер '$Container' уже запущен."
        exit 0
    }

    docker start $Container
} else {
    docker run `
        --detach `
        --name $Container `
        --mount "type=bind,source=$ProjectDir,target=/workspace" `
        $Image `
        sleep infinity
}

if ($LASTEXITCODE -ne 0) {
    throw "Не удалось запустить контейнер '$Container'."
}
