param(
    [string]$Container = "check-code-humble"
)

$ErrorActionPreference = "Stop"

docker exec -it $Container bash -lc 'source /opt/ros/humble/setup.bash; if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi; exec bash'
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось войти в контейнер '$Container'. Сначала запустите scripts/run.ps1."
}
