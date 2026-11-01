param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$taskPlan=Get-Content -LiteralPath (Join-Path $Root 'priority_plan.json') -Raw | ConvertFrom-Json
$taskParent=Get-CimInstance Win32_Process -Filter ('ProcessId='+$taskPlan.coordinator)
if(-not $taskParent -or $taskParent.CreationDate.ToUniversalTime() -ne ([datetimeoffset]$taskPlan.coordinator_created).UtcDateTime){throw 'Coordinator identity changed'}
$taskManifest=$taskPlan.manifest.Replace('/','\').ToLowerInvariant()
while($true){
    $taskNow=Get-CimInstance Win32_Process -Filter ('ProcessId='+$taskPlan.coordinator)
    if(-not $taskNow -or $taskNow.CreationDate -ne $taskParent.CreationDate){break}
    $taskWorkers=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine -match '--job\s+\d+(\s|$)' -and $_.CommandLine.Replace('/','\').ToLowerInvariant().Contains($taskManifest) })
    if($taskWorkers.Count -gt 8){throw 'Unexpected duplicate worker'}
    foreach($taskWorker in $taskWorkers){
        $taskProcess=Get-Process -Id ([int]$taskWorker.ProcessId) -ErrorAction SilentlyContinue
        if($taskProcess -and [string]$taskProcess.PriorityClass -ne 'Normal'){
            $taskProcess.PriorityClass='Normal'
            @{pid=$taskWorker.ProcessId;created=$taskWorker.CreationDate;priority='Normal';at=[datetime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress | Write-Output
        }
    }
    Start-Sleep -Seconds 15
}
@{complete=$true;coordinator_exited=$true;scientific_completion=$false;finished=[datetime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'priority_complete.json') -Encoding utf8
