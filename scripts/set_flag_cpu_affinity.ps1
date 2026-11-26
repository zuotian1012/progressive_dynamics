# This host's topology was read with GetLogicalProcessorInformationEx:
# P cores: logical 0..11 (six SMT pairs); E cores: logical 12..19.
# Change only this suite's live Python workers, preserving Normal priority.
$ErrorActionPreference = 'Stop'
$flagCpu = Get-CimInstance Win32_Processor
if ($flagCpu.Name -notmatch '13600KF' -or $flagCpu.NumberOfLogicalProcessors -ne 20) {
    throw 'Affinity layout was verified for this 13600KF host only.'
}
$flagWorkspace = Split-Path -Parent $PSScriptRoot
$flagSuite = Join-Path $flagWorkspace 'output/flag_to_L9_20261122'
$flagMasks = @{ semi=15; velpro=240; direct=3840 }
foreach ($flagRunFile in Get-ChildItem "$flagSuite/L*/L*/run.json") {
    $flagMeta = Get-Content -LiteralPath $flagRunFile.FullName -Raw | ConvertFrom-Json
    if ($flagMeta.status -eq 'completed' -or $flagMeta.status -eq 'failed' -or -not $flagMeta.process_id) { continue }
    $flagInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($flagMeta.process_id)"
    if (-not $flagInfo -or $flagInfo.CommandLine -notmatch 'run_flag_parallel_wind.py' -or $flagInfo.CommandLine -notmatch 'flag_to_L9_20261122') { continue }
    $flagProcess = Get-Process -Id $flagMeta.process_id
    $flagBefore = $flagProcess.ProcessorAffinity.ToInt64()
    $flagMask = $flagMasks[$flagMeta.method]
    if ($flagBefore -eq $flagMask) { continue }
    $flagProcess.ProcessorAffinity = [IntPtr]$flagMask
    $flagAfter = (Get-Process -Id $flagMeta.process_id).ProcessorAffinity.ToInt64()
    if ($flagAfter -ne $flagMask) { throw 'Affinity readback did not match.' }
    $flagRecord = @{ run=$flagRunFile.Directory.Name; pid=$flagMeta.process_id; previous_mask=$flagBefore; mask=$flagAfter; utc=(Get-Date).ToUniversalTime().ToString('o') }
    $flagLine = ConvertTo-Json -InputObject $flagRecord -Compress
    Add-Content -LiteralPath (Join-Path $flagWorkspace 'output/flag_cpu_affinity.jsonl') -Value $flagLine
    Write-Output $flagLine
}
