param([string]$DocumentName = '')
$ErrorActionPreference = 'Stop'
$taskPack = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'Submission_Pack'))
$taskWord = $null
try {
    $taskWord = New-Object -ComObject Word.Application
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskWord.AutomationSecurity = 3
    $taskFiles = Get-ChildItem -LiteralPath $taskPack -Filter '*.docx' -Recurse
    if ($DocumentName) { $taskFiles = $taskFiles | Where-Object { $_.Name -eq $DocumentName } }
    foreach ($taskFile in $taskFiles) {
        $taskDoc = $null
        try {
            $taskDoc = $taskWord.Documents.Open($taskFile.FullName, $false, $true, $false)
            $taskDoc.Repaginate()
            $taskPdf = [IO.Path]::ChangeExtension($taskFile.FullName, '.pdf')
            $taskDoc.ExportAsFixedFormat($taskPdf, 17)
            Write-Output ('Rendered ' + $taskFile.Name)
        } finally {
            if ($null -ne $taskDoc) {
                $taskDoc.Close(0)
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskDoc)
            }
        }
    }
} finally {
    if ($null -ne $taskWord) {
        $taskWord.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)
    }
}
