$ErrorActionPreference = "Stop"
$ArtifactPath = 'c:\Users\caike\.gemini\antigravity\brain\eaadbc6d-8fc4-4fc5-9ee7-1c09c6974b2e\project_source_dump.md'
$SourceDir = 'c:\Users\caike\Desktop\ptbr-llm'

Write-Host "Scanning files in $SourceDir..."
$Files = Get-ChildItem -Path $SourceDir -Recurse -Include *.rs, *.toml, *.json | Where-Object { $_.FullName -notmatch "target" -and $_.FullName -notmatch "temp" -and $_.Name -ne "tokenizer.json" }

Write-Host "Found $($Files.Count) files."

foreach ($f in $Files) {
    try {
        $RelPath = $f.FullName.Substring($SourceDir.Length + 1)
        $Ext = $f.Extension.TrimStart('.')
        
        $Header = "`n## $RelPath`n```$Ext"
        Add-Content -Path $ArtifactPath -Value $Header
        
        $Content = Get-Content -LiteralPath $f.FullName -Raw
        Add-Content -Path $ArtifactPath -Value $Content
        
        Add-Content -Path $ArtifactPath -Value "```"
        Write-Host "Processed: $RelPath"
    } catch {
        Write-Warning "Error processing $RelPath : $_"
    }
}
Write-Host "Done."
