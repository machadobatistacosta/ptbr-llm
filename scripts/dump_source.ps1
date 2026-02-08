$ArtifactPath = "c:\Users\caike\.gemini\antigravity\brain\eaadbc6d-8fc4-4fc5-9ee7-1c09c6974b2e\project_source_dump.md"
$SourceDir = "c:\Users\caike\Desktop\ptbr-llm"
$Files = Get-ChildItem -Path $SourceDir -Recurse -Include *.rs,*.toml,*.json | Where-Object { $_.FullName -notmatch "target" -and $_.FullName -notmatch "temp" -and $_.Name -ne "tokenizer.json" }

foreach ($f in $Files) {
    try {
        $RelPath = $f.FullName.Substring($SourceDir.Length + 1)
        $Ext = $f.Extension.TrimStart('.')
        
        Add-Content -Path $ArtifactPath -Value "`n## $RelPath`n```$Ext"
        Get-Content $f.FullName | Add-Content -Path $ArtifactPath
        Add-Content -Path $ArtifactPath -Value "```"
    } catch {
        Write-Warning "Failed to process $($f.FullName): $_"
    }
}
Write-Host "Done processing $($Files.Count) files."
