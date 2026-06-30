$ErrorActionPreference = "SilentlyContinue"

$stdin = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($stdin)) { exit 0 }

try {
  $inputJson = $stdin | ConvertFrom-Json
} catch {
  exit 0
}

if ($inputJson.hook_event_name -ne "SessionStart") { exit 0 }
if ($inputJson.source -ne "compact") { exit 0 }

$cwd = [string]$inputJson.cwd
$transcriptPath = [string]$inputJson.transcript_path
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$skillRelativePath = ".agents\skills\debug-receipt\SKILL.md"
$skillMarkers = @(
  "debug-receipt",
  "debug_receipt",
  ".agents\\skills\\debug-receipt\\SKILL.md",
  ".agents/skills/debug-receipt/SKILL.md",
  "/debug-receipt"
)

$inReceiptParserRepo = $false
if ($cwd) {
  try {
    $resolvedCwd = (Resolve-Path -LiteralPath $cwd).Path.TrimEnd("\", "/")
    $resolvedRepoRoot = (Resolve-Path -LiteralPath $repoRoot).Path.TrimEnd("\", "/")
    $repoPrefix = $resolvedRepoRoot + [System.IO.Path]::DirectorySeparatorChar
    if (
      $resolvedCwd.Equals($resolvedRepoRoot, [StringComparison]::OrdinalIgnoreCase) -or
      $resolvedCwd.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)
    ) {
      $inReceiptParserRepo = $true
    } else {
      $skillPath = Join-Path $resolvedCwd $skillRelativePath
      if (Test-Path -LiteralPath $skillPath -PathType Leaf) {
        $inReceiptParserRepo = $true
      }
    }
  } catch {
    $inReceiptParserRepo = $false
  }
}

if (-not $inReceiptParserRepo) { exit 0 }

function Test-ContainsSkillMarker {
  param(
    [string]$Text
  )

  if ([string]::IsNullOrEmpty($Text)) { return $false }

  foreach ($marker in $skillMarkers) {
    if ($Text.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  return $false
}

function Test-TranscriptContainsSkillMarker {
  param(
    [string]$Path
  )

  try {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
      $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8, $true, 65536)
      $buffer = New-Object char[] 65536
      $carry = ""
      $maxMarkerLength = ($skillMarkers | ForEach-Object { $_.Length } | Measure-Object -Maximum).Maximum

      while (-not $reader.EndOfStream) {
        $charsRead = $reader.Read($buffer, 0, $buffer.Length)
        if ($charsRead -le 0) { break }

        $chunk = $carry + (New-Object string -ArgumentList $buffer, 0, $charsRead)
        if (Test-ContainsSkillMarker -Text $chunk) {
          return $true
        }

        if ($chunk.Length -gt $maxMarkerLength) {
          $carry = $chunk.Substring($chunk.Length - $maxMarkerLength)
        } else {
          $carry = $chunk
        }
      }
    } finally {
      if ($reader) { $reader.Dispose() }
      $stream.Dispose()
    }
  } catch {
    return $false
  }

  return $false
}

$active = $false

if ($transcriptPath -and (Test-Path -LiteralPath $transcriptPath -PathType Leaf)) {
  $active = Test-TranscriptContainsSkillMarker -Path $transcriptPath
}

if (-not $active) { exit 0 }

$checkpoint = @"
Compaction just occurred during a debug-receipt workflow. Before editing files or running another benchmark, restate:
- target fixture(s) and current mode
- non-negotiable rules from the debug-receipt skill
- last trusted benchmark or accuracy artifact
- last explicit user decision
- current blockers or truth-file questions
- next planned action and why it is still general-purpose

If any item is unknown, inspect current repo state and artifacts first. Do not continue from memory alone.

Production parsing code must not special-case specific merchants, receipt IDs, known fixture ranges, known dates, known product lists, or known final totals. It may implement general layout/format strategies when they are triggered by structural OCR evidence and validated by arithmetic consistency.

If identical image/OCR fixtures have conflicting truth expectations, stop and ask which truth convention should win. Do not add parser logic that distinguishes identical inputs by fixture identity.

Before marking complete, verify that no new production code contains hardcoded fixture-range helpers, full hardcoded line_items answer lists, or late known-final-output overrides.
"@

$output = @{
  hookSpecificOutput = @{
    hookEventName = "SessionStart"
    additionalContext = $checkpoint
  }
}

$output | ConvertTo-Json -Depth 5 -Compress
