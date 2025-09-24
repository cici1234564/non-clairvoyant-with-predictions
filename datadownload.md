# ATLAS/PAI Trace: Download and Verification Guide

This guide provides step-by-step instructions for downloading, verifying, and extracting the ATLAS/PAI trace dataset. The process is organized by operating system (Windows, macOS, Linux) and includes all necessary commands. Only download scripts and verification methods are provided; large files are not stored directly in Git. Please select the section for your operating system.

## Which files should I download?

Example (3 core tables):
- pai_group_tag_table.tar.gz
- pai_task_table.tar.gz
- pai_job_table.tar.gz

You can change the destination folder by replacing `C:\AlibabaTrace` (Windows) or `~/alibaba-trace` (macOS/Linux) with your preferred path.

---

## Windows (PowerShell)

Windows 10/11 includes `curl.exe` and `tar.exe` by default, so no extra installation is needed. `curl.exe -L` follows redirects, and `-O` saves with the original filename.

### 1) Create and enter the directory
```powershell
New-Item -Type Directory -Path "C:\AlibabaTrace" -Force
Set-Location "C:\AlibabaTrace"     # Check current path: Get-Location
```

### 2) Download the three archives
```powershell
$urls = @(
  "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz",
  "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz",
  "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz"
)
foreach ($u in $urls) { curl.exe -L -O $u }
```
Alternatively, using full PowerShell syntax:
```powershell
foreach ($u in $urls) {
  $fn = Split-Path $u -Leaf
  Invoke-WebRequest -Uri $u -OutFile $fn
}
```

### 3) Verify download completion
```powershell
Get-ChildItem *.tar.gz | Format-Table Name, Length, LastWriteTime
# Quick existence check:
Test-Path .\pai_group_tag_table.tar.gz
Test-Path .\pai_task_table.tar.gz
Test-Path .\pai_job_table.tar.gz
```

### 4) (Optional) Verify file hash (integrity)
```powershell
Get-FileHash .\pai_group_tag_table.tar.gz
Get-FileHash .\pai_task_table.tar.gz
Get-FileHash .\pai_job_table.tar.gz
```
Compare the SHA256 values with those on the release page.

### 5) Extract or view archive contents
```powershell
# View contents (without extracting)
tar -tzf .\pai_group_tag_table.tar.gz | Select-Object -First 10
# Extract to current directory
tar -xzf .\pai_group_tag_table.tar.gz
tar -xzf .\pai_task_table.tar.gz
tar -xzf .\pai_job_table.tar.gz
```

---

## macOS

### 1) Create and enter the directory
```bash
mkdir -p ~/alibaba-trace && cd ~/alibaba-trace
```

### 2) Download
```bash
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz"
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz"
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz"
```

### 3) Verify download completion
```bash
ls -lh *.tar.gz
```

### 4) (Optional) Verify SHA256
```bash
shasum -a 256 *.tar.gz
```

### 5) Extract or view
```bash
# View contents
tar -tzf pai_group_tag_table.tar.gz | head
# Extract
tar -xzf pai_group_tag_table.tar.gz
tar -xzf pai_task_table.tar.gz
tar -xzf pai_job_table.tar.gz
```

---

## Linux

### 1) Create and enter the directory
```bash
mkdir -p ~/alibaba-trace && cd ~/alibaba-trace
```

### 2) Download
```bash
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz"
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz"
curl -L -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz"
```

### 3) Verify download completion
```bash
ls -lh *.tar.gz
```

### 4) (Optional) Verify SHA256
```bash
sha256sum *.tar.gz
# If you have an official checksums file:
# sha256sum -c checksums.sha256
```

### 5) Extract or view
```bash
tar -tzf pai_group_tag_table.tar.gz | head
tar -xzf pai_group_tag_table.tar.gz
tar -xzf pai_task_table.tar.gz
tar -xzf pai_job_table.tar.gz
```

---

## FAQ

**Where are the downloaded files?**
They are in your current working directory. On Windows, use `Get-Location`; on macOS/Linux, use `pwd`.

**PowerShell curl errors?**
Use `curl.exe` (with `.exe`) to call the real cURL. Otherwise, older PowerShell may alias `curl` to `Invoke-WebRequest`, which uses different parameters. You can also use `Invoke-WebRequest -OutFile`.

---

## References
- [Microsoft Learn: PowerShell File Management](https://learn.microsoft.com/en-us/powershell/scripting/learn/deep-dives/file-management)
- [curl Documentation](https://curl.se/docs/manpage.html)
- [ss64.com: tar, shasum](https://ss64.com)
- [man7.org: sha256sum](https://man7.org/linux/man-pages/man1/sha256sum.1.html)
