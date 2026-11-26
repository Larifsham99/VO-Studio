# Latif Voice - pemasang automatik untuk Windows (WSL2 + GPU NVIDIA).
# Cara guna: buka PowerShell, tampal satu baris ini, tekan Enter:
#   irm https://raw.githubusercontent.com/Larifsham99/VO-Studio/ccr-e9aac4d2-pw4crp/studio/install_windows.ps1 | iex
# Selamat dijalankan berulang kali: langkah yang sudah siap akan dilangkau.

$ErrorActionPreference = 'Stop'
$Repo   = 'https://github.com/Larifsham99/VO-Studio.git'
$Branch = 'ccr-e9aac4d2-pw4crp'
$Model  = if ($env:LATIF_MODEL) { $env:LATIF_MODEL } else { 'chatterbox_ml' }

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Note($m) { Write-Host $m -ForegroundColor Yellow }
function Fail($m) { Write-Host "`nRALAT: $m" -ForegroundColor Red; Read-Host 'Tekan Enter untuk keluar'; throw $m }
function Wsl([string]$distro, [string]$cmd, [switch]$Root) {
    # Shows the command's output on screen and returns ONLY its exit code.
    $a = @('-d', $distro)
    if ($Root) { $a += @('-u', 'root') }
    $a += @('--', 'bash', '-lc', $cmd)
    & wsl.exe @a | Out-Host
    return $LASTEXITCODE
}

# 1. GPU ----------------------------------------------------------------------
Say 'Semak GPU NVIDIA'
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail 'Pemacu NVIDIA tidak dijumpai. Pasang "NVIDIA App" dari nvidia.com, kemas kini pemacu, restart, kemudian cuba lagi.'
}
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. WSL + Ubuntu --------------------------------------------------------------
Say 'Semak WSL dan Ubuntu'
$distro = $null
try {
    $list = (& wsl.exe -l -q 2>$null) -replace "`0", '' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $distro = $list | Where-Object { $_ -like 'Ubuntu*' } | Select-Object -First 1
} catch { }
if (-not $distro) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        Fail 'Ubuntu belum dipasang. Tutup tetingkap ini, klik kanan PowerShell > "Run as administrator", dan tampal arahan pemasang sekali lagi.'
    }
    Say 'Memasang WSL + Ubuntu (beberapa minit)'
    & wsl.exe --install -d Ubuntu
    Note "`nLANGKAH SETERUSNYA:"
    Note ' 1. RESTART laptop.'
    Note ' 2. Buka "Ubuntu" dari menu Start. Cipta nama pengguna & kata laluan (huruf kecil, tanpa ruang).'
    Note ' 3. Buka PowerShell biasa dan tampal arahan pemasang ini sekali lagi.'
    Read-Host 'Tekan Enter untuk keluar'
    return
}
Write-Host "Dijumpai: $distro"
$user = (& wsl.exe -d $distro -- whoami).Trim()
if ($user -eq 'root') {
    Fail "Buka '$distro' dari menu Start sekali dan cipta nama pengguna dahulu, kemudian jalankan pemasang ini semula."
}

# 3. GPU visible inside WSL ----------------------------------------------------
Say 'Semak GPU dalam Ubuntu'
if ((Wsl $distro 'nvidia-smi -L') -ne 0) {
    Fail 'GPU tidak kelihatan dalam Ubuntu. Kemas kini pemacu NVIDIA di Windows dan jalankan "wsl --update" dalam PowerShell.'
}

# 4. System packages -----------------------------------------------------------
Say 'Pasang pakej asas dalam Ubuntu (ffmpeg, python, git)'
$apt = 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq ffmpeg python3-venv python3-dev git curl build-essential >/dev/null'
if ((Wsl $distro $apt -Root) -ne 0) { Fail 'Pemasangan pakej Ubuntu gagal. Semak sambungan internet.' }

# 5. Code ----------------------------------------------------------------------
Say 'Muat turun / kemas kini kod Latif Voice'
$git = "set -e; if [ -d ~/vo-studio/.git ]; then cd ~/vo-studio && git fetch -q origin $Branch && git checkout -q $Branch && git pull -q --ff-only origin $Branch; else git clone -q -b $Branch $Repo ~/vo-studio; fi"
if ((Wsl $distro $git) -ne 0) { Fail 'Gagal memuat turun kod dari GitHub.' }

# 6. Python environment + model weights ---------------------------------------
Say "Pasang model suara ($Model). Ini mengambil masa 10-30 minit dan memuat turun ~6 GB."
if ((Wsl $distro "cd ~/vo-studio && bash studio/setup_server.sh $Model") -ne 0) {
    Fail 'Pemasangan model gagal. Ambil gambar skrin mesej di atas dan hantar kepada Claude.'
}

# 7. Tailscale inside Ubuntu (phone -> laptop) --------------------------------
Say 'Pasang Tailscale (supaya fon boleh sambung ke laptop)'
# Tailscale's background service needs systemd inside WSL.
if ((Wsl $distro '[ -d /run/systemd/system ]') -ne 0) {
    $null = Wsl $distro 'printf ''%s\n'' ''[boot]'' systemd=true >> /etc/wsl.conf' -Root
    & wsl.exe --shutdown
    Note 'Ubuntu telah dikonfigurasi semula. Tampal arahan pemasang sekali lagi untuk menyambung (langkah yang siap akan dilangkau).'
    Read-Host 'Tekan Enter untuk keluar'
    return
}
if ((Wsl $distro 'command -v tailscale >/dev/null') -ne 0) {
    if ((Wsl $distro 'curl -fsSL https://tailscale.com/install.sh | sh' -Root) -ne 0) {
        Note 'Tailscale gagal dipasang. Anda boleh cuba semula kemudian.'
    }
}
if ((Wsl $distro 'tailscale ip -4 >/dev/null 2>&1') -ne 0) {
    Note "`nSatu pautan log masuk Tailscale akan dipaparkan. Buka pautan itu dalam pelayar dan log masuk"
    Note '(guna akaun Google yang sama yang anda guna dalam aplikasi Tailscale di fon).'
    $null = Wsl $distro 'tailscale up' -Root
}

# 8. Desktop shortcut ----------------------------------------------------------
Say 'Cipta pintasan "Latif Voice" di Desktop'
$desk = [Environment]::GetFolderPath('Desktop')
$bat = @"
@echo off
title Latif Voice (jangan tutup tetingkap ini semasa guna fon)
wsl.exe -d $distro -- bash -lc "cd ~/vo-studio && bash studio/start_server.sh $Model"
pause
"@
Set-Content -Path (Join-Path $desk 'Latif Voice.bat') -Value $bat -Encoding ASCII

Say 'SIAP!'
Write-Host @"
Cara guna:
 1. Pasang aplikasi Tailscale di fon Android (Play Store) dan log masuk dengan akaun yang sama.
 2. Klik dua kali "Latif Voice" di Desktop laptop.
 3. Buka pautan http://100.x.x.x:8765/#token=... yang dipaparkan, dalam Chrome di fon.
 4. Chrome: menu > "Tambah ke skrin utama".
Laptop mesti hidup (dan dicas) semasa anda guna fon.
"@ -ForegroundColor Green
Read-Host 'Tekan Enter untuk keluar'
