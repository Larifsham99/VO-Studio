# Latif Voice: guna dari fon Android

PC di rumah yang menjana suara (perlu GPU NVIDIA). Fon hanya alat kawalan:
anda taip teks, pilih emosi, dengar dan muat turun MP3/WAV. Tiada caj ikut
aksara, dan semua suara serta rakaman kekal dalam PC anda.

```
Fon Android (Chrome)  ──Wi-Fi rumah / Tailscale──▶  PC: server + model AI (GPU)
```

> **Status: prototaip.** Model mana yang paling bagus untuk bahasa Melayu
> belum dipastikan. Jalankan benchmark (`benchmark/README.md`) untuk
> mengesahkannya. Server menggunakan VoxCPM2 secara lalai, dan model boleh
> ditukar dalam tab **Status**.

## 1. Pasang di PC (sekali sahaja)
Gunakan Linux, atau **WSL2 Ubuntu** di Windows. Anda perlukan Python 3.11,
git, FFmpeg dan pemacu NVIDIA.
```bash
git clone https://github.com/larifsham99/vo-studio.git
cd vo-studio
git checkout ccr-e9aac4d2-pw4crp
sudo apt install -y ffmpeg python3.11 python3.11-venv
bash studio/setup_server.sh
```
Jika ada ralat CUDA, jalankan semula dengan
`TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash studio/setup_server.sh`.

## 2. Daftar dua suara tetap (untuk Mod B)
```bash
cp benchmark/refs/manifest.example.yaml benchmark/refs/manifest.yaml
```
Letakkan rakaman dalam `benchmark/refs/voices/...` dan isi
`manifest.yaml`. Isi juga medan `consent`. Panduan rakaman ada dalam
`benchmark/refs/README.md`.

## 3. Sambungan fon ke PC dengan Tailscale (percuma, boleh dari luar rumah)
1. Pasang **Tailscale** di PC. Untuk WSL2, pasang dalam Ubuntu:
   `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`.
2. Pasang aplikasi **Tailscale** dari Play Store di fon dan log masuk dengan
   akaun yang sama.

Jika anda hanya guna di rumah, Tailscale tidak diperlukan. Sambungkan fon
ke Wi-Fi yang sama dengan PC.

## 4. Mulakan server
```bash
bash studio/start_server.sh
```
Terminal akan memaparkan pautan seperti
`http://100.x.x.x:8765/#token=...`. Buka pautan itu dalam **Chrome di fon**,
kemudian pilih menu ⋮ → **Tambah ke skrin utama**. Selepas itu aplikasi
boleh dibuka seperti aplikasi biasa.

**Jangan kongsi pautan itu**, kerana token di dalamnya memberi akses penuh.
Untuk menukar token, padam `studio/data/token.txt` dan mulakan semula server.

## Cara guna aplikasi
- **Projek → + Projek baharu.** Pilih:
  - **Mod B**: salah satu suara tetap anda. Boleh ditambah gaya daripada
    rakaman dalam *Tetapan lanjut*.
  - **Mod A**: tiru suara dan gaya daripada satu rakaman rujukan yang
    dibenarkan.
- **Taip teks** dan ketik butang emosi (Teruja, Sedih, Berbisik, Ketawa…)
  untuk memasukkan tag seperti `[excited]`. Tag menjadi arahan emosi kepada
  model dan **tidak dibaca kuat**.
- **Jana suara.** Bar kemajuan akan dipaparkan. Bila siap, dengar terus atau
  muat turun **WAV / MP3**.
- **Rakaman.** Muat naik atau rakam terus di fon (m4a, mp3, wav, ogg semua
  boleh). Tempoh 3–60 saat, terbaik 10–20 saat. Medan kebenaran wajib diisi.
  Server menganalisis tempo, variasi nada, jeda dan tekanan dalam rakaman.
- **Status.** Keadaan PC dan GPU, model aktif, dan butang untuk menukar
  model.

## Had yang perlu diketahui
- PC mesti hidup dan server berjalan semasa anda guna fon.
- Butang rakam terus dalam Chrome menggunakan perakam fon melalui pemilih
  fail, kerana mikrofon dalam pelayar memerlukan HTTPS. Jika mahu HTTPS,
  gunakan `tailscale serve`.
- Satu kerja dijana pada satu masa. Kerja lain menunggu giliran.
- Mod B dengan rakaman gaya memerlukan VoxCPM2. Chatterbox tidak menyokong
  pemindahan gaya.

## Untuk pembangun
- Server: `studio/server/` (FastAPI). Ujian: `pytest studio/server/tests`.
- Aplikasi fon: `studio/web/` (React + Vite). `dist/` yang sudah dibina
  disertakan, jadi PC tidak perlukan Node.js. Untuk mengubah UI, jalankan
  `npm install && npm run build`.
