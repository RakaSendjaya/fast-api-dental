# DentiScan FastAPI Deploy

Panduan singkat untuk deploy FastAPI inference service ke Railway dan menghubungkannya ke Vercel.

Files penting:
- [main.py](main.py) - FastAPI app dengan endpoint `POST /predict`
- [model.keras](model.keras) - model TensorFlow yang ikut di-push ke repo
- [requirements.txt](requirements.txt) - dependency Python
- [Procfile](Procfile) - command start untuk Railway
- [runtime.txt](runtime.txt) - versi Python untuk Railway
- [.railwayignore](.railwayignore) - file yang diabaikan saat deploy

## 1) Deploy ke Railway

1. Pastikan repo sudah ter-push ke GitHub dan file `model.keras` ada di root folder FastAPI.
2. Buka Railway → buat project baru → pilih **Deploy from GitHub repo**.
3. Pilih repository ini.
4. Set **Root Directory** ke folder FastAPI kalau FastAPI berada di subfolder. Jika service FastAPI ada di root repo, biarkan kosong.
5. Railway akan membaca `Procfile` dan `runtime.txt`.
6. Setelah deploy sukses, salin URL publik Railway, misalnya:

```text
https://nama-project.railway.app
```

## 2) Set `AI_API_URL` di Vercel

1. Buka Vercel Dashboard → Project Next.js kamu.
2. Masuk ke **Settings** → **Environment Variables**.
3. Tambahkan variable berikut:

```text
AI_API_URL=https://nama-project.railway.app
```

4. Simpan lalu redeploy project Next.js agar environment variable baru terbaca.

## 3) Catatan penting

- `main.py` akan load model dari `./model.keras`.
- `Procfile` menjalankan server dengan:

```text
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

- `.railwayignore` mencegah file cache dan `.env` ikut terkirim.
- Jika Railway gagal build karena TensorFlow, biasanya masalah ada di memori atau kompatibilitas wheel Python.