# AI Powered Construction Violation Detection

Full-stack web application for Karachi, Pakistan: citizens submit construction imagery; **SBCA (Sindh Building Control Authority)** staff review AI-assisted reports on a dashboard. **Model training and inference are not included** — connect your detector in `backend/services/ai_service.py` (see below).

## Prerequisites

- **Python 3.11 or 3.12** (strongly recommended). **Python 3.14+** often has no binary wheels yet for `pydantic-core` and **Pillow**, so `pip` may try to compile Rust/C and fail. Use a 3.11/3.12 venv for the backend unless you install Rust and build tooling.
- **Python 3.11+** was stated for flexibility; for reliable installs on Windows, prefer **3.11.x** with `--prefer-binary`.
- **Node.js 18+**

## Project layout

```
Fyp/
├── backend/          # FastAPI, SQLite, JWT, uploads
├── frontend/         # React + Vite + Tailwind
└── README.md
```

## Backend setup

```powershell
cd backend
# On Windows, prefer Python 3.11–3.13 so wheels exist (avoid 3.14 until upstream wheels land):
# py -3.13 -m venv venv
python -m venv venv
.\venv\Scripts\activate
pip install --prefer-binary -r requirements.txt
```

`requirements.txt` pins `bcrypt<5` because **passlib** 1.7.4 is not compatible with **bcrypt** 5.x.

Copy environment file and edit secrets:

```powershell
copy .env.example .env
```

Seed the default **admin** account:

```powershell
python seed.py
```

Default credentials (change after first login in production):

| Field    | Value        |
|----------|--------------|
| Username | `admin`      |
| Password | `Admin@1234` |
| Email    | `admin@example.com` |

Optional (recommended for production): Alembic migrations — initialize and upgrade after adjusting `alembic.ini` / `env.py` to your `DATABASE_URL`.

Start the API:

```powershell
uvicorn main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

### Connecting the AI model

The repo is now wired to load `best_floor.pt` as the default street-view floor detector.

1. Install backend dependencies with `pip install --prefer-binary -r requirements.txt`.
2. Keep `best_floor.pt` in the repo root, or set `AI_STREET_MODEL_PATH` in `backend/.env`.
3. Submit a `Street View` image from the citizen flow to trigger YOLO inference and save an annotated image under `/uploads/...`.
4. Aerial uploads are currently routed to manual review until you add a dedicated setback / encroachment model in `backend/services/ai_service.py`.

### SQLite backup (manual)

Stop the app or copy while idle:

```powershell
copy backend\vioscan.db backend\backups\vioscan_%date%.db
```

Schedule this via Task Scheduler or cron on your server.

### HTTPS / production

Run **uvicorn** (or gunicorn + uvicorn workers) behind **nginx** or another reverse proxy with TLS certificates (Let’s Encrypt). Set `FRONTEND_URL` to your HTTPS origin for CORS.

### Workers

For higher throughput: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4` (no `--reload`).

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Do **not** set `VITE_API_BASE_URL` in dev by default (see `frontend/.env.development`): the UI talks to the same origin (`http://localhost:5173`) and Vite **proxies** `/api` and `/uploads` to **`http://localhost:8000`**. Run the API with `uvicorn main:app --reload --port 8000`. Use `FRONTEND_URL=http://localhost:5173` in `backend/.env`. For `npm run preview`, the same proxy applies; CORS also allows port **4173**.

Production build:

```powershell
npm run build
npm run preview
```

## Environment variables (`backend/.env`)

| Variable                   | Description                                      |
|---------------------------|--------------------------------------------------|
| `SECRET_KEY`              | JWT signing secret (long random string)           |
| `ALGORITHM`               | JWT algorithm (default `HS256`)                   |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime                              |
| `DATABASE_URL`            | SQLAlchemy URL (default SQLite `./vioscan.db`)    |
| `GEOAPIFY_API_KEY`        | Geoapify reverse geocoding (optional; fallback UI if empty) |
| `UPLOAD_DIR`              | Folder for uploaded images                       |
| `MAX_FILE_SIZE_MB`        | Max upload size                                  |
| `AI_STREET_MODEL_PATH`    | Path to the YOLO `.pt` model used for street-view floor detection |
| `AI_STREET_MODEL_CONFIDENCE` | Confidence threshold passed to YOLO prediction |
| `AI_STREET_MODEL_IOU`    | IOU threshold passed to YOLO prediction         |
| `AI_DEVICE`               | Inference device for YOLO (`cpu`, `cuda`, or `auto`) |
| `FRONTEND_URL`            | Allowed CORS origin (e.g. `http://localhost:5173` in local dev) |
| `FRONTEND_DIST`           | Override path to Vite `dist` (optional; default `../frontend/dist`, or `dist` inside a PyInstaller bundle) |
| `NOTICE_REPLY_DAYS`       | Deadline shown on printable notice (default `7`) |
| `NOTICE_OFFICE_LINE1` / `NOTICE_OFFICE_LINE2` / `NOTICE_CONTACT_LINE` | Optional lines printed top-right on the **Printable notice** (leave empty for fill-in blanks) |

The authority report detail page includes **Printable report** / **Download report** — a structured **Authority Screening Report** (HTML): identification tables, location, executive summary, automated screening fields, evidence reference, notes, and follow-up guidance. (`NOTICE_*` env lines still appear in the header.) Print via the browser or save the downloaded `.html` as PDF.

### Which “API keys” do I need?

| Name | Vendor API? | Required? | Purpose |
|------|-------------|-----------|---------|
| **`SECRET_KEY`** | No (your secret) | **Yes** for real deployments | Signs JWT access tokens for staff login. Put a long random string in `backend/.env`. |
| **`GEOAPIFY_API_KEY`** | **Geoapify** | **No** | Optional text hints when GPS falls outside the packaged Karachi boundaries. Primary detection uses `backend/data/karachi_town_bboxes.json` (no API key). Create a key at [Geoapify](https://www.geoapify.com/) if you want this fallback. |

No other third-party API keys are required for the stock app (maps use OpenStreetMap tiles in the browser).

### Karachi GPS → district & town

Reports store **`Town (District)`** (e.g. `Lyari (Karachi South)`). The backend matches latitude/longitude to town boxes in **`backend/data/karachi_town_bboxes.json`**. These boxes are **approximate** and intended for demos; for production accuracy, replace them with official TMC / union-council **polygon GeoJSON** and extend `services/admin_areas_service.py` to test points against polygon geometry (the same `GET /api/geocoding/reverse` and `/api/geocoding/lookup` contract can stay). Where coordinates fall in overlapping boxes, the **smallest** box wins.

### Run UI + API together (single server)

After `npm run build` in `frontend/`, the backend can serve the React app as well as `/api/*`:

```powershell
cd backend
.\venv\Scripts\activate
$env:FRONTEND_URL = "http://127.0.0.1:8000"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** — same origin as the API (no Vite dev proxy needed).

### Windows desktop bundle (EXE)

The packaged app runs **one server**: FastAPI serves **both** the REST API and the **built React UI** from `frontend/dist` embedded under `_internal\dist`. There is no separate “frontend process.”

Build (requires frontend build + PyInstaller):

```powershell
.\scripts\build_exe.ps1
```

Output folder:

`backend\dist\ConstructionViolationDetection\`

**How to run**

1. Copy the **entire `ConstructionViolationDetection` folder** somewhere (USB, Desktop). Do **not** move only `ConstructionViolationDetection.exe` — keep **`_internal\`** next to it.
2. Optionally copy **`backend\.env`** → **`.env`** in that **same folder as `ConstructionViolationDetection.exe`** (for `SECRET_KEY`, optional `GEOAPIFY_API_KEY`). If omitted, the exe still starts with a bundled dev JWT secret (change for real use).
3. Double‑click **`ConstructionViolationDetection.exe`**. It switches to its install folder, starts **http://127.0.0.1:8000**, and tries to **open your browser** automatically after ~2 seconds.
4. Leave the **black console window** open while you use the app — closing it stops the server.

**If nothing opens or it closes instantly**

- Read **`app_error.log`** or **`app_launch.log`** next to `ConstructionViolationDetection.exe`.
- Install **[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)** (x64).
- Ensure port **8000** is free, or set **`VIOSCAN_PORT=8010`** in `.env` (then open `http://127.0.0.1:8010`).
- Prefer rebuilding with a **standalone Python 3.11–3.13** venv (not Conda base) if DLL warnings appeared during PyInstaller.

## API summary

| Prefix            | Purpose                                      |
|-------------------|----------------------------------------------|
| `/api/auth`       | Login, `/me`, logout                         |
| `/api/citizen`    | Submit report, poll status, districts        |
| `/api/geocoding`  | Reverse geocode (Geoapify proxy)             |
| `/api/authority`  | Reports list/detail, stats, map, status PATCH|
| `/api/admin`      | User CRUD (admin only)                       |
| `/uploads/*`      | Static image files                           |

## Frontend routes

| Path                      | Description                    |
|---------------------------|--------------------------------|
| `/`                       | Landing                        |
| `/login`                  | Staff login                    |
| `/citizen`                | 4-step citizen wizard          |
| `/authority/reports`      | Dashboard + reports table      |
| `/authority/reports/:id`  | Report detail + status actions |
| `/authority/map`          | Leaflet map                    |
| `/admin/users`            | User management (admin)        |

## Security notes

- Passwords hashed with **bcrypt** (`passlib`).
- **JWT** Bearer tokens; axios attaches `Authorization` from Zustand-persisted storage.
- **RBAC**: authority routes require login; admin routes require `role === true` (admin).
- **Rate limiting** (`slowapi`): login and citizen report submission limited per IP.
- **Access logging**: request middleware logs method, path, status, duration.

## Recent updates (May 2026)

- **Branding update**: user-facing product name is now **AI Powered Construction Violation Detection** (frontend UI labels, API title, docs, and Windows bundle naming).
- **GPS admin-area mapping**: added GPS-to-`Town (District)` detection for Karachi using `backend/data/karachi_town_bboxes.json` via `services/admin_areas_service.py`.
- **Geocoding behavior**: `/api/geocoding/reverse` and `/api/geocoding/lookup` now return structured district/town labels from local Karachi admin mapping first, with optional Geoapify fallback.
- **Citizen submission flow**: complaint `district_location` now supports mapped area labels (for example, `Lyari (Karachi South)`), improving downstream filtering.
- **Frontend responsiveness**: reduced delay in “Karachi Districts and Towns” step by bundling area labels in frontend constants and removing unnecessary district-list API round-trips in key views.
- **Rule-engine compatibility**: district rule lookup now accepts both full labels (`Town (District)`) and town-only names.
- **Desktop packaging changes**: replaced `backend/vioscan.spec` with `backend/windows_bundle.spec`; bundle output and docs now reference `ConstructionViolationDetection.exe`.
- **Type-checker reliability**: added root `pyrightconfig.json` to point static analysis to `backend/venv`.

## License / disclaimer

Prototype for academic / civic use. Regulatory rules in `rule_engine.py` are **illustrative** — replace with official SBCA data before any production use.
