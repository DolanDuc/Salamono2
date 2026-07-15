# Perimetr — MVP

System bezpieczenstwa na budowie oparty o AI i wizje komputerowa.
Automatycznie wykrywa niebezpieczne sytuacje (pracownik za blisko pojazdu,
wejscie w strefe niebezpieczna, brak PPE na bramce) i alarmuje kierownika
budowy w czasie rzeczywistym. Repo nadal fizycznie nazywa sie `Salamono2` —
przemianowanie po rejestracji domeny/TM.

## Architektura

```
[Telefon/kamera]  ──HTTP POST──>  [Backend FastAPI + YOLO]  ──WebSocket──>  [Panel web]
   2-3 kl/s                        detekcja obiektow                        podglad na zywo
                                   analiza zagrozen                         alarm + historia
                                   filtr temporalny
```

## Szybki start

### 1. Instalacja

```bash
pip install -r requirements.txt
```

### 2. Etap 0 — Test na nagraniu

```bash
python etap0/detect_video.py --input wideo.mp4 --output wynik.mp4
```

Opcje:
- `--sample-fps 3` — klatki na sekunde do analizy (domyslnie 3)
- `--model yolo11n.pt` — model YOLO (domyslnie nano)
- `--show` — podglad na zywo (wymaga GUI)

### 3. Etap 1 — System real-time

**Uruchom serwer:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Otworz panel:** `http://localhost:8000`

**Uruchom symulator (zamiast telefonu):**
```bash
python tools/simulate_phone.py --video wideo.mp4 --fps 2 --loop
```

**Lub uzyj telefonu:** Otworz `http://<adres-serwera>:8000/phone/capture.html`
na telefonie w tej samej sieci.

## Multi-camera: wspolne strefy + fuzja pozycji

Kilka kamer z roznych katow patrzy na te same strefy — pozycje osob sa
fuzowane we wspolnym ukladzie metrycznym (dokladniejsza odleglosc, jeden
alarm zamiast N).

**Procedura (te same 4 markery ArUco dla wszystkich kamer):**

1. Rozloz 4 markery (np. ID 10, 20, 30, 40) w rogach obszaru
   referencyjnego o znanych wymiarach (np. 3×3 m).
2. Skalibruj KAZDY telefon/kamere osobno na `/calibrate.html`
   (albo `POST /api/calibration/{camera_id}`) — kazda kamera dostaje
   wlasna homografie do TEJ SAMEJ plaszczyzny metrycznej.
3. Zaloz strefe world-space (metry, klucz `_site`):

```bash
curl -X PUT http://localhost:8000/api/zones/_site \
  -H "Content-Type: application/json" \
  -d '{"zones":[{"name":"Wykop","severity":"DANGER",
       "coordinate_space":"world",
       "polygon":[[1.0,1.0],[4.0,1.0],[4.0,3.0],[1.0,3.0]]}]}'
```

4. Panel pokaze karte **Mapa placu** — widok z gory: strefy w metrach,
   sfuzowane kropki osob (badge = liczba kamer, czerwona przy breachu).

**Uwagi:**
- Strefy trzymaj blisko prostokata referencyjnego — blad homografii
  rosnie z odlegloscia od markerow. Prog klastrowania:
  `WORLD_ASSOC_THRESHOLD_M` (domyslnie 0.7 m; na duzych dystansach 1.0).
- Kamery moga byc niezsynchronizowane (skew ~0.5 s jest OK — TTL fuzji
  liczy sie po czasie serwera).
- Alarm multi-cam ma `kind=world_zone_breach`, `camera_id=site`,
  kamery zrodlowe w `details.cameras` — jeden rekord niezaleznie od
  liczby kamer.

**Proba generalna bez sprzetu** (syntetyczne markery + 2 wirtualne kamery):
```bash
python tools/simulate_two_cameras.py --server http://localhost:8000
```

## Konfiguracja

Parametry mozna ustawic przez zmienne srodowiskowe:

| Zmienna | Domyslnie | Opis |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | Model YOLO |
| `YOLO_CONFIDENCE` | `0.35` | Prog pewnosci detekcji |
| `YOLO_DEVICE` | `cpu` | Urzadzenie (`cpu` lub `cuda:0`) |
| `DANGER_PROXIMITY_PX` | `50` | Odleglosc w pikselach = "za blisko" |
| `DANGER_CONSECUTIVE_FRAMES` | `3` | Ile klatek z rzedu = alarm |
| `DANGER_COOLDOWN_SEC` | `10` | Przerwa miedzy powtornymi alarmami |
| `SERVER_PORT` | `8000` | Port serwera |
| `PANEL_PASSWORD` | `` (off) | Jesli ustawione, panel wymaga HTTP basic auth |
| `DEMO_TOKEN` | `` (off) | Jesli ustawione, `?demo=<TOKEN>` omija auth i ustawia cookie |

### Demo w iframe (pitch)

Zeby wkleic panel jako iframe do slajdu pitcha:

1. Na Railway ustaw `DEMO_TOKEN` na losowy sekret, np.:
   ```bash
   openssl rand -hex 24
   ```
2. W slajdzie HTML wklej:
   ```html
   <iframe src="https://twoj-railway.up.railway.app/?demo=TWOJ_TOKEN"
           style="width:100%;height:100%;border:0"
           allow="camera; microphone; autoplay"></iframe>
   ```
3. Pierwszy request ustawia cookie `perimetr_demo` (12h, `SameSite=None; Secure`),
   wiec kolejne fetchi assetow i WebSocket przechodza bez query paramu.

CSP `frame-ancestors *` jest ustawiane przez backend automatycznie,
aby iframe dzialal z dowolnego origin.

**Bezpieczenstwo:** token to `secrets.compare_digest`-owy check, ale ma tylko
jeden poziom (wszystko-albo-nic). Do prod z wieloma uzytkownikami dodaj OAuth
albo per-user access tokens.

## API

| Endpoint | Metoda | Opis |
|---|---|---|
| `/api/frame` | POST | Wyslij klatke (multipart: image + camera_id) |
| `/api/alerts` | GET | Historia alarmow |
| `/api/zones/{camera_id}` | GET/PUT/DELETE | Strefy per kamera (`_site` = world-space, metry) |
| `/api/calibration` | GET | Lista skalibrowanych kamer |
| `/api/calibration/{camera_id}` | POST/GET/DELETE | Kalibracja kamery z 4 markerow |
| `/api/world/state` | GET | Sfuzowany stan placu (osoby w metrach, strefy, kamery) |
| `/api/stats` | GET | Statystyki systemu |
| `/api/health` | GET | Health check |
| `/ws/live` | WebSocket | Stream wynikow do panelu |

## Testy

```bash
pytest tests/
```

## Struktura projektu

```
etap0/          — proof of concept na nagraniu
backend/        — serwer FastAPI + detekcja + logika zagrozen
frontend/       — panel web kierownika budowy
phone/          — strona do przechwytywania z kamery telefonu
tools/          — narzedzia do testowania (symulator kamery)
config.py       — centralna konfiguracja
```

## Wdrozenie w chmurze (Docker)

### Szybki start z Docker (CPU)

```bash
cp .env.example .env        # dostosuj konfiguracje
docker compose up --build
```

Serwer dostepny na `http://<adres-ip>:8000`

### Z GPU (produkcja / wiele kamer)

Wymaga [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
cp .env.example .env
# Ustaw w .env: YOLO_DEVICE=cuda:0
docker compose --profile gpu up --build
```

### Z HTTPS (wymagane dla kamery przez internet)

Przegladarki blokuja dostep do kamery (`getUserMedia`) na stronach bez HTTPS.
Dwie opcje:

**Opcja A — Cloudflare Tunnel (najlatwiej, darmowe):**
```bash
# 1. Uruchom backend
docker compose up -d

# 2. Zainstaluj cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# 3. Stworz tunel (darmowe, bez domeny)
cloudflared tunnel --url http://localhost:8000
# Dostaniesz URL typu https://xxx-xxx.trycloudflare.com
```

**Opcja B — Nginx + Let's Encrypt (wlasna domena):**
```bash
# 1. Umiesc certyfikaty w nginx/certs/
#    fullchain.pem + privkey.pem (np. z certbot)
# 2. Uruchom z profilem https
docker compose --profile https up --build
```

### Rekomendowane VM do GPU

| Dostawca | GPU | Koszt | Uwagi |
|---|---|---|---|
| Runpod | RTX A4000 | ~$0.20/h | On-demand, latwy start |
| Vast.ai | RTX 3060 | ~$0.15/h | Najtanszy, auction-based |
| Hetzner | GTX 1080 | ~40 EUR/mies. | Staly serwer, EU |
| Lambda | A10 | ~$0.60/h | Stabilny, US |

Z GPU: YOLO przetwarza klatke w ~5-10ms (vs ~150ms na CPU).
Jeden GPU obsluguje 5-10 kamer jednoczesnie przy 3 fps.

## Reguly zagrozen (MVP)

1. **person_vehicle_overlap** — bounding box osoby naklada sie z pojazdem (DANGER)
2. **person_near_vehicle** — osoba w odleglosci < 50px od pojazdu (WARNING)

Filtr temporalny: alarm dopiero po 3 kolejnych klatkach z zagrozeniem,
co eliminuje falszywe alarmy z pojedynczych bledow detekcji.
