# Testy akwizycji na sprzęcie — 2026-07-30

Notatka z pierwszego realnego podłączenia **dwóch kamer przemysłowych** (zestaw
Piotra Garbata) do panelu perimetr na Macu. Branch jest **tymczasowy** — zapis stanu
z sesji terenowej, nie kandydat do mergowania na produkcję.

Hasła i realne `cameras.json` **nie są tu zapisane** — repo jest publiczne. Konfiguracja
z credentialami zostaje lokalnie w `data/cameras.json` (gitignored, wzór w
`data/cameras.example.json`).

---

## Sprzęt

| | Kamera A | Kamera B |
|---|---|---|
| Model | Hikvision **DS-2CD2T46G1-4I** (4 MP AcuSense) | Axis (starszy, model nieustalony) |
| Firmware | V5.5.71 | stare — brak VAPIX (`/axis-cgi/param.cgi` → 404) |
| Adresacja | **statyczna** `192.168.112.111` | **link-local** `169.254.x.x` (losowana!) |
| MAC | `F8:4D:FC:7E:9B:F0` | `00:40:8C:8D:80:96` |
| Kodek | **H.264 Main** | **MPEG-4 Part 2** (`mp4v`) |
| Rozdzielczość | **2688×1520** (po zmianie; było 1920×1080) | 640×480 |
| Klatkaż | **25 fps** (po zmianie; było 10 fps) | 9,4 fps |
| RTSP | `rtsp://<user>:<pass>@192.168.112.111` (bez ścieżki) | `rtsp://<user>:<pass>@<ip>/mpeg4/media.amp` |

**Kamera A jest wyraźnie lepszym materiałem na demo** — 4 MP vs 0,3 MP i H.264 vs
`mp4v`, który wymaga konwersji do podglądu (patrz niżej).

---

## Sieć — dwa różne tryby, jedna przejściówka

Dongle USB-C→Ethernet zgłasza się jako `en6` („USB 10/100 LAN"). Obie kamery
wymagają **innego** trybu, więc na jednym donglu naraz działa tylko jedna:

```bash
# Kamera A — statyczne IP w podsieci kamery
networksetup -setmanual "USB 10/100 LAN" 192.168.112.110 255.255.255.0

# Kamera B — link-local; macOS nadaje 169.254.x.x sam, wystarczy DHCP
networksetup -setdhcp "USB 10/100 LAN"
```

Ustalenia, które kosztowały najwięcej czasu:

- **`sudo` nie jest potrzebne** do `networksetup -setmanual` — wystarcza członkostwo
  w grupie `admin` (`id -Gn | grep admin`). Wcześniejsza blokada wynikała z braku
  konta admina, nie z samego mechanizmu.
- **Kamery ignorują ping na broadcast** (`icmp_echo_ignore_broadcasts=1` — to Linuksy).
  Wykrywanie „w ciemno" sweepem nic nie zwraca, choć kamera jest osiągalna.
  Namierzaj po **ARP**: `arp -an | grep en6`.
- **Link-local nie przechodzi przez router.** Kamera B wpięta w router była widoczna
  po Wi-Fi tylko dopóki Mac siedział w tym samym segmencie L2. Po przesiadce na inną
  sieć Wi-Fi zniknęła — to nie awaria, to definicja link-local.
- Adres link-local **losuje się po restarcie kamery** → po każdym reboocie trzeba
  odczytać nowy z ARP i podmienić `rtsp_url`. Dlatego docelowo kamera B powinna dostać
  statyczne `192.168.112.112` w swoim web panelu (blokuje brak hasła do weba — patrz
  „Otwarte kwestie").

Weryfikacja połączenia bez zgadywania:

```bash
arp -an | grep en6                  # jakie MAC-i są na kablu
nc -z -G 2 <ip> 554                 # RTSP otwarty?
POST /api/recorder/test {"camera_id": "..."}   # albo przez panel
```

---

## Zmiany wprowadzone w kamerze A (ISAPI)

Kamera przyjechała ustawiona **poniżej swoich możliwości** i z zepsutym zegarem.
Oba problemy naprawione przez ISAPI (digest auth; Basic nie działa).
**Oryginał kanału zabezpieczony** przed zmianą w `data/camera_backups/` (gitignored).

### 1. Zegar — `01-01-1970` w OSD

Kamera miała `timeMode=NTP`, a stoi w izolowanej podsieci bez internetu → NTP nigdy
nie odpowiedział. Nagrania bez timestampu są bezużyteczne przy analizie zdarzeń BHP.

```bash
# Strefa — uwaga: Hikvision ODWRACA znak. "CST-2:00:00" znaczy UTC+2.
curl --digest -u "$U" -X PUT "$H/ISAPI/System/time/timeZone" \
  -H "Content-Type: text/plain" --data "CST-2:00:00"

# Czas z komputera + tryb manualny
NOW=$(date "+%Y-%m-%dT%H:%M:%S%z" | sed 's/\(..\)$/:\1/')
curl --digest -u "$U" -X PUT "$H/ISAPI/System/time" \
  -H "Content-Type: application/xml" --data \
  "<Time version=\"2.0\" xmlns=\"http://www.hikvision.com/ver20/XMLSchema\">
   <timeMode>manual</timeMode><localTime>$NOW</localTime>
   <timeZone>CST-2:00:00</timeZone></Time>"
```

**Pułapka z DST:** firmware V5.5.71 trzyma regułę DST w stringu strefy, ale jej
**nie stosuje** — i nie ma przełącznika w ISAPI (`/System/time/dstEnable` → `Invalid
Operation`). Skutek: przy `CST-1` kamera pokazywała godzinę za wcześnie. Rozwiązanie
to ustawianie strefy **na sztywno do aktualnego offsetu komputera** (`CST-2` latem,
`CST-1` zimą) — powtórzenie sync po zmianie czasu samo naprawia offset.

**Zegar jest w trybie manualnym, więc uciecze po odcięciu zasilania kamery.**
Trzeba go wtedy wstrzyknąć ponownie. (Automatyzacja tego — moduł ISAPI wołany przy
starcie nagrywania — była zaczęta i **świadomie porzucona**; setup może się nie
utrzymać, więc nie warto go zaszywać w kodzie.)

### 2. Jakość strumienia — było 1080p/10 fps

Kamera zgłasza w `capabilities`:

```
videoResolutionWidth  opt="1280,1920,2688"
videoResolutionHeight opt="720,1080,1520"
maxFrameRate          opt="2500,2200,2000,...,100,50,25,12,6"   # setne fps
```

czyli potrafi **2688×1520 @ 25 fps**, a nadawała 1920×1080 @ 10 fps — marnowała
60% klatek i połowę pikseli. Zmiana: `GET /ISAPI/Streaming/channels/101`, podmiana
pól, `PUT` całego dokumentu (ISAPI wymaga pełnego XML-a):

| Pole | Było | Jest |
|---|---|---|
| `videoResolutionWidth` / `Height` | 1920 / 1080 | **2688 / 1520** |
| `maxFrameRate` | 1000 | **2500** (= 25 fps) |
| `constantBitRate`, `vbrUpperCap` | 6144 | 16384 |

Potwierdzone na żywym strumieniu, nie z metadanych panelu:

```
2688x1520 [SAR 1:1 DAR 168:95], 4842 kb/s, 25.12 fps
```

**Objętość zapisu: ~36 MB/min = ~2,1 GB/h** (było ~915 MB/h przy 1080p/10 fps).
Przy dłuższej akwizycji w terenie trzeba to policzyć pod dysk.

---

## Nagrania — `mp4v` nie odtwarza się w QuickTime

Zapis idzie `-c copy` (surowy strumień, bez re-enkodowania), więc kodek kamery
trafia do pliku bez zmian. **Kamera A (H.264) otwiera się w QuickTime normalnie.
Kamera B (`mp4v`) nie** — QuickTime nie ma dekodera MPEG-4 Part 2. Plik jest zdrowy,
tylko nieodtwarzalny; objaw to przekreślony trójkąt przy poprawnie pokazanym czasie.

```bash
# Kontrola, czy plik jest OK
ffmpeg -i cam112lan_*.mp4     # → Duration 00:00:12.08, 640x480, mpeg4 (ASP)

# Konwersja do przeglądania (oryginał zostaje nietknięty)
ffmpeg -i wejscie.mp4 -c:v libx264 -preset veryfast -crf 20 \
       -pix_fmt yuv420p -movflags +faststart wyjscie_h264.mp4
```

Pipeline detekcji `mp4v` **nie przeszkadza** — OpenCV dekoduje go bez problemu.
Problem dotyczy wyłącznie podglądu. **VLC nie jest zainstalowany na Macu Adama**,
choć runbook każe nim weryfikować strumień w terenie — do doinstalowania.

---

## Obserwacje z realnego kadru (istotne dla detekcji)

Nagranie testowe z kamery A objęło osobę **w kamizelce odblaskowej** i **marker
ArUco** — czyli dokładnie to, co pipeline ma wykrywać. Trzy rzeczy widać od razu:

1. **Silny backlight** — osoba pod okno wychodzi niemal sylwetką; kamizelka pozostaje
   czytelna (kolor się przebija), reszta postaci tonie w cieniu. To realny scenariusz
   na budowie i **najtrudniejszy przypadek dla YOLO**. W panelu kamery jest
   WDR / BLC / HLC (Image → Backlight Settings) — do włączenia i przetestowania
   **przed** wyjazdem, nie na miejscu.
2. **Wyraźna dystorsja beczkowa** szerokiego kąta (widoczna na krawędziach kadru) →
   kalibracja potrzebuje **współczynników dystorsji**, nie tylko homografii.
3. Pierwsza klatka po otwarciu strumienia bywa **przepalona na biało** — auto-exposure
   potrzebuje kilku sekund. Nie traktować jako awarii; do probe'owania brać klatkę
   z kilkusekundowym offsetem.

---

## Otwarte kwestie

- [ ] **Hasło do web panelu kamery B** — `root:perim` działa na RTSP, ale HTTP zwraca
      401. Bez niego nie da się jej przestawić na statyczne `.112` ani podbić
      rozdzielczości powyżej 640×480. Do wyciągnięcia od Piotra.
- [ ] **WDR/BLC na kamerze A** — włączyć i sprawdzić detekcję w kontrze.
- [ ] **Obie kamery naraz** — wymaga switcha albo przestawienia B na `192.168.112.112`
      (`route add -net 169.254.0.0/16 -interface en6` obchodzi to tymczasowo, ale
      wymaga `sudo` i nie przetrwa restartu).
- [ ] **Nagrywanie obrazu z telefonu** tak jak z kamery RTSP — telefon wysyła JPEG-i
      na `POST /api/frame`, trzeba je składać w MP4 (`-f image2pipe`). Nie zrobione.
- [ ] Zapas dysku pod ~2,1 GB/h przy dłuższej akwizycji.
