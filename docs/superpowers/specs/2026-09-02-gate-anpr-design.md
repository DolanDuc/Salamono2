# Moduł bramki wjazdowej — rejestr pojazdów i odczyt tablic (tryb `gate`)

- **Data:** 2026-09-02
- **Branch:** `feat/gate-anpr-vehicle-log`
- **Baza:** `feature/perimetr-v2-2-clean-upgrade` (v2.2 — to, co stoi na Railwayu)
- **Status:** projekt zatwierdzony, przed planem wdrożeniowym

## Skąd to się wzięło

Spotkanie z **Atlas World** (2026-09-02). Dyrektor projektu **Hubert Puza** prowadzi
inwestycję ~300 mln PLN przy rondzie Daszyńskiego (budynek usługowo-mieszkaniowy).
Poza ścieżką BHP zgłosił własną potrzebę:

> detekcja ilości oraz rodzaju pojazdów, które wjeżdżają, a także detekcja godzin
> i numerów rejestracyjnych

To jest osobny use case od BHP — nie zagrożenie, tylko **dziennik ruchu na placu**.
Kupujący też jest inny: nie BHP-owiec, tylko dyrektor projektu / kierownik budowy,
który rozlicza dostawy i pilnuje, kto jest na placu.

## Cel

Kamera przy bramie wjazdowej prowadzi automatyczny dziennik przejazdów: kiedy,
w którą stronę, jaki pojazd, jaki numer rejestracyjny. Z tego lecą raporty dzienne
i tygodniowe dla kierownictwa budowy.

**Horyzont:** moduł produkcyjny na realną bramkę (nie demo na spotkanie).

## Zakres v1

W zakresie:

- trzeci tryb kamery `gate` obok `site` i `checkpoint`
- wykrycie przejazdu z kierunkiem (wjazd / wyjazd)
- klasyfikacja pojazdu w czterech grubych klasach COCO
- odczyt tablicy rejestracyjnej z confidence i ręczną korektą w panelu
- dziennik przejazdów (JSONL) + panel przeglądania
- raport CSV/JSON dla kierownictwa
- retencja i tryb pseudonimizacji tablic (RODO)

Poza zakresem, świadomie:

- **sterowanie szlabanem / kontrola dostępu** — inny produkt i inna odpowiedzialność
  prawna; system, który *otwiera bramę*, odpowiada za to, kogo wpuścił
- **biała lista pojazdów** — wynika z powyższego
- **marka i kolor pojazdu** — nikt o to nie prosił
- **korelacja tablica ↔ dostawca ↔ zamówienie** — to dane z ERP klienta, nie z kamery
- **rozpoznanie typu sprzętu budowlanego** (betonowóz vs wywrotka) — iteracja 2,
  wymaga dotrenowanego modelu; patrz „Ograniczenia, które trzeba zakomunikować"

## Architektura

### Wpięcie w istniejący system

Nowy tryb `mode="gate"` w `FrameResultOut.mode` (dziś `"site" | "checkpoint"`).
Kamera przy bramie deklaruje ten tryb i **nie robi reguł BHP** — pipeline zagrożeń
jest dla niej pominięty. Jeden dodatkowy branch w dispatcherze `backend/routes/ingest.py`,
analogiczny do istniejącej gałęzi `checkpoint`.

`ingest.py` ma na v2.2 ponad 2500 linii, więc **cała logika bramki idzie do osobnego
pakietu** — w `ingest.py` zostaje tylko wywołanie.

```
backend/gate/
├── __init__.py
├── plate_reader.py        # Protocol PlateReader + FastAlprReader + PlateRecognizerReader
├── vehicle_classifier.py  # COCO → etykiety PL
├── passage_tracker.py     # tracker pojazdu, przejście przez bramę, consensus tablicy
└── passage_store.py       # dziennik JSONL (wzór: backend/alert_storage.py)

backend/routes/gate.py     # /api/gate/*, dopięty w main.py z prefiksem /api
frontend/gate.html
frontend/gate.js
config.py                  # GateConfig + odczyt w _from_env()
tools/simulate_gate.py     # odtwarzanie wideo z bramy (wzór: tools/simulate_phone.py)
```

Stan trzymany jak reszta systemu — w `app.state`, budowany w `lifespan()`:
`app.state.plate_reader`, `app.state.passage_tracker`, `app.state.passage_store`.

### Silnik ANPR za interfejsem

```python
class PlateReader(Protocol):
    def read(self, roi: np.ndarray) -> list[PlateRead]: ...
```

Dwie implementacje w v1:

| Implementacja | Silnik | Kiedy |
|---|---|---|
| `FastAlprReader` | [`fast-alpr`](https://ankandrew.github.io/fast-alpr/latest/) v0.4.0, ONNX, CPU | domyślna |
| `PlateRecognizerReader` | Plate Recognizer SDK on-prem (docker, HTTP) | fallback |

Wybór przez `GATE_PLATE_ENGINE=fastalpr|platerecognizer`. Powód dwóch implementacji:
jakość odczytu zależy od konkretnej kamery przy konkretnej bramie i **nie da się jej
przewidzieć zza biurka**. Pilot u Atlasa jest pomiarem — jeśli `fast-alpr` na ich
sprzęcie nie wyrabia, podmieniamy zmienną środowiskową, a nie pipeline.

`fast-alpr` jest domyślny, bo nie generuje kosztu bieżącego ani zależności od
zewnętrznego dostawcy w produkcie sprzedawanym jako compliance-by-design.
Plate Recognizer to od $50/mies. za 50 tys. odczytów, koszt rośnie liniowo z liczbą placów.

**Odrzucone: kamera ANPR sprzętowa (Hikvision/Dahua z ISAPI).** Najwyższa niezawodność,
bo optyka i shutter są projektowane pod tablice, ale wymaga kupna dedykowanej kamery
(~3–5k PLN/szt.) i tym samym łamie GTM Perimetru — sprzedajemy software na kamery,
które klient już ma. Interfejs `PlateReader` zostawia tę drogę otwartą, gdyby klient
sam chciał dołożyć sprzęt.

### Kierunek przejazdu — dwie strefy, nie linia

Kierunek wynika z przejścia między dwiema strefami, a nie z przecięcia linii tripwire.

Do istniejącego modelu `Zone` (`backend/zones_store.py`) dochodzi jedno pole:

```python
role: str = ""     # "" (zwykła strefa BHP) | "gate_outside" | "gate_inside"
```

Pole z wartością domyślną nie łamie istniejących `zones.json` ani reguł BHP,
a rysowanie stref, `ZoneStore` i `frontend/zones.js` działają bez zmian —
operator rysuje dwie strefy tym samym narzędziem, którego już używa.

- `gate_outside → gate_inside` = **wjazd** (`direction: "in"`)
- `gate_inside → gate_outside` = **wyjazd** (`direction: "out"`)

Koszt tej decyzji: przy każdej bramie trzeba narysować dwie strefy zamiast jednej linii.
Zysk: zero nowego kodu do rysowania, przechowywania i walidacji geometrii.

### Ścieżka jednej klatki w trybie `gate`

```
YOLO detect (klasy pojazdowe COCO)
  → passage_tracker.update()          # dopasowanie po IoU, jak w vehicle_motion._Track
  → dla tracka w kadrze i dość dużego:
       plate_reader.read(ROI bboxa)   # NIE na całej klatce
       → bufor odczytów tracka
  → sprawdzenie strefy środka bboxa (gate_outside / gate_inside)
  → przy zmianie strefy: oznacz kierunek, uzbrój zamknięcie
  → przy zamknięciu: consensus tablicy → PassageRecord
       → passage_store.append() → broadcast /ws/live
```

**Odczyt tablicy tylko na ROI bboxa pojazdu** i tylko gdy bok bboxa przekracza
`min_plate_box_px` — inaczej palilibyśmy CPU na pojazd na końcu ulicy, którego
tablicy i tak nie widać.

**Consensus zamiast pojedynczego odczytu.** Przejeżdżający pojazd daje kilkanaście
odczytów o różnej jakości. Przy zamknięciu zdarzenia wybieramy tekst o najwyższej
sumie confidence spośród wszystkich odczytów; `plate_confidence` to średnia dla
zwycięskiego tekstu, `plate_reads` to liczba klatek, które go potwierdziły.
Dzięki temu jeden pojazd = jeden rekord, a nie trzydzieści.

**Zamknięcie zdarzenia:** track przekroczył granicę stref i zniknął z kadru na
`close_after_sec` (domyślnie 3 s).

### Model danych

```python
class PassageRecord(BaseModel):
    id: str
    timestamp: float                      # moment przekroczenia granicy stref
    camera_id: str
    direction: str                        # "in" | "out"
    vehicle_class: str                    # COCO: car | truck | bus | motorcycle
    vehicle_label: str                    # PL: osobowy | ciężarowy | autobus | motocykl
    plate: str | None
    plate_confidence: float               # 0.0–1.0
    plate_reads: int
    plate_corrected_by: str | None = None  # ślad korekty ręcznej w panelu
    thumbnail_url: str | None = None
    details: dict = {}                    # m.in. rejected_reads, track_id, czas w kadrze
```

Persystencja: `data/passages.jsonl`, append-only, cache ostatnich rekordów w RAM —
dokładnie wzór `AlertStore` z `backend/alert_storage.py`.

### API

| Endpoint | Opis |
|---|---|
| `GET /api/gate/passages` | dziennik z filtrami `from`, `to`, `direction`, `camera_id`, `has_plate` |
| `PATCH /api/gate/passages/{id}` | ręczna korekta odczytu tablicy (zapisuje `plate_corrected_by`) |
| `GET /api/gate/report` | raport zbiorczy, `format=csv\|json` |

`routes/reports.py` ma już wzorce `report_summary` i `export_csv` — raport bramki
idzie tą samą konwencją.

Raport zawiera: liczbę wjazdów i wyjazdów per dzień, rozkład godzinowy, podział na
typy pojazdów, oraz **pojazdy, które wjechały i nie wyjechały**. Ta ostatnia pozycja
jest jedyną w raporcie, która jest realnym wskaźnikiem operacyjnym, a nie ciekawostką —
reszta to liczniki.

### Konfiguracja

`GateConfig` w `config.py`, czytany w `_from_env()` (konwencja repo: żadnego
`os.getenv` w logice):

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `GATE_PLATE_ENGINE` | `fastalpr` | `fastalpr` \| `platerecognizer` |
| `GATE_PLATE_MIN_CONFIDENCE` | `0.5` | próg przyjęcia odczytu |
| `GATE_MIN_PLATE_BOX_PX` | `120` | minimalny bok bboxa, by w ogóle czytać |
| `GATE_CLOSE_AFTER_SEC` | `3.0` | ile bez tracka przed zamknięciem zdarzenia |
| `GATE_MAX_PASSAGE_SEC` | `120.0` | twardy limit trwania jednego przejazdu |
| `GATE_RETENTION_DAYS` | `30` | retencja miniatur i (opcjonalnie) tablic |
| `GATE_PLATE_STORAGE` | `plain` | `plain` \| `hashed` |

## Obsługa błędów

| Sytuacja | Zachowanie |
|---|---|
| Brak modelu ANPR / reader niedostępny | tryb `gate` działa dalej, `plate=None`, przejazd liczy się do statystyk. Wzór: opcjonalny `ppe.pt`, który po prostu wyłącza tryb `checkpoint` |
| Wszystkie odczyty poniżej progu | `plate=None` + `details.rejected_reads` z odrzuconymi tekstami — żeby dało się zdiagnozować, *dlaczego* bramka nie czyta, zamiast zgadywać |
| Pojazd stoi w bramie (rozładunek) | track żyje, dopóki bbox jest w kadrze; `max_passage_sec` wymusza zamknięcie, żeby zdarzenie nie wisiało w nieskończoność |
| Dwa pojazdy jednocześnie | tracker per-track → dwa rekordy. Przy wzajemnym zasłonięciu IoU może je skleić; akceptujemy to w v1 i logujemy w `details` |
| Pojazd podjeżdża do bramy i zawraca | brak zmiany strefy → brak rekordu. Pokryte testem |

## Testy

Konwencja repo: testy podmieniają `app.state` fixture'em z `tmp_path` i mockują
detektor — **nie ładujemy modeli w testach**, ani YOLO, ani ONNX.

- `tests/test_gate_tracker.py` — syntetyczne sekwencje bboxów przez strefy: wjazd,
  wyjazd, zawrócenie bez przekroczenia, dwa pojazdy naraz, pojazd stojący ponad
  `max_passage_sec`
- `tests/test_plate_consensus.py` — bufor odczytów → wynik: zgodne odczyty, dwa różne
  teksty po połowie, wszystkie poniżej progu, bufor pusty
- `tests/test_gate_api.py` — `POST /api/frame` z `mode=gate` przy `FakePlateReader`,
  filtry dziennika, `PATCH` korekty
- `tools/simulate_gate.py` — odtwarzanie nagrania z bramy zamiast sprzętu

## RODO

Numer rejestracyjny pośrednio identyfikuje osobę, więc jest daną osobową. W module,
nie w dokumentacji obok:

- `GATE_RETENTION_DAYS` — zadanie czyszczące usuwa miniatury po terminie
- `GATE_PLATE_STORAGE=hashed` — zapisujemy wyłącznie HMAC numeru. Nadal da się
  policzyć „ten sam pojazd wjechał dziś 5×", bez przechowywania samego numeru
- treść tabliczki informacyjnej przy bramie (obowiązek informacyjny) → do
  `4_perimetr/regulations.md` w vaulcie

Dla klienta, który kupuje od nas compliance-by-design, to jest argument sprzedażowy,
a nie koszt.

## Ograniczenia, które trzeba zakomunikować Atlasowi przed wdrożeniem

Obie pozycje mówimy **na spotkaniu**, nie po wdrożeniu.

**1. „Rodzaj pojazdu" ≠ rodzaj sprzętu budowlanego.** YOLO11n na COCO rozróżnia
`car / truck / bus / motorcycle`. Nie odróżni betonowozu od wywrotki ani nie policzy
gruszek. Rozpoznanie sprzętu budowlanego wymaga dotrenowanego modelu — to ta sama
ścieżka, którą zespół już idzie przy detekcji upadków, ale to iteracja 2.

**2. Odczyt tablicy nie będzie stuprocentowy.** Kamera dozorowa ma rolling shutter
i długi czas naświetlania; kamera ANPR ma optykę zaprojektowaną pod tablice.
Nocą i przy większej prędkości wjazdu część odczytów będzie pusta lub błędna.
Dlatego panel pokazuje `confidence` i pozwala poprawić rekord ręcznie.
Dziennik z widoczną jakością odczytu jest sprzedawalny. Obietnica 100% nie jest —
zweryfikuje ją pierwszy tydzień pilota.

## Otwarte przed planem wdrożeniowym

- Jaka kamera stoi (lub stanie) przy bramie u Atlasa — model, rozdzielczość,
  doświetlenie IR, kąt do tablicy. Od tego zależy, czy `fast-alpr` wystarczy
- Czy plac ma jedną bramę, czy osobne wjazd i wyjazd (druga brama = druga kamera
  w tym samym trybie, moduł jest per-kamera, więc działa bez zmian)
