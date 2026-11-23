# Fotobudka – selekcja zdjęć po sesji

Szybki program na Windows do przeglądania i odsiewu zdjęć po sesji (stworzony pod fotografię zwierząt).

**made by PanRodla – wszelkie prawa zastrzeżone**

## Co robi
- Otwiera folder z sesją: **CR3** (Canon R6 itd.), CR2, NEF, ARW, DNG, JPG. Wczytuje podgląd JPEG wbudowany w plik RAW, więc działa szybko bez wywoływania RAW-ów.
- **Wykrywa serie** po czasie wykonania zdjęcia. Domyślnie zdjęcia zrobione w odstępie do 1,5 s to jedna seria.
- **Ocenia ostrość** najostrzejszego fragmentu kadru, więc rozmyte tło (bokeh) nie zaniża wyniku.
- Oznacza **„?” sugerowane do odrzutu**: zdjęcia nieostre i wyraźnie słabsze od najlepszego w serii. **TOP** to najostrzejsza klatka w serii.
- Oceny 1–5★, odrzut i kolorowe etykiety zapisuje do plików **.xmp**, które czytają Lightroom, Capture One i Bridge.
- **Kopiuj wybrane** kopiuje zdjęcia z oceną ≥ N do innego folderu. **Przenieś odrzucone** przenosi je do podfolderu `_ODRZUCONE`; nic nie jest kasowane.
- **Lista nazw** kopiuje do schowka nazwy plików wybranych zdjęć.
- Postęp zapisuje się sam w pliku `.fotobudka.json` w folderze sesji. Analiza liczy się raz, a po ponownym otwarciu folderu wszystko jest od razu gotowe.

## Skróty
| Klawisz | Akcja |
|---|---|
| ← → / kółko myszy | poprzednie / następne |
| 1–5, 0 | ocena / usuń ocenę |
| X | odrzuć / przywróć |
| 6 7 8 9 | etykieta czerwona / żółta / zielona / niebieska |
| Z / Spacja / klik | zoom 100% (zostaje przy przechodzeniu, wygodne do porównania oczu w serii) |
| G / Shift+G | następna / poprzednia seria |
| B | najlepsze w serii |
| Ctrl+Z | cofnij |
| Ctrl+S | zapisz XMP |
| F11 | pełny ekran |

## Lightroom
- Najpierw zrób selekcję, potem importuj folder do Lightrooma. Oceny i etykiety wczytają się same z plików .xmp.
- Jeśli folder jest już zaimportowany, zaznacz zdjęcia i wybierz **Metadane → Odczytaj metadane z plików**.
- Odrzucone dostają ocenę −1. Lightroom może jej nie pokazać jako flagi „odrzucone”, dlatego prościej przenieść je przyciskiem do `_ODRZUCONE`.
- Pliki .xmp powstają tylko dla RAW-ów, bo Lightroom nie czyta plików .xmp dla JPG.

## Budowanie
EXE buduje się automatycznie przy każdej zmianie i trafia do **Releases**: https://github.com/barejmariusz-blip/Fotobudka/releases/latest

Uruchomienie ze źródeł: `pip install -r requirements.txt` → `python main.py`
