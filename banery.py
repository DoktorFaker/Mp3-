#!/usr/bin/env python3
"""Banery podążające za twarzami (MediaPipe Face Detection + OpenCV + ffmpeg).

Otwiera film.mp4, wykrywa dwie twarze i od 2. do 8. sekundy rysuje 50 px nad
każdą z nich zaokrąglony czarny baner z białym napisem:
  - nad mężczyzną: "Google"
  - nad kobietą:   "Agencies"
Dźwięk jest kopiowany z oryginału bez rekompresji. Wynik: wynik.mp4.

MediaPipe Face Detection nie rozpoznaje płci, więc osoby są rozróżniane po
położeniu w kadrze: domyślnie mężczyzna to twarz po LEWEJ stronie. Jeśli jest
odwrotnie, uruchom:  python banery.py --mezczyzna prawa

Wymagania:  pip install mediapipe opencv-python   oraz ffmpeg w PATH.
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_PLIK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "blaze_face_short_range.tflite"
)

ODSTEP = 50          # px między banerem a górną krawędzią twarzy
WYGLADZANIE = 0.25   # 0..1; mniej = płynniej (ale z większym opóźnieniem)
FONT = cv2.FONT_HERSHEY_DUPLEX


def pobierz_model():
    if not os.path.exists(MODEL_PLIK):
        print("Pobieram model detektora twarzy...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PLIK)
    return MODEL_PLIK


def zaokraglony_prostokat(img, x1, y1, x2, y2, r, kolor):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), kolor, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), kolor, -1, cv2.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, kolor, -1, cv2.LINE_AA)


def rysuj_baner(img, tekst, srodek_x, gora_twarzy, skala):
    """Baner wyśrodkowany nad twarzą; jego dolna krawędź jest ODSTEP px nad nią."""
    h, w = img.shape[:2]
    grubosc = max(1, round(skala * 1.5))
    (tw, th), _ = cv2.getTextSize(tekst, FONT, skala, grubosc)
    pad_x, pad_y = round(th * 0.9), round(th * 0.6)
    bw, bh = tw + 2 * pad_x, th + 2 * pad_y

    x1 = round(srodek_x - bw / 2)
    y2 = round(gora_twarzy - ODSTEP)
    # Baner nie może wyjść poza kadr.
    x1 = max(0, min(x1, w - bw))
    y2 = max(bh, min(y2, h))
    y1, x2 = y2 - bh, x1 + bw

    zaokraglony_prostokat(img, x1, y1, x2, y2, bh // 2, (0, 0, 0))
    cv2.putText(img, tekst, (x1 + pad_x, y2 - pad_y), FONT, skala,
                (255, 255, 255), grubosc, cv2.LINE_AA)


class Sledzenie:
    """Przypisuje wykryte twarze do dwóch osób i wygładza ich pozycje (EMA)."""

    def __init__(self, etykiety):
        self.etykiety = etykiety  # [napis dla lewej twarzy, napis dla prawej]
        self.poz = None           # [[cx, gora, szer], [cx, gora, szer]]

    def aktualizuj(self, twarze):
        """twarze: lista (cx, gora, szer) w pikselach."""
        if self.poz is None:
            if len(twarze) < 2:
                return
            # Start: dwie największe twarze, uporządkowane od lewej do prawej.
            dwie = sorted(sorted(twarze, key=lambda t: -t[2])[:2])
            self.poz = [list(t) for t in dwie]
            return

        # Dopasowanie do najbliższej dotychczasowej pozycji (zachłannie).
        pary = sorted(
            (abs(t[0] - p[0]) + abs(t[1] - p[1]), i, j)
            for i, p in enumerate(self.poz) for j, t in enumerate(twarze)
        )
        uzyte_i, uzyte_j = set(), set()
        for odl, i, j in pary:
            if i in uzyte_i or j in uzyte_j:
                continue
            if odl > 2.5 * self.poz[i][2]:   # za daleko – to nie ta osoba
                continue
            uzyte_i.add(i)
            uzyte_j.add(j)
            a = WYGLADZANIE
            self.poz[i] = [(1 - a) * s + a * n
                           for s, n in zip(self.poz[i], twarze[j])]
        # Jeśli twarz chwilowo zniknęła, baner zostaje w ostatniej pozycji.

    def banery(self):
        if self.poz is None:
            return []
        return [(e, p[0], p[1]) for e, p in zip(self.etykiety, self.poz)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wejscie", default="film.mp4")
    ap.add_argument("--wyjscie", default="wynik.mp4")
    ap.add_argument("--start", type=float, default=2.0, help="początek [s]")
    ap.add_argument("--koniec", type=float, default=8.0, help="koniec [s]")
    ap.add_argument("--mezczyzna", choices=["lewa", "prawa"], default="lewa",
                    help="po której stronie kadru jest mężczyzna")
    args = ap.parse_args()

    if not os.path.exists(args.wejscie):
        sys.exit(f"Nie znaleziono pliku: {args.wejscie}")
    if shutil.which("ffmpeg") is None:
        sys.exit("Brak ffmpeg w PATH.")

    cap = cv2.VideoCapture(args.wejscie)
    if not cap.isOpened():
        sys.exit(f"Nie można otworzyć: {args.wejscie}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, klatka = cap.read()
    if not ok:
        sys.exit("Film nie zawiera klatek.")
    h, w = klatka.shape[:2]
    skala = max(0.6, h / 720)  # rozmiar napisu proporcjonalny do rozdzielczości

    etykiety = ["Google", "Agencies"]  # [lewa, prawa]
    if args.mezczyzna == "prawa":
        etykiety.reverse()
    sledzenie = Sledzenie(etykiety)

    detektor = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pobierz_model()),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
    )

    # Klatki idą potokiem prosto do ffmpeg, który dokleja oryginalny dźwięk.
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-",
         "-i", args.wejscie,
         "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", args.wyjscie],
        stdin=subprocess.PIPE,
    )

    nr = 0
    try:
        while ok:
            t = nr / fps
            if args.start <= t <= args.koniec:
                rgb = cv2.cvtColor(klatka, cv2.COLOR_BGR2RGB)
                obraz = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                wynik = detektor.detect_for_video(obraz, round(t * 1000))
                twarze = [
                    (d.bounding_box.origin_x + d.bounding_box.width / 2,
                     d.bounding_box.origin_y,
                     d.bounding_box.width)
                    for d in wynik.detections
                ]
                sledzenie.aktualizuj(twarze)
                for tekst, cx, gora in sledzenie.banery():
                    rysuj_baner(klatka, tekst, cx, gora, skala)
            ffmpeg.stdin.write(klatka.tobytes())
            nr += 1
            ok, klatka = cap.read()
    finally:
        cap.release()
        detektor.close()
        ffmpeg.stdin.close()
        kod = ffmpeg.wait()

    if kod != 0:
        sys.exit("ffmpeg zakończył się błędem.")
    if sledzenie.poz is None:
        print("UWAGA: między --start a --koniec nie wykryto dwóch twarzy "
              "jednocześnie – banery nie zostały narysowane.")
    print(f"Zapisano {args.wyjscie} ({nr} klatek, {fps:.2f} fps).")


if __name__ == "__main__":
    main()
