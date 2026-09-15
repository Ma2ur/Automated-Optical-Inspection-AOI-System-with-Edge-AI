import cv2
import serial
import time
import numpy as np
from anomalib.deploy import TorchInferencer
import os

os.environ["TRUST_REMOTE_CODE"] = "1"

# --- KONFIGURACJA SPRZĘTU ---
PORT_COM = 'COM5'
BAUD_RATE = 9600
ID_KAMERY = 1
ILOSC_ZDJEC = 8

# --- KONFIGURACJA AI ---
SCIEZKA_WAG = "gotowy_model/weights/torch/model.pt"

print("Inicjalizacja systemu AI. Ładowanie modelu z wbudowaną metadaną...")
try:
    inferencer = TorchInferencer(path=SCIEZKA_WAG, device="auto")
except Exception as e:
    print(f"Błąd ładowania modelu AI: {e}\n")
    exit()

print("Inicjalizacja kamery i sprzętu...")
cap = cv2.VideoCapture(ID_KAMERY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"Łączenie z Arduino na porcie {PORT_COM}...")
try:
    ser = serial.Serial(PORT_COM, BAUD_RATE, timeout=2)
except Exception as e:
    print(f"Błąd portu COM: {e}. Upewnij się, że Serial Monitor w IDE jest zamknięty.")
    exit()

time.sleep(2)

print("Oczekiwanie na sprzet...")
while True:
    if ser.in_waiting > 0:
        if ser.readline().decode('utf-8').strip() == "SYSTEM_READY":
            print("\n===============================")
            print(" SYSTEM W PEŁNI GOTOWY")
            print("===============================")
            print("[SPACJA] -> Rozpocznij cykl skanowania 360°")
            print("[Q]      -> Wyjście z programu\n")
            break

# --- GŁÓWNA PĘTLA Z ZABEZPIECZENIEM SPRZĘTOWYM ---
try:
    while True:
        ret, frame_live = cap.read()
        if ret:
            cv2.imshow("Stacja AOI - Podglad Na Zywo", frame_live)
            
        klawisz = cv2.waitKey(1) & 0xFF
        
        if klawisz == ord('q'):
            print("Zamykanie systemu...")
            break
            
        elif klawisz == 32:
            print("\n>>> START CYKLU INSPEKCJI <<<")
            
            cykl_odrzucony = False
            najgorsza_heatmapa = None
            najwyzszy_wynik_anomalii = 0.0
            najgorsza_klatka = None

            for i in range(ILOSC_ZDJEC):
                ser.write(b'N')
                while True:
                    ret, frame_live = cap.read()
                    if ret:
                        cv2.imshow("Stacja AOI - Podglad Na Zywo", frame_live)
                    
                    cv2.waitKey(1) 
                    
                    if ser.in_waiting > 0:
                        odpowiedz = ser.readline().decode('utf-8').strip()
                        if odpowiedz == "PICTURE_READY":
                            break
            
                # Stabilizacja obrazu po zatrzymaniu stołu
                cap.read() 
                ret, frame = cap.read()
                
                if ret:
                    cv2.imshow("Stacja AOI - Podglad Na Zywo", frame)
                    cv2.waitKey(1)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    wynik_ai = inferencer.predict(image=frame_rgb)
                    
                    czy_wada = wynik_ai.pred_label.item()
                    wynik_pewnosci = wynik_ai.pred_score.item()
                    
                    status_tekst = "WADA!" if czy_wada else "OK"
                    print(f"  Kąt {i * 45}°: {status_tekst} (Anomaly Score: {wynik_pewnosci:.3f})")
                    
                    if czy_wada:
                        cykl_odrzucony = True
                        if wynik_pewnosci > najwyzszy_wynik_anomalii:
                            najwyzszy_wynik_anomalii = wynik_pewnosci
                            najgorsza_heatmapa = wynik_ai.anomaly_map[0].squeeze().cpu().numpy()
                            najgorsza_klatka = frame.copy()
                            
            # --- WERDYKT KOŃCOWY ---
            print("\n--- PODSUMOWANIE DETALU ---")
            if cykl_odrzucony:
                heatmap_norm = cv2.normalize(najgorsza_heatmapa, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                wysokosc, szerokosc = najgorsza_klatka.shape[:2]
                heatmap_norm = cv2.resize(heatmap_norm, (szerokosc, wysokosc))

                heatmapa_kolor = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

                cv2.imshow("DEBUG: 1. Surowa Heatmapa z AI", cv2.resize(heatmapa_kolor, (1280, 720)))
                while cv2.waitKey(0) & 0xFF != 32:
                    pass 
                cv2.destroyWindow("DEBUG: 1. Surowa Heatmapa z AI")

                max_val = np.max(heatmap_norm)
                prog_dynamiczny = int(max_val * 0.85) 
                    
                _, binary_mask = cv2.threshold(heatmap_norm, prog_dynamiczny, 255, cv2.THRESH_BINARY)
                    
                cv2.imshow("DEBUG: 2. Maska Binarna", cv2.resize(binary_mask, (1280, 720)))
                while cv2.waitKey(0) & 0xFF != 32:
                    pass
                cv2.destroyWindow("DEBUG: 2. Maska Binarna")

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                obraz_wynikowy = najgorsza_klatka.copy()
                    
                licznik_wad = 0
                for cnt in contours:
                    pole_konturu = cv2.contourArea(cnt)
                    limit_maksymalny = (szerokosc * wysokosc) * 0.30
                        
                    if 20 < pole_konturu < limit_maksymalny:
                        licznik_wad += 1
                        cv2.drawContours(obraz_wynikowy, [cnt], -1, (0, 0, 255), 3)
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(obraz_wynikowy, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(obraz_wynikowy, f"WADA {licznik_wad}", (x, y - 8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                obraz_wynikowy_podglad = cv2.resize(obraz_wynikowy, (1280, 720))
                cv2.imshow("WERDYKT: WADA! (Spacja by kontynuowac)", obraz_wynikowy_podglad)

                while cv2.waitKey(0) & 0xFF != 32:
                    pass 
                cv2.destroyWindow("WERDYKT: WADA! (Spacja by kontynuowac)")
                    
            else:
                print("WYNIK: DETAL IDEALNY (OK).")
                img_ok = np.zeros((300, 500, 3), dtype=np.uint8)
                img_ok[:] = (0, 200, 0)
                cv2.putText(img_ok, "DETAL OK", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                
                cv2.imshow("WERDYKT: OK (Spacja by kontynuowac)", img_ok)
                while cv2.waitKey(0) & 0xFF != 32:
                    pass
                cv2.destroyWindow("WERDYKT: OK (Spacja by kontynuowac)")

except KeyboardInterrupt:
    print("\n[!] Program przerwany przez użytkownika.")
except Exception as e:
    print(f"\n[!] Wystąpił niespodziewany błąd: {e}")
finally:
    print("\nKoniec pracy. Zwalnianie zasobów sprzętowych...")
    cap.release()
    if 'ser' in locals() and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    print("Zakończono bezpiecznie.")