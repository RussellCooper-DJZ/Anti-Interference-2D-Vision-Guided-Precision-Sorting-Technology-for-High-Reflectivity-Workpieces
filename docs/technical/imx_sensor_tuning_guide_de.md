# Sony IMX Sensor Register-Tuning-Leitfaden zur Blendunterdrückung

## 1. Einleitung

In der industriellen Bildverarbeitung reicht die ausschließliche Verwendung von Backend-Bildverarbeitungsalgorithmen oft nicht aus, um gesättigte Pixel bei hochreflektierenden Metallwerkstoffen (wie Edelstahl, galvanisierten Teilen, Aluminiumlegierungen) vollständig wiederherzustellen. Dieser Leitfaden beschreibt, wie durch die direkte Konfiguration der Register von Sony IMX-Sensoren (z. B. IMX290, IMX335, IMX415) spiegelnde Reflexionen direkt an der **optischen Erfassungsquelle** unterdrückt werden können.

## 2. Kernregister und Konfigurationsstrategien

### 2.1 Aktivierung des DOL-HDR (Digital Overlap HDR) Modus
Die DOL-HDR-Technologie von Sony ermöglicht es dem Sensor, nacheinander mehrere Belichtungen (lang, mittel, kurz) innerhalb einer einzigen Bildperiode aufzunehmen und verschachtelte Datenströme über die CSI-2-Schnittstelle auszugeben.
*   **Konfigurationsmethode**: Schreiben Sie `0x01` in Register `0x300C` (für IMX335), um den 2-Frame-DOL-HDR-Modus zu aktivieren.
*   **Zweck**: Die kurze Belichtung erfasst Kantendetails in Glanzlichtbereichen, während die lange Belichtung Schattenbereiche erfasst.

### 2.2 Minimierungsstrategie für analoge Verstärkung (Analog Gain)
Das blinde Erhöhen der Verstärkung in hochreflektierenden Umgebungen führt zu einer schnellen Sättigung der Full-Well-Kapazität.
*   **Konfigurationsmethode**: Setzen Sie das Register für die analoge Verstärkung (z. B. `0x30E8`) auf `0x00` (0 dB).
*   **Zweck**: Maximierung des Dynamikbereichs des Sensors.

### 2.3 Dynamische Anpassung des Schwarzwerts (Black Level Offset)
*   **Konfigurationsmethode**: Feineinstellung von Register `0x3015`.
*   **Zweck**: Unter starker Umgebungsbeleuchtung filtert das Anheben des Schwarzwerts schwaches Rauschen heraus.

## 3. Materialadaptive Tuning-Empfehlungen

| Werkstoff | Empfohlene Verstärkung | HDR-Modus | Spezielle Registeranpassungen |
| :--- | :--- | :--- | :--- |
| **Edelstahl** | 0 dB | Aktiviert (DOL-HDR) | Standard-Schwarzwert (10-15) |
| **Aluminiumlegierung** | 2-3 dB | Aktiviert | Belichtungszeit leicht erhöhen |
| **Galvanisierte Teile** | 0 dB | Erzwungen aktiviert | Extrem kurze Belichtungssequenz (100us) |

## 4. Fazit

Durch die Kombination von Low-Level-Register-Tuning und dem integrierten Simple ISP des RA8P1 wird ein geschlossener Regelkreis aus hardwareseitiger Blendunterdrückung und Software-KI geschaffen.

---
**Urheberrecht Manus AI Industrial Vision Lab**
