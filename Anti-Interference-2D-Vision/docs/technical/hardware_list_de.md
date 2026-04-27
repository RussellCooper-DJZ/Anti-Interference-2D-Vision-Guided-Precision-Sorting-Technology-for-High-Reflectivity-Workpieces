# Hardwareliste

Die folgenden Hardwarekomponenten sind für die störungsfreie 2D-Vision-geführte Präzisionssortierlösung für hochreflektierende Werkstücke erforderlich:

## 1. Zentrale Verarbeitungseinheit

*   **Renesas RA8P1 Mikrocontroller-Entwicklungsplatine**: Ausgestattet mit einem Arm Cortex-M85-Kern, der für die Ausführung eingebetteter Bildverarbeitungsalgorithmen und Steuerungslogik verwendet wird. Es wird empfohlen, ein offizielles Evaluierungsboard oder ein kundenspezifisches Entwicklungsboard zu verwenden.

## 2. Bilderfassungssystem

*   **Industriekamera**:
    *   **Typ**: Hochauflösende (z. B. 5 Megapixel oder höher), hochfrequente (z. B. 60 fps oder höher) Global-Shutter-CMOS-Industriekamera.
    *   **Schnittstelle**: CSI-2-Schnittstelle, kompatibel mit RA8P1, oder andere Hochgeschwindigkeitsschnittstellen (z. B. USB3.0, erfordert einen Bridge-Chip).
    *   **Funktionalität**: Ausgestattet mit HDR-Funktionen oder Unterstützung von Mehrfachbelichtungsmodi.
*   **Industrieobjektiv**:
    *   **Typ**: Verzerrungsarmes, hochauflösendes Objektiv mit fester Brennweite.
    *   **Brennweite**: Wählen Sie eine geeignete Brennweite basierend auf dem Arbeitsabstand und dem Sichtfeld.
*   **Polarisationsfilter (Optional)**: Wenn Hardware-Polarisationsbildgebung verwendet wird, muss ein Polarisationsfilter vor dem Objektiv installiert werden.

## 3. Beleuchtungssystem

*   **Diffuse Lichtquelle**:
    *   **Typ**: Ulbricht-Kugel-Lichtquelle, Dome-Licht oder Ringdiffusor-Lichtquelle.
    *   **Leistung**: Wählen Sie die geeignete Leistung basierend auf der Werkstückreflexion und den Umgebungslichtbedingungen, um eine gleichmäßige Ausleuchtung zu gewährleisten.
*   **Polarisationslichtquelle (Optional)**: Wenn Hardware-Polarisationsbildgebung verwendet wird, ist eine Polarisationslichtquelle erforderlich.
*   **Lichtquellensteuerung**: Wird zur präzisen Steuerung der Helligkeit der Lichtquelle, des Blitzmodus usw. verwendet.

## 4. Robotersystem

*   **Industrieroboter**: Sechsachsiger oder mehrachsiger Industrieroboter mit hoher Wiederholgenauigkeit.
*   **Robotersteuerung**: Kommuniziert mit RA8P1, empfängt visuelle Führungsbefehle und führt Greifaktionen aus.
*   **Endeffektor**: Wählen Sie einen geeigneten Greifer oder Saugnapf basierend auf der Werkstückform und dem Material.

## 5. Kalibrierungswerkzeuge

*   **Hochpräzises Kalibrierungsboard**: Wird für die Kameraintrinsikkalibrierung und die Hand-Auge-Kalibrierung verwendet, z. B. ein Schachbrett, ein ChArUco-Board oder ein Punktmuster-Kalibrierungsboard.

## 6. Zusatzausrüstung

*   **Stromversorgung**: Stabile Stromversorgung für alle Hardwarekomponenten.
*   **Industrie-PC**: Wird für die Algorithmusentwicklung, das Modelltraining, die Hand-Auge-Kalibrierung und die Kommunikationsfehlersuche mit RA8P1 verwendet.
*   **Monitor**: Zur Fehlersuche und Überwachung.
*   **Verbindungskabel**: Einschließlich Kameradatenkabel, Stromkabel, Kommunikationskabel usw.

## 7. Softwareumgebung

*   **Entwicklungstoolchain**: Renesas e2 studio IDE, Arm GNU Toolchain.
*   **FSP (Flexible Software Package)**: Von Renesas bereitgestelltes Softwareentwicklungspaket, einschließlich Treibern und Middleware.
*   **TensorFlow Lite for Microcontrollers (TFLM)**: Wird für die Bereitstellung von Deep-Learning-Modellen auf RA8P1 verwendet.
*   **OpenCV (PC-Version)**: Wird für die Entwicklung und Verifizierung von Bildverarbeitungsalgorithmen auf dem PC verwendet.
*   **Python/PyTorch/TensorFlow (PC-Version)**: Wird für das Training und die Konvertierung von Deep-Learning-Modellen verwendet.
*   **Roboterprogrammierumgebung**: Vom Roboterhersteller bereitgestellte Programmiersoftware.

## 8. Erweiterte Komponenten (Aktualisierung vom 2026-04-23)

### 8.1 PBR-Beleuchtungssystem-Komponenten

Für das erweiterte PBR (Physically Based Rendering) Beleuchtungssimulation:

| Komponente | Spezifikation | Zweck |
|-----------|---------------|-------|
| **Multi-direktionales LED-Array** | ≥4 Richtungen, 0-100% Intensitätssteuerung | Simuliert komplexe industrielle Beleuchtung |
| **Hochintensives gerichtetes LED** | ≥10000 Lux äquivalent | `pbr_sun`-Modus Sonnenlichtsimulation |
| **Reflektierende Oberflächeneinrichtung** | Matt-schwarz + Spiegeloberflächen | `pbr_mixed` Wasserreflexionsmodus |

### 8.2 Inferenzrechenkomponenten

| Komponente | Spezifikation | Anwendungsfall |
|-----------|---------------|----------------|
| **NVIDIA GPU (Training)** | CUDA-fähige GPU, ≥8GB VRAM | PBR-Datengenerierung, Modelltraining |
| **ONNX Runtime (Edge)** | Helium-optimierter Build für RA8P1 | Eingebettete Inferenz, ~40ms Latenz |
| **TensorRT (Optional)** | NVIDIA GPU erforderlich | PC/Edge-Inferenz, ~10ms Latenz |

### 8.3 ABB-Roboter-Schnittstellenkomponenten

Für die AbbRobotEGM UDP EGM-Protokollschnittstelle:

| Komponente | Spezifikation | Zweck |
|-----------|---------------|-------|
| **ABB-Robotersteuerung** | IRC5 oder neuer mit EGM-Option | Echte Robotersteuerung |
| **Netzwerk-Switch** | 1000Mbps, niedrige Latenz | UDP EGM-Kommunikation (Port 6510) |
| **EGM-Konfiguration** | Protobuf oder binäres Fallback | RobotStudio-Integration |
