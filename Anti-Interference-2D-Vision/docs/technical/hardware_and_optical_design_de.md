# Hardware- und Optikdesign für die störungsfreie 2D-Vision-geführte Sortierung hochreflektierender Werkstücke

## 1. Lösungsübersicht

Diese Lösung adressiert die präzisen Sortieranforderungen hochreflektierender Metallwerkstücke (z. B. Edelstahl, Aluminiumlegierung, galvanisierte Teile) in industriellen Umgebungen. Sie nutzt die integrierte **Simple ISP**-Funktionalität des Renesas **RA8P1** voll aus, um eine leistungsstarke Bilderkennung mit kostengünstigen Hardwarekombinationen zu erreichen. Die Kernidee besteht darin, Bayer-RAW-Daten von kostengünstigen CMOS-Sensoren mit dem Simple ISP zu verarbeiten, kombiniert mit Mehrfachbelichtungs-HDR-Technologie und polarisiertem Optikdesign, um Störungen durch spiegelnde Reflexionen zu eliminieren.

## 2. Hardware-Auswahlliste

| Komponenten-Kategorie | Empfohlenes Modell/Spezifikation | Auswahlgrund |
| :--- | :--- | :--- |
| **Kern-Controller** | **Renesas RA8P1** (Cortex-M85) | Integrierter **Simple ISP**, unterstützt RAW8/10/12-Eingabe; Helium-Beschleunigungsbefehlssatz steigert die KI-Inferenzgeschwindigkeit; geringer Stromverbrauch, hohe Integration. |
| **Bildsensor** | **OmniVision OV5640** oder **Sony IMX-Serie** (RAW-Ausgabe) | Unterstützt Bayer-RGB-RAW-Ausgabe, passt perfekt zum Simple ISP; 5MP-Auflösung erfüllt Präzisionsanforderungen; extrem niedrige Kosten. |
| **Objektiv** | 12mm/16mm industrielles Festbrennweitenobjektiv (geringe Verzeichnung) | Ausgewählt basierend auf dem Arbeitsabstand, um sicherzustellen, dass das Werkstück genügend Pixel im Sichtfeld einnimmt; geringe Verzeichnung begünstigt die Subpixel-Positionierung. |
| **Polarisator** | Linearer Polarisator (vor dem Objektiv montiert) | Arbeitet mit polarisierten Lichtquellen zusammen, um über 90 % des spiegelnden Reflexionslichts durch das orthogonale Polarisationsprinzip zu eliminieren. |
| **Lichtquelle** | **Polarisiertes Balkenlicht** oder **Polarisiertes Ringlicht** | 500-1000 Lux Helligkeit; integrierte Polarisationsfolie reduziert Umgebungslichtstörungen und unterdrückt Oberflächenblendung auf dem Werkstück. |
| **Roboterschnittstelle** | UART / CAN / Ethernet | Die umfangreichen Schnittstellen des RA8P1 können direkt mit gängigen Industrierobotern (z. B. KUKA, FANUC) kommunizieren. |

## 3. Optisches Design

### 3.1 Prinzip der polarisierten Bildgebung
Spiegelndes Reflexionslicht von der Oberfläche hochreflektierender Werkstücke weist starke Polarisationseigenschaften auf. Durch die Installation eines Polarisators vor der Lichtquelle und eines weiteren orthogonalen (um 90° gedrehten) Polarisators vor dem Objektiv kann starkes direkt reflektiertes Licht effektiv herausgefiltert werden, sodass nur diffuses Reflexionslicht übrig bleibt, das Texturinformationen der Werkstückoberfläche trägt.

### 3.2 Mehrfachbelichtungs-HDR-Strategie
Nutzen Sie den RA8P1 zur Steuerung des Sensors für eine schnelle Dreifachbelichtung (Unterbelichtung, Normalbelichtung, Überbelichtung):
1.  **Unterbelichtung**: Erfasst Kantendetails in hochreflektierenden Bereichen und verhindert Merkmalsverlust durch Überbelichtung.
2.  **Normalbelichtung**: Liefert klare Bilder des Hintergrunds und normaler Bereiche.
3.  **Überbelichtung**: Extrahiert Merkmale aus dunklen Bereichen (z. B. Bereiche mit Ölflecken oder Fingerabdruckstörungen).
Nach der Vorverarbeitung durch den Simple ISP führen Softwarealgorithmen eine Fusion durch, um Bilder mit hohem Dynamikbereich zu erzeugen.

## 4. Simple ISP-Parameter-Tuning-Strategie

Für hochreflektierende Szenen wird empfohlen, die folgenden Parameter über die V4L2-Schnittstelle im RA8P1 einzustellen:

| Parameter-ID | Empfohlene Einstellung | Zweck |
| :--- | :--- | :--- |
| `V4L2_CID_RZ_ISP_GAMMA` | 150 - 200 (1.5 - 2.0) | Verbessert den Kontrast in dunklen Bereichen und unterdrückt Highlights. |
| `V4L2_CID_RZ_ISP_2DNR` | 70 - 100 | Eliminiert planares Rauschen des Sensors bei hoher Verstärkung. |
| `V4L2_CID_RZ_ISP_EMP` | 2 (Normal) | Stärkt Werkstückkanten und verbessert die Genauigkeit der Subpixel-Positionierung. |
| `V4L2_CID_RZ_ISP_BL` | 10 - 20 | Erhöht den Schwarzwert angemessen, um schwache Hintergrundreflexionen herauszufiltern. |

## 5. Kostenvorteilsanalyse
*   **Kein externer ISP erforderlich**: Spart ca. $5-$10 an Hardwarekosten.
*   **Kostengünstige Sensoren**: Unterstützt gängige Bayer-RAW-Sensoren auf dem Markt und reduziert die Sensorkosten um über 30 % im Vergleich zu Sensoren mit integrierten ISPs.
*   **Hohe Integration**: Der RA8P1-Einzelchip erledigt Bilderfassung, ISP-Verarbeitung, KI-Inferenz und Bewegungssteuerung und vereinfacht so das Schaltungsdesign.

## 6. Erweitertes PBR-Beleuchtungssystem (Aktualisierung vom 2026-04-23)

Das Hardware-Design ist nun mit dem PBRLightingSystem für verbesserte Spiegelreflexionssimulation integriert:

### 6.1 Hardware-Anforderungen für PBR-Modus

| Komponente | Spezifikation | Zweck |
|-----------|---------------|-------|
| **Lichtquelle** | Multi-direktionales LED-Array mit einstellbarer Intensität | Unterstützt `pbr_sun`-Modus mit gerichteter Sonnenlichtsimmulation |
| **Kamera** | Globaler Verschluss CMOS, ≥3 Belichtungsreihen | Multi-Exposure-HDR-Aufnahme für PBR-Trainingsdaten |
| **Recheneinheit** | NVIDIA GPU für Training, ONNX Runtime für Edge | TensorRT FP16 ~10ms Inferenz am Edge |

### 6.2 PBR-Physikalische Parameter

| Parameter | Hardware-Implikation | Wertebereich |
|-----------|---------------------|-------------|
| `roughness` | Oberflächenfinish-Auswahl | 0,01 (spiegelnd) ~ 1,0 (diffus) |
| `metallic` | Materialtyp | 0,0 (Nichtmetall) ~ 1,0 (reines Metall) |
| `specular_scale` | Lichtquellenhelligkeit | 0,0 ~ 2,0 |

### 6.3 Softwaredefinierte Beleuchtungsmodi

Das PBR-System unterstützt mehrere Beleuchtungsmodi, die per Software konfigurierbar sind:

| Modus | Beschreibung | Hardware-Anforderung |
|------|-------------|---------------------|
| `pbr` | Standard Blinn-Phong BRDF | Basis-LED-Array |
| `pbr_sun` | PBR + gerichtetes Sonnenlicht | Hochintensives gerichtetes LED |
| `pbr_mixed` | PBR + Wasserreflexion | Diffus + reflektierende Oberflächeneinrichtung |
