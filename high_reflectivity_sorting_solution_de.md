# Störungsfreie 2D-Vision-geführte Präzisionssortiertechnologie für hochreflektierende Werkstücke

## 1. Einleitung

Diese technische Lösung zielt darauf ab, die Kernherausforderungen der "unklaren Bildgebung, ungenauen Positionierung und instabilen Greifens" bei der 2D-Vision-geführten Sortierung von hochreflektierenden Metallwerkstücken zu bewältigen. Durch die Integration fortschrittlicher Bildverarbeitungs-, Deep-Learning- und Robotersteuerungstechnologien wird eine hochpräzise, hochstabile Erkennung und Positionierung von hochreflektierenden Werkstücken erreicht, um die strengen Produktionsanforderungen von Industriestandorten zu erfüllen. Derzeit ist die visuelle Inspektion von hochreflektierenden Oberflächen eine große Herausforderung in der Bildverarbeitung, und Forscher weltweit erforschen aktiv die Kombination von multimodaler Bildgebung, Deep Learning und anderen Spitzentechnologien, um die Einschränkungen traditioneller Methoden zu überwinden [10, 11].

## 2. Überblick über die wichtigsten technischen Indikatoren

Gemäß den Benutzeranforderungen muss diese Lösung die folgenden wichtigsten technischen Indikatoren erfüllen:

| Indikatorkategorie | Spezifische Anforderungen |
| :----------------- | :------------------------ |
| **Erkennungsgenauigkeit & Stabilität** | Erfolgsrate der Werkstückerkennung ≥ 99,5% (unter typischen industriellen Beleuchtungs- und starken Störlichtbedingungen); Fehlerrate ≤ 0,1% |
| **Positionierungsgenauigkeit & Geschwindigkeit** | 2D-Planar-Positionierungsgenauigkeit: Fehler ≤ ±0,2 mm (oder ≤ 0,5 Pixel); Winkelpositionierungsgenauigkeit: Fehler ≤ ±0,5°; Bildverarbeitungszyklus: ≤ 300 ms (von der Bilderfassung bis zur Koordinatenausgabe) |
| **Algorithmusrobustheit** | Anpassung an ein gewisses Maß an Ölverschmutzungen und Fingerabdruckstörungen auf der Werkstückoberfläche; Erkennungsrate ohne signifikante Verschlechterung bei ±20% Schwankungen der Lichtquellenhelligkeit; Unterstützung der allgemeinen Erkennung von mindestens 3 verschiedenen Materialien (z. B. Edelstahl, Aluminiumlegierung, galvanisierte Teile) von hochreflektierenden Werkstücken |

## 3. Optisches Lösungsdesign

Für die Bildgebungsherausforderungen von hochreflektierenden Werkstücken ist das Design der optischen Lösung entscheidend. Ihr Ziel ist es, spiegelnde Reflexionen und Umgebungslichtstörungen so weit wie möglich zu unterdrücken und gleichzeitig die wahren Merkmale des Werkstücks hervorzuheben. Nationale und internationale Forschungsergebnisse zeigen, dass ein vernünftiges optisches Design die Grundlage für den Erfolg der visuellen Inspektion von hochreflektierenden Oberflächen ist [10, 11].

### 3.1 Auswahl und Anordnung der Lichtquelle

Angesichts der hohen Reflexionseigenschaften neigen herkömmliche direkte Lichtquellen dazu, lokale Überbelichtung zu verursachen. Diese Lösung wird die folgenden Strategien anwenden:

*   **Diffuse Lichtquelle**: Bevorzugen Sie die Verwendung von Ulbricht-Kugel-Lichtquellen, Dome-Lichtquellen oder Ringdiffusor-Lichtquellen, um eine gleichmäßige, schattenfreie Beleuchtung zu gewährleisten und lokale Glanzlichter, die durch spiegelnde Reflexionen verursacht werden, effektiv zu reduzieren. Für Materialien wie Edelstahl und Aluminiumlegierungen kann diffuses Licht ihre Konturen besser hervorheben.
*   **Polarisierte Beleuchtung**: Für galvanisierte Teile mit besonders starken spiegelnden Reflexionen sollten polarisierte Lichtquellen mit polarisierten Kameras oder Polarisationsfiltern in Betracht gezogen werden. Durch Anpassen der Polarisationsrichtung können spiegelnde Reflexionen effektiv herausgefiltert werden, wodurch die wahre Textur und die Kanten der Werkstückoberfläche sichtbar werden. Studien zeigen, dass die polarisierte Bildgebung erhebliche Vorteile bei der Unterdrückung spiegelnder Reflexionen und der Verbesserung von Oberflächendetails bietet [12, 13]. Dies wird ein wichtiger Schwerpunkt für die softwareseitige Simulation und Optimierung in dieser Lösung sein.
*   **Kombination aus mehreren Winkeln/Lichtquellen**: Bei einigen komplex geformten Werkstücken kann eine einzelne Lichtquelle keine vollständige Abdeckung bieten. Eine Kombination von Lichtquellen mit geringer Leistung aus mehreren Winkeln kann verwendet werden, um umfassendere Informationen durch Bildfusionstechnologie zu erhalten.

### 3.2 Kamera- und Objektivauswahl

*   **Kamera**: Wählen Sie eine Industriekamera mit hoher Auflösung (z. B. 5 Megapixel oder mehr) und hoher Bildrate, um die Anforderungen an die Positionierungsgenauigkeit und den Bildverarbeitungszyklus zu erfüllen. Darüber hinaus muss die Kamera ein gutes Signal-Rausch-Verhältnis und eine Wide Dynamic Range (HDR)-Funktion aufweisen, um mit Beleuchtungsschwankungen umgehen zu können. Die High Dynamic Range-Bildgebungstechnologie bewältigt Über- und Unterbelichtungsprobleme in hochreflektierenden Szenen hervorragend [14].
*   **Objektiv**: Wählen Sie ein verzerrungsarmes, hochauflösendes Objektiv mit fester Brennweite, um die geometrische Genauigkeit des Bildes zu gewährleisten. Berechnen Sie die geeignete Brennweite und den Arbeitsabstand basierend auf der Werkstückgröße und dem Sichtfeld.

### 3.3 Bilderfassungsstrategie

*   **Mehrfachbelichtungserfassung**: Um eine High Dynamic Range (HDR)-Bildgebung zu erreichen, führt die Kamera eine Mehrfachbelichtungserfassung durch, d.h. sie nimmt mehrere Bilder mit unterschiedlichen Belichtungszeiten (unterbelichtet, normal belichtet, überbelichtet) auf. Diese Bilder werden in nachfolgenden Algorithmen fusioniert. Diese Methode wurde in der 3D-Messung und Merkmalsextraktion von hochreflektierenden Oberflächen weit verbreitet angewendet [10, 14].
*   **Polarisierte Bilderfassung (Simulation)**: Ohne Erhöhung der Hardwarekosten wird diese Lösung die Simulation des Entspiegelungseffekts der polarisierten Bildgebung durch Softwarealgorithmen untersuchen, z. B. durch Analyse der Reflexionseigenschaften in Mehrfachbelichtungsbildern, um spiegelnde Reflexionen zu unterdrücken. Während Hardware-Polarisationskameras bessere Ergebnisse liefern, kann die Softwaresimulation als kostengünstige Alternative dienen, indem Bildverarbeitungstechniken (wie die Reflexionstrennung basierend auf Farbe oder Textur) zur Unterdrückung spiegelnder Reflexionen eingesetzt werden [12].

## 4. Algorithmusprinzipien und -architektur

Die Algorithmusarchitektur dieser Lösung wird in drei Hauptmodule unterteilt: Bildvorverarbeitung, Merkmalsextraktion und Positionierung sowie Roboterführung und -steuerung. Die schnelle Entwicklung des Deep Learning bietet leistungsstarke Werkzeuge zur Lösung komplexer Bildverarbeitungsaufgaben, insbesondere bei der Merkmalsextraktion und Robustheit [15, 16].

### 4.1 Bildvorverarbeitungsmodul

#### 4.1.1 High Dynamic Range (HDR)-Bildgebung und Mehrfachbelichtungsfusion

Um lokale Überbelichtung und "Blackout"-Phänomene, die durch hohe Reflexion verursacht werden, zu beheben, wird die Mehrfachbelichtungsfusionstechnologie eingesetzt. Durch die Erfassung mehrerer Bilder mit unterschiedlichen Belichtungszeiten (unterbelichtet, normal belichtet, überbelichtet) und die Verwendung von Fusionsalgorithmen (wie Debevec [1], Mertens [2] oder Drago [3] Algorithmen) wird ein HDR-Bild mit einem weiten Dynamikbereich erzeugt. Dieses Bild kann gleichzeitig Details in Glanzlicht- und Schattenbereichen anzeigen und die wahren Merkmale der Werkstückoberfläche effektiv wiederherstellen.

*   **Prinzip**: Pixelwerte verschiedener Belichtungsbilder werden in einen einheitlichen Strahlungsraum abgebildet, und dann wird ein informationsreicheres Bild durch gewichtete Mittelwertbildung oder Gradientenbereichsfusionsmethoden erzeugt.
*   **Vorteile**: Löst effektiv die Dynamikbereichsbeschränkungen traditioneller Einzelbilder und liefert hochwertige Eingaben für die nachfolgende Merkmalsextraktion.

#### 4.1.2 Simulation des Prinzips der polarisierten Lichtbildgebung und Unterdrückung spiegelnder Reflexionen

Obwohl Hardware-Polarisationskameras kostspielig sind, wird diese Lösung die Simulation des Entspiegelungseffekts der polarisierten Lichtbildgebung durch Softwarealgorithmen untersuchen. Dies kann die Analyse der Reflexionseigenschaften in Mehrfachbelichtungsbildern oder die Verwendung von Bildverarbeitungstechniken (wie die Reflexionstrennung basierend auf Farbe oder Textur) zur Unterdrückung spiegelnder Reflexionen umfassen. Diese Methode zielt darauf ab, ähnliche Effekte ohne die Verwendung physikalischer Polarisationsfilter zu erzielen und die Sichtbarkeit von Kanten und Texturen von hochreflektierenden Werkstücken zu verbessern [12].

*   **Prinzip**: Spiegelndes Reflexionslicht hat Polarisationseigenschaften, während diffuses Reflexionslicht dies nicht hat. Durch die Analyse der Intensität des reflektierten Lichts in verschiedenen Richtungen im Bild können Versuche unternommen werden, die spiegelnde Reflexionskomponente zu trennen. Die Softwaresimulation wird versuchen, ähnliche Effekte ohne die Verwendung physikalischer Polarisationsfilter zu erzielen.
*   **Vorteile**: Verbessert die Sichtbarkeit von Kanten und Texturen von hochreflektierenden Werkstücken ohne Erhöhung der Hardwarekosten.

#### 4.1.3 Adaptive Bildverbesserung

Nach der HDR-Fusion und Reflexionsunterdrückung wird eine adaptive Bildverbesserung durchgeführt, einschließlich Kontrastspreizung, Helligkeitsanpassung und Rauschfilterung. Ziel ist es, das Lichtpunktrauschen weiter zu unterdrücken und die Gesamtbildqualität zu verbessern, ohne die Kantengenauigkeit zu beeinträchtigen.

*   **Methoden**: CLAHE (Contrast Limited Adaptive Histogram Equalization), Non-local Means-Filterung oder Guided Filter usw.

### 4.2 Robuste Merkmalsextraktion und Positionierungsalgorithmus gegen Beleuchtungsstörungen

Dieses Modul ist der Schlüssel zur Erzielung einer hochpräzisen Positionierung und wird Deep-Learning-Methoden einsetzen, um die Einschränkungen traditioneller Algorithmen unter komplexen Beleuchtungsbedingungen zu überwinden. Deep Learning hat leistungsstarke Fähigkeiten in der Computerbildgebung und der automatisierten optischen Inspektion gezeigt [15, 16].

#### 4.2.1 Deep Learning-basiertes Merkmalsextraktionsnetzwerk

Faltungsneuronale Netze (CNNs) oder Transformer-basierte Netzwerke werden zur Merkmalsextraktion von Werkstücken verwendet. Dieses Netzwerk lernt robuste Merkmale des Werkstücks direkt aus vorverarbeiteten Bildern, anstatt sich auf traditionelle gradientenbasierte Methoden zu verlassen. Studien zeigen, dass Deep-Learning-Modelle komplexe Beleuchtungs- und Oberflächenvariationen effektiv verarbeiten können [15, 16].

*   **Netzwerkarchitektur**: U-Net [6], Mask R-CNN [7] oder YOLO [8] Objekterkennungs- und Segmentierungsnetzwerke können in Betracht gezogen und an die tatsächlichen Bedürfnisse angepasst werden. Für die Konturextraktion sind semantische Segmentierungsnetzwerke (wie DeepLabV3+) möglicherweise besser geeignet.
*   **Trainingsdaten**: Erfordert eine große Anzahl annotierter Bilder mit unterschiedlicher Beleuchtung, Materialien und Oberflächendefekten (Ölflecken, Fingerabdrücke) für das Training. Datenaugmentierungstechniken (wie zufällige Helligkeit, Kontrast, Rauschen, Rotation, Skalierung) werden verwendet, um die Generalisierungsfähigkeit des Modells zu verbessern.
*   **Robustheit**: Durch Deep-Learning-Modelle gelernte Merkmale können effektiv zwischen echten Werkstückgrenzen und durch Reflexionen gebildeten "falschen Kanten" unterscheiden, wodurch eine präzise Konturextraktion unter komplexer Beleuchtung erreicht wird.

#### 4.2.2 Technologie zur Entfernung falscher Kanten

Basierend auf der Deep-Learning-Merkmalsextraktion werden Nachbearbeitungsalgorithmen kombiniert, um die Kantenerkennungsergebnisse weiter zu optimieren.

*   **Methoden**: Verwenden Sie morphologische Operationen, Connected-Component-Analyse, geometrische Einschränkungen (z. B. bekannte Werkstückform-Prioris) usw., um die vom Netzwerk ausgegebenen Kanten zu verfeinern und "falsche Kanten" oder Rauschpunkte zu entfernen, die nicht den wahren Werkstückmerkmalen entsprechen.
*   **Vorteile**: Stellt sicher, dass die extrahierten Kanten die wahren physikalischen Grenzen des Werkstücks sind und keine Beleuchtungsartefakte.

#### 4.2.3 Subpixel-Positionierung

Um eine Positionierungsgenauigkeit von ±0,2 mm zu erreichen, sind eine Subpixel-Kantenerkennung und -anpassung an den extrahierten Werkstückkonturen erforderlich. Nationale und internationale Forschung hat erhebliche Fortschritte bei der Subpixel-Positionierung erzielt, insbesondere bei der hochauflösenden Bildverarbeitung [10, 11].

*   **Methoden**: Basierend auf Zernike-Momenten, Gauß-Anpassung oder Interpolationsmethoden werden die groben Pixel-Kanten verfeinert, um subpixelgenaue Kantenpunkte zu erhalten. Anschließend werden der geometrische Mittelpunkt des Werkstücks, die Hauptachsenrichtung und andere wichtige Positionierungsinformationen mithilfe der Methode der kleinsten Quadrate oder anderer Optimierungsalgorithmen angepasst.
*   **Vorteile**: Verbessert die Positionierungsgenauigkeit erheblich und erfüllt strenge industrielle Anforderungen.

### 4.3 Hochpräzise Hand-Auge-Kalibrierung und visuelle Servosteuerung

#### 4.3.1 Hand-Auge-Kalibrierungsmodell

Ein hochpräzises Hand-Auge-Kalibrierungsmodell wird erstellt, um eine präzise Zuordnung zwischen dem Bildpixelkoordinatensystem und dem Roboterbasiskoordinatensystem zu erreichen. Hierfür werden klassische Tsai-Lenz [4] oder Park-Martin [5] Methoden verwendet. In den letzten Jahren sind auch Online-Hand-Auge-Kalibrierung und Deep-Learning-basierte Kalibrierungsmethoden zu Forschungsschwerpunkten geworden, die die Flexibilität und Robustheit der Kalibrierung verbessern [17, 18, 19].

*   **Prinzip**: Durch die Aufnahme von Bildern einer Kalibrierplatte in verschiedenen Posen wird die Transformationsbeziehung zwischen dem Kamerakoordinatensystem und dem Kalibrierplattenkoordinatensystem sowie die Transformationsbeziehung zwischen dem Roboterendeffektorkoordinatensystem und dem Roboterbasiskoordinatensystem ermittelt. Anschließend wird durch mathematische Ableitung die Transformationsmatrix (Hand-Auge-Matrix) zwischen dem Kamerakoordinatensystem und dem Roboterendeffektorkoordinatensystem gelöst.
*   **Genauigkeitsverbesserung**: Hochpräzise Kalibrierplatten, mehrfache Wiederholungsmessungen, Optimierungsalgorithmen (wie Levenberg-Marquardt) und Fehlerkompensationstechniken werden verwendet, um die Kalibriergenauigkeit sicherzustellen.

#### 4.3.2 Visuelle Servosteuerung

In Fällen, in denen die visuellen Positionierungskoordinaten leichte Schwankungen aufweisen oder während der Roboterbewegung, wird eine Roboterbewegungskompensationsstrategie basierend auf visuellem Feedback angewendet, um ein präzises Greifen von Werkstücken zu erreichen.

*   **Methoden**: Positionsbasierte visuelle Servosteuerung (PBVS) oder bildbasierte visuelle Servosteuerung (IBVS). PBVS berechnet die Pose des Ziels im Kamerakoordinatensystem und wandelt sie dann in das Roboterbasiskoordinatensystem um, um die Roboterbewegung zu steuern. IBVS verwendet direkt Bildmerkmalsfehler, um die Roboterbewegung zu steuern.
*   **Robustheit**: Kombinieren Sie prädiktive Steuerung, Kalman-Filterung und andere Techniken, um visuelle Positionierungsergebnisse zu glätten, Roboterzittern zu reduzieren und die Greifstabilität zu verbessern.

## 5. Fazit und Ausblick

Diese Lösung, die fortschrittliches optisches Design, Bildvorverarbeitung, Deep-Learning-Merkmalsextraktion und hochpräzise Robotersteuerungstechnologien kombiniert, zielt darauf ab, eine umfassende, robuste und kostengünstige Lösung für die 2D-Vision-geführte Präzisionssortierung von hochreflektierenden Werkstücken bereitzustellen. Zukünftige Arbeiten umfassen die tatsächliche Algorithmusimplementierung, Leistungsoptimierung und Verallgemeinerungsfähigkeiten für komplexere Werkstücke.

## 6. Referenzen

[1] Debevec, P. E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. *Proceedings of the 24th annual conference on Computer graphics and interactive techniques* (pp. 369-378). ACM.
[2] Mertens, T., Kautz, J., & Van Reeth, F. (2009). Exposure fusion: A simple and effective way to combine pictures with different exposures. *Computer Graphics Forum, 28*(1), 161-171.
[3] Drago, F., Myszkowski, K., Annen, T., & Seidel, H. P. (2003). Adaptive logarithmic mapping for displaying high contrast scenes. *Computer Graphics Forum, 22*(3), 419-426.
[4] Tsai, R. Y., & Lenz, R. K. (1989). A new technique for fully autonomous and efficient 3D robotics hand/eye calibration. *IEEE Transactions on Robotics and Automation, 5*(3), 345-358.
[5] Park, F. C., & Martin, B. J. (1994). Robot sensor calibration: A review. *Robotica, 12*(6), 505-518.
[6] Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3431-3440).
[7] He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *Proceedings of the IEEE international conference on computer vision* (pp. 2961-2969).
[8] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 779-788).
[9] Su, S. (2022). Research on the Hand–Eye Calibration Method of Variable ... *Sensors, 12*(9), 4415.
[10] "Progress in in-situ detection technology and application of high dynamic range structured light stripes." (2025). *Acta Optica Sinica*.
[11] "Review of automatic optical (vision) inspection technology and its application in defect detection." (2025). *Acta Optica Sinica*.
[12] "Research status of deep learning polarized image fusion." (2025). *Infrared and Laser Engineering*.
[13] Instrumentation for Estimating Surface Radiometry. (n.d.). *DTU Orbit*.
[14] Debevec, P. E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs. *Proceedings of the 24th annual conference on Computer graphics and interactive techniques* (pp. 369-378). ACM.
[15] Abu Ebayyeh, A. A. R. M. (2022). *Deep Learning for Automatic Optical Inspection and Quality ...*. Brunel University London.
[16] Wang, J. (n.d.). *Frontier Progress in Computational Imaging*. Carnegie Mellon University.
[17] Lin, W. (2022). Research of Online Hand–Eye Calibration Method Based ... *Sensors, 12*(9), 4415.
[18] Bahadir, O. (2023). A Deep Learning-Based Hand-eye Calibration Approach ... *University of Glasgow*.
[19] Li, L. (2023). Automatic Robot Hand-Eye Calibration Enabled by ... *arXiv preprint arXiv:2311.01335*.
