
# Anomaliedetektion

Dies ist ein Programm zur Anomaliedetektion. Es wird entwickelt unter der Kooperation zwischen der TH Nürnberg und der Irlbacher Blickpunkt Glas GmbH. Das Programm verwendet die Anomalib Bibliothek für die Anomaliedetektion und FiftyOne für die Visualisierung der Daten und Ergebnisse.

Im Moment dient ein Command Line Interface (CLI) als hauptsächliche  Interaktionsmethode. Es werden ausserdem .yaml-Konfigurationsdateien verwendet um Modell- oder Trainingseinstellungen zu kommunizieren. 

# Aktuelle Funktionen

 - Laden und Speichern von Anomaliedatensätzen.
 - Datenvisualisierung (Clustering, Embedding) mittels FiftyOne Webinterface
 - Laden von Anomalibmodellen mittels .yaml-Konfigdatei inklusive Pre- und Postprocessor und Evaluator
 - Kombinieren von mehreren Datensätzen
 - Training der Modelle auf einem Datensatz.
 - Abspeichern und Laden von Modellgewichten
 - Speichern von Ergebnissen mittels Datensatzabespeicherung
 - Visualisierung der Ergebnisse mittels FiftyOne Webinterface
 - Simple Aufnahme eigener Datensätze mittels angeschlossener Kamera

 
## Installation

Installation mittels pip und requirements.txt

```bash
python3 -m venv envName
pip3 install -r requirements.txt
```
    
Die von Anomalib implementierte Metrik "Adapative F1-Mass" hat in der Implementierung einen Fehler. Daher muss eine Anpassung in der f1_adapative_threshold.py vorgenommen werden. Diese ist unter anomalib/metrics/threshold/f1_adapative_threshold.py zu finden.

Die Zeile 
```python
return thresholds if thresholds.dim() == 0 else thresholds[torch.argmax(f1_score)]
```
muss zu
```python
return thresholds if thresholds.dim() == 0 else thresholds[torch.argmax(f1_score)-1]
```
abgeändert werden.


## Verwendung/Beispiele

Das Programm wird mit 
```python
python src/main.py 
```
gestartet.

Danach kann ein neues Modell mit "1" erstellt werden. Als Konfigurationsdatei kann "padim.yaml" gewählt werden.

Danach kann der verkleinerte MVTecAD Datensatz mittels "6" und "MVTecADShort" geladen werden. Dem Datensatz kann beim Laden ein beliebiger Name zur Wiedererkennung gegeben werden (z.B. "short"). Nach dem Laden des Modells sollte sich ein Browserfenster durch FiftyOne öffnen das den Datensatz visualisiert.

Nun kann das Modell mittels Option "4" und "padim_training.yaml" trainiert werden.

Nach dem Training sollten die Vorhersagen des Modells im Browser erscheinen.

