from dataclasses import dataclass
from cv2.typing import MatLike
from imageProcessing import ROI_Descriptor

@dataclass
class AnwesenheitsKontrolle():
    image:MatLike                                   # Bild mit potentiellem Objekt
    ROI:ROI_Descriptor                              # Beschreibung der ROI 
    showROI:bool                                    # soll die ROI angezeigt werden aufgrund dessen die Kontrolle durchgeführt wird
    showResults:bool                                # soll das Ergebnis der Kontrolle angezeigt werden
    sollAbstand:int                                 # prüfe ob Maße realistisch
    toleranz:int                                    # um sollAbstand

    def execute(self):
        pass
