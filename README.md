# 📊 EA2 – Regression mit Feedforward Neural Network (FFNN) in TensorFlow.js

Diese Webanwendung demonstriert das Erlernen einer nichtlinearen Funktion mithilfe eines **Feedforward Neural Network (FFNN)** mit **TensorFlow.js**. Ziel ist die Durchführung und Visualisierung einer Regressionsanalyse unter verschiedenen Bedingungen: ohne Rauschen, mit Rauschen sowie unter Berücksichtigung von Overfitting.

---

## 🎯 Aufgabenstellung

- **Ziel:** Approximation einer unbekannten Funktion  
  \\( y(x) = 0.5(x+0.8)(x+1.8)(x-0.2)(x-0.3)(x-1.9) + 1 \\)
- **Framework:** TensorFlow.js (rein clientseitig)
- **Visualisierung:** Plotly.js (interaktiv)
- **Noise:** Additives Rauschen (Normalverteilung, Varianz 0.05)
- **Modelle:** 3 verschiedene Trainingsszenarien:
  1. Modell ohne Rauschen
  2. Best-Fit Modell (mit Rauschen)
  3. Overfitting-Modell (mit Rauschen und zu vielen Epochen)

---

## 🧠 Lernziele

- Verstehen, wie neuronale Netze Funktionen approximieren  
- Training eines FFNN-Modells mit echten und verrauschten Daten  
- Analyse von Generalisierung und Overfitting  
- Bewertung von Trainings- und Test-Loss anhand von MSE

---

## ⚙️ Verwendete Technologien

| Bibliothek        | Verwendung                                          |
|-------------------|-----------------------------------------------------|
| **TensorFlow.js** | Aufbau, Training und Auswertung des neuronalen Netzes |
| **Plotly.js**     | Interaktive Diagramme und Vorhersagevisualisierung |
| **seedrandom.js** | Reproduzierbare Zufallszahlen für Datengenerierung |

---

## 🧪 Struktur der Anwendung

- **R1 – Datensätze:** Trainings-/Testdaten mit & ohne Rauschen
- **R2 – Modell ohne Rauschen:** Modelltraining auf idealen Daten
- **R3 – Best-Fit Modell:** Training mit Rauschen, gute Generalisierung
- **R4 – Overfitting Modell:** Training mit Rauschen, zu viele Epochen

Alle Abschnitte zeigen jeweils zwei Diagramme (Train/Test) + Loss-Werte.

---

## 🖥️ Nutzung

1. Öffne `index.html` im Browser (empfohlen: Chrome).
2. Alles wird automatisch geladen und trainiert – kein Benutzereingriff notwendig.
3. Die Diagramme erscheinen interaktiv (Zoom, Hover, Legende etc.).

---

## 📌 Hinweise

- Die Anwendung läuft **vollständig im Browser**, kein Backend erforderlich.
- **Keine Model-Speicherung** – alle Modelle werden live im Frontend trainiert.
- **Responsives Layout** für Desktop und Mobile optimiert.

---

## 👨‍💻 Autor

Erstellt im Rahmen der EA2 im Modul *Maschinelles Lernen mit JavaScript* (Stand: Juni 2025).
