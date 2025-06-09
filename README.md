# 📈 EA2 – Nichtlineare Regression mit einem Feedforward Neural Network (FFNN) in TensorFlow.js

Diese Webanwendung zeigt, wie ein **Feedforward Neural Network (FFNN)** in **TensorFlow.js** zur Approximation einer komplexen, nichtlinearen Funktion eingesetzt wird. Das Projekt wurde im Rahmen der EA2 des Moduls **Deep Learning (BHT MIM 20 S25)** entwickelt.

👩‍💻 **Autorin:** Ing. Aroua Sdiri  
📅 **Stand:** Juni 2025

---

## 🎯 Ziel der Anwendung

Die Anwendung soll demonstrieren, wie neuronale Netze in der Lage sind, eine analytisch gegebene Funktion zu approximieren – sowohl unter idealen Bedingungen als auch unter dem Einfluss von Rauschen bzw. Overfitting.

Die zu approximierende Funktion lautet:

\\[
y(x) = 0.5(x+0.8)(x+1.8)(x-0.2)(x-0.3)(x-1.9) + 1
\\]

---

## 🧪 Experimentelle Szenarien

Drei verschiedene Trainingsbedingungen wurden implementiert:

1. **Modell ohne Rauschen**  
   – Training auf perfekten Daten (kein Noise)

2. **Best-Fit Modell**  
   – Training auf verrauschten Daten mit moderater Epochenzahl

3. **Overfitting-Modell**  
   – Training auf verrauschten Daten mit überhöhter Epochenzahl

---

## 🧠 Lerninhalte & Ziele

- Anwendung eines FFNN zur Regressionsaufgabe
- Einfluss von Rauschen auf Trainingsverlauf verstehen
- Wirkung von Overfitting analysieren
- Visualisierung von Trainings-/Testfehlern mit MSE

---

## ⚙️ Eingesetzte Technologien

| Technologie       | Zweck                                                |
|------------------|------------------------------------------------------|
| **TensorFlow.js** | Erstellung, Training und Evaluation des FFNN        |
| **Plotly.js**     | Interaktive Visualisierung von Daten und Modellen   |
| **seedrandom.js** | Reproduzierbare Datensätze durch festen Seed        |

---

## 🗂️ Aufbau der Anwendung

- **Daten:** Erzeugung von Trainings- & Testdaten (mit/ohne Rauschen)
- **Modellarchitektur:** FFNN mit mehreren Dense-Schichten
- **Visualisierung:** Interaktive Plots für Vorhersagen und MSE-Werte
- **Layout:** Responsive Darstellung für Desktop & Mobile

---

## ▶️ Anwendung starten

1. Öffne `index.html` im Browser (z. B. Chrome).
2. Es erfolgt ein automatisches Training und Plot-Rendering.
3. Ergebnisse sind sofort interaktiv sichtbar (Zoom, Hover etc.).

---

## 📌 Weitere Hinweise

- Die App funktioniert **komplett clientseitig** – kein Server notwendig.
- **Modelle werden nicht gespeichert** – jede Ausführung trainiert neu.
- Alle Diagramme sind vollständig interaktiv und leicht vergleichbar.

---

## 🧾 Kontext

Diese Arbeit entstand im Rahmen der **EA2** im Modul  
**Deep Learning (BHT MIM 20 S25)** an der Berliner Hochschule für Technik.

