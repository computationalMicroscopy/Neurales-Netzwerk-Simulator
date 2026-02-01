import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- UI & Branding ---
st.set_page_config(page_title="KI in der Produktion", layout="wide")
st.title("🛠️ KI-Qualitätsprüfung im Maschinenbau")
st.write("""
Beobachte live, wie ein Neuronales Netz lernt, **Ausschuss (Rot)** von **Gutteilen (Grün)** zu unterscheiden – allein basierend auf Sensorwerten für **Temperatur** und **Vibration**.
""")

# --- Simulation der Maschinendaten ---
def get_factory_data():
    # X: [Temperatur, Vibration] (normiert 0-1)
    # y: 1 = OK, 0 = Ausschuss
    X = np.array([
        [0.2, 0.2], [0.3, 0.1], [0.5, 0.4], [0.4, 0.5], # OK Bereich
        [0.8, 0.8], [0.8, 0.9], [0.9, 0.7],             # OK (Hochlast)
        [0.1, 0.8], [0.2, 0.9],                         # Defekt (Vibration ohne Wärme)
        [0.9, 0.1], [0.8, 0.2],                         # Defekt (Überhitzung ohne Last)
        [0.5, 0.1], [0.1, 0.5]                          # Grenzfälle
    ])
    y = np.array([[1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0]])
    return X, y

# --- Mathe-Kern ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)

# --- Sidebar ---
with st.sidebar:
    st.header("Anlagen-Steuerung")
    hidden_size = st.slider("Anzahl Neuronen (Gehirn-Kapazität)", 2, 16, 6)
    lr = st.select_slider("Lernrate", options=[0.01, 0.05, 0.1, 0.5], value=0.1)
    steps = st.slider("Trainings-Zyklen", 500, 5000, 2000)
    train_button = st.button("Training in der Fabrik starten")

# --- Trainings-Loop ---
if train_button:
    X, y = get_factory_data()
    # Initialisierung der Gewichte
    w1 = np.random.uniform(size=(2, hidden_size))
    w2 = np.random.uniform(size=(hidden_size, 1))
    
    col1, col2 = st.columns([3, 1])
    chart_placeholder = col1.empty()
    metrics_placeholder = col2.empty()

    for step in range(steps):
        # Forward Pass
        l1 = sigmoid(np.dot(X, w1))
        output = sigmoid(np.dot(l1, w2))

        # Backpropagation (Fehlerkorrektur)
        error = y - output
        d_out = error * d_sigmoid(output)
        d_l1 = d_out.dot(w2.T) * d_sigmoid(l1)

        w2 += l1.T.dot(d_out) * lr
        w1 += X.T.dot(d_l1) * lr

        # Visualisierung alle 100 Schritte
        if step % 100 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Hintergrund-Heatmap der Entscheidung
            res = 40
            _x, _y = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
            grid = np.c_[_x.ravel(), _y.ravel()]
            z = sigmoid(np.dot(sigmoid(np.dot(grid, w1)), w2)).reshape(_x.shape)
            
            cp = ax.contourf(_x, _y, z, levels=20, cmap='RdYlGn', alpha=0.7)
            ax.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdYlGn', edgecolors='k', s=80)
            ax.set_xlabel("Temperatur (normiert)")
            ax.set_ylabel("Vibration (normiert)")
            ax.set_title(f"KI-Lernfortschritt: Schritt {step}")
            chart_placeholder.pyplot(fig)
            plt.close()
            
            metrics_placeholder.metric("Genauigkeit", f"{100 - np.mean(np.abs(error))*100:.2f} %")
            metrics_placeholder.write("Die KI bildet gerade 'Inseln' des Wissens, um Defekte zu erkennen.")

    st.success("Fertig! Die KI ist nun bereit für die Qualitätskontrolle am Fließband.")
