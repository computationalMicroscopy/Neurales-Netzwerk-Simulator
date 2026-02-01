import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Seite & Layout ---
st.set_page_config(page_title="KI-Produktions-Check", layout="wide")

# --- Header mit Szenario ---
st.title("🧠 Schulung: Neuronale Netze in der Produktion")
with st.expander("📖 Szenario-Beschreibung (Hier klicken für Details)", expanded=True):
    st.markdown("""
    ### Das Ziel: Automatische Qualitätskontrolle
    Wir trainieren eine KI, um **Gutteile (Grün)** von **Ausschuss (Rot)** zu unterscheiden. 
    Dazu nutzen wir zwei Sensoren einer Spritzguss-Maschine:
    
    * **X-Achse (Temperatur):** Die Wärme der Gussform (normiert).
    * **Y-Achse (Vibration):** Die Laufruhe der Maschine (normiert).
    
    **Die Herausforderung:** Es gibt keine einfache Trennlinie. Manche Zustände sind nur in Kombination kritisch. 
    Das Netz muss durch *Backpropagation* lernen, die komplexen Muster in den Daten zu erkennen.
    """)

# --- Funktionen ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)

def get_factory_data():
    # Beispielhafte Datenpunkte: [Temp, Vib]
    X = np.array([[0.2,0.2], [0.3,0.1], [0.5,0.4], [0.8,0.8], [0.9,0.9], 
                  [0.1,0.8], [0.2,0.9], [0.9,0.1], [0.8,0.2], [0.5,0.1]])
    y = np.array([[1],[1],[1],[1],[1], [0],[0], [0],[0], [0]])
    return X, y

# --- Sidebar: Architektur-Einstellungen ---
with st.sidebar:
    st.header("🧱 Netzwerk-Design")
    hidden_nodes = st.slider("Neuronen im Hidden Layer", 2, 12, 5)
    lr = st.select_slider("Lern-Geschwindigkeit", options=[0.01, 0.1, 0.5], value=0.1)
    epochs = st.slider("Trainings-Epochen", 500, 5000, 2000)
    train_button = st.button("Training starten 🚀")

# --- Visualisierung der Netzwerk-Struktur ---
def draw_neural_network(n_hidden):
    fig, ax = plt.subplots(figsize=(5, 3))
    layer_sizes = [2, n_hidden, 1]
    x_pos = [1, 2, 3]
    for i in range(len(layer_sizes) - 1):
        for n1 in range(layer_sizes[i]):
            for n2 in range(layer_sizes[i+1]):
                y1 = n1 - (layer_sizes[i]-1)/2
                y2 = n2 - (layer_sizes[i+1]-1)/2
                ax.plot([x_pos[i], x_pos[i+1]], [y1, y2], c='gray', alpha=0.3, lw=1)
    for i, size in enumerate(layer_sizes):
        for n in range(size):
            y = n - (size-1)/2
            ax.scatter(x_pos[i], y, s=300, c='white', edgecolors='#1f77b4', zorder=3)
    ax.axis('off')
    return fig

# --- Layout ---
col_net, col_map = st.columns([1, 1.5])

with col_net:
    st.subheader("Architektur")
    st.pyplot(draw_neural_network(hidden_nodes))
    st.caption("Jede Linie ist eine mathematische Gewichtung, die die KI anpasst.")

# --- Training ---
if train_button:
    with col_map:
        st.subheader("Die KI lernt die Qualitätsregeln...")
        X, y = get_factory_data()
        w1 = np.random.uniform(size=(2, hidden_nodes))
        w2 = np.random.uniform(size=(hidden_nodes, 1))
        plot_spot = st.empty()

        for i in range(epochs):
            l1 = sigmoid(np.dot(X, w1))
            out = sigmoid(np.dot(l1, w2))
            err = y - out
            d_out = err * d_sigmoid(out)
            d_l1 = d_out.dot(w2.T) * d_sigmoid(l1)
            w2 += l1.T.dot(d_out) * lr
            w1 += X.T.dot(d_l1) * lr

            if i % 250 == 0:
                fig_m, ax_m = plt.subplots()
                res = 50
                _x, _y = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
                grid = np.c_[_x.ravel(), _y.ravel()]
                z = sigmoid(np.dot(sigmoid(np.dot(grid, w1)), w2)).reshape(_x.shape)
                
                # Plotting the landscape
                contour = ax_m.contourf(_x, _y, z, levels=20, cmap='RdYlGn', alpha=0.8)
                ax_m.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdYlGn', edgecolors='k', s=60)
                ax_m.set_xlabel("Temperatur")
                ax_m.set_ylabel("Vibration")
                plot_spot.pyplot(fig_m)
                plt.close()
        
        st.success("✅ Modell bereit für die Produktion!")
        
        # --- NEU: Detaillierte Legende ---
        st.markdown("---")
        st.subheader("🎨 Legende der KI-Entscheidung")
        c1, c2, c3 = st.columns(3)
        c1.markdown("🔴 **Tiefrot:** Die KI ist sich sicher: **Ausschuss** (Wahrscheinlichkeit nahe 0).")
        c2.markdown("🟡 **Gelb/Beige:** Grauzone / Unsicherheit. Hier ist die Entscheidung 'knapp'.")
        c3.markdown("🟢 **Tiefgrün:** Die KI ist sich sicher: **Gutteil** (Wahrscheinlichkeit nahe 1).")
        st.info("**Experten-Tipp:** Beobachte während des Trainings, wie die gelben Übergangszonen schmaler werden. Das bedeutet, dass die KI lernt, schärfere Grenzen zwischen den Qualitäten zu ziehen.")
