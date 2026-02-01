import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Seite & Layout ---
st.set_page_config(page_title="KI-Produktions-Check", layout="wide")

# --- Header mit Szenario ---
st.title("Neuronale Netze in der Produktion")
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

# --- Visualisierung der Netzwerk-Struktur (ERWEITERT MIT BESCHRIFTUNG) ---
def draw_neural_network(n_hidden, w1=None, w2=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    layer_sizes = [2, n_hidden, 1]
    x_pos = [1, 2, 3]
    
    # Normalisierungs-Funktion für Linienstärken
    def get_lw(weight):
        if weight is None: return 1
        return 0.5 + (np.abs(weight) / np.max(np.abs([w1.max(), w2.max()] if w1 is not None else [1]))) * 4

    # Zeichne Verbindungen (Layer 0 -> Layer 1)
    for n1 in range(layer_sizes[0]):
        for n2 in range(layer_sizes[1]):
            y1 = n1 - (layer_sizes[0]-1)/2
            y2 = n2 - (layer_sizes[1]-1)/2
            lw = get_lw(w1[n1, n2]) if w1 is not None else 1
            color = '#1f77b4' if (w1 is not None and w1[n1, n2] > 0) else 'gray'
            ax.plot([x_pos[0], x_pos[1]], [y1, y2], c=color, alpha=0.3, lw=lw)

    # Zeichne Verbindungen (Layer 1 -> Layer 2)
    for n1 in range(layer_sizes[1]):
        for n2 in range(layer_sizes[2]):
            y1 = n1 - (layer_sizes[1]-1)/2
            y2 = n2 - (layer_sizes[2]-1)/2
            lw = get_lw(w2[n1, n2]) if w2 is not None else 1
            color = '#1f77b4' if (w2 is not None and w2[n1, n2] > 0) else 'gray'
            ax.plot([x_pos[1], x_pos[2]], [y1, y2], c=color, alpha=0.3, lw=lw)

    # Zeichne Neuronen und Beschriftungen
    node_labels = {
        0: ["Temperatur", "Vibration"],
        1: [f"H{i+1}" for i in range(n_hidden)],
        2: ["Urteil"]
    }
    
    for i, size in enumerate(layer_sizes):
        for n in range(size):
            y = n - (size-1)/2
            ax.scatter(x_pos[i], y, s=400, c='white', edgecolors='#1f77b4', zorder=4)
            # Beschriftung der Knoten
            label = node_labels[i][n]
            ax.text(x_pos[i], y - 0.35 if i != 1 else y, label, 
                    ha='center', va='top' if i != 1 else 'center', 
                    fontsize=8, fontweight='bold', zorder=5)
            
    # Layer-Überschriften
    titles = ["Input\n(Sensoren)", "Hidden Layer\n(Verarbeitung)", "Output\n(Ergebnis)"]
    for i, title in enumerate(titles):
        ax.text(x_pos[i], 1.5 + (n_hidden*0.1), title, ha='center', fontsize=10, fontweight='bold', color='#333')

    ax.axis('off')
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    return fig

# --- Sidebar: Architektur-Einstellungen ---
with st.sidebar:
    st.header("🧱 Netzwerk-Design")
    hidden_nodes = st.slider("Neuronen im Hidden Layer", 2, 12, 5)
    lr = st.select_slider("Lern-Geschwindigkeit", options=[0.01, 0.1, 0.5], value=0.1)
    epochs = st.slider("Trainings-Epochen", 500, 5000, 2000)
    train_button = st.button("Training starten 🚀")

# --- Layout ---
col_net, col_map = st.columns([1, 1.5])

# Initialer Zustand
if 'w1' not in st.session_state:
    with col_net:
        st.subheader("Architektur")
        st.pyplot(draw_neural_network(hidden_nodes))
        st.caption("Jede Linie ist eine mathematische Gewichtung, die die KI anpasst.")

# --- Training ---
if train_button:
    with col_map:
        st.subheader("Die KI lernt die Qualitätsregeln...")
        X, y = get_factory_data()
        w1 = np.random.uniform(-1, 1, size=(2, hidden_nodes))
        w2 = np.random.uniform(-1, 1, size=(hidden_nodes, 1))
        plot_spot = st.empty()
        net_spot = col_net.empty()

        for i in range(epochs):
            l1 = sigmoid(np.dot(X, w1))
            out = sigmoid(np.dot(l1, w2))
            err = y - out
            d_out = err * d_sigmoid(out)
            d_l1 = d_out.dot(w2.T) * d_sigmoid(l1)
            w2 += l1.T.dot(d_out) * lr
            w1 += X.T.dot(d_l1) * lr

            if i % 250 == 0:
                # Update Karte
                fig_m, ax_m = plt.subplots()
                res = 50
                _x, _y = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
                grid = np.c_[_x.ravel(), _y.ravel()]
                z = sigmoid(np.dot(sigmoid(np.dot(grid, w1)), w2)).reshape(_x.shape)
                ax_m.contourf(_x, _y, z, levels=20, cmap='RdYlGn', alpha=0.8)
                ax_m.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdYlGn', edgecolors='k', s=60)
                ax_m.set_xlabel("Temperatur")
                ax_m.set_ylabel("Vibration")
                plot_spot.pyplot(fig_m)
                plt.close()
                
                # Update Netzwerk-Grafik
                net_spot.pyplot(draw_neural_network(hidden_nodes, w1, w2))

        st.session_state.w1, st.session_state.w2 = w1, w2
        st.success("✅ Modell bereit für die Produktion!")
        
        # --- ERWEITERTE LEGENDE ---
        st.markdown("---")
        st.subheader("🎨 Legende der Visualisierung")
        
        leg1, leg2 = st.columns(2)
        with leg1:
            st.markdown("**1. Die Datenpunkte (Einzelne Kreise):**")
            st.write("Dies sind die historischen Messwerte aus der Fabrik.")
            st.markdown("* 🟢 **Punkt ist Grün:** Dieses Bauteil war ein Gutteil.")
            st.markdown("* 🔴 **Punkt ist Rot:** Dieses Bauteil war Ausschuss.")
            
        with leg2:
            st.markdown("**2. Das Hintergrundfeld (Die 'Wissens-Karte'):**")
            st.write("Dies zeigt, wie die KI den gesamten Bereich bewertet.")
            st.markdown("* 🟩 **Grüner Bereich:** Hier würde die KI neue Teile als 'Gut' einstufen.")
            st.markdown("* 🟥 **Roter Bereich:** Hier würde die KI neue Teile als 'Ausschuss' ablehnen.")
            st.markdown("* 🟨 **Gelbe Übergänge:** Hier ist sich die KI unsicher.")

        st.info("**Merke:** Die Beschriftung im Netzwerk zeigt, wie die Sensordaten durch die 'Hidden Neuronen' (H1, H2...) fließen, um am Ende zum Qualitäts-Urteil zu gelangen. Die dicken Linien zeigen dabei die wichtigsten Entscheidungswege.")
