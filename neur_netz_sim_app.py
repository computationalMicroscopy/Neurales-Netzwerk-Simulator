import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Seite & Layout ---
st.set_page_config(page_title="KI-Architektur-Visualisierer", layout="wide")
st.title("🛠️ KI-Qualitätsprüfung: Struktur & Training")

# --- Funktionen ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)

def get_factory_data():
    X = np.array([[0.2,0.2], [0.3,0.1], [0.5,0.4], [0.8,0.8], [0.9,0.9], 
                  [0.1,0.8], [0.2,0.9], [0.9,0.1], [0.8,0.2], [0.5,0.1]])
    y = np.array([[1],[1],[1],[1],[1], [0],[0], [0],[0], [0]])
    return X, y

# --- Sidebar: Architektur-Einstellungen ---
with st.sidebar:
    st.header("🧱 Netzwerk-Design")
    hidden_nodes = st.slider("Neuronen im Hidden Layer", 2, 12, 5)
    lr = st.select_slider("Lern-Geschwindigkeit", options=[0.01, 0.1, 0.5], value=0.1)
    epochs = st.slider("Trainings-Epochen", 500, 5000, 1500)
    train_button = st.button("Netzwerk trainieren 🚀")

# --- Visualisierung der Netzwerk-Struktur ---
def draw_neural_network(n_hidden):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Layer-Definitionen (x-Positionen)
    layer_sizes = [2, n_hidden, 1]
    x_pos = [1, 2, 3]
    
    # Zeichne Verbindungen zuerst (damit sie hinter den Neuronen liegen)
    for i in range(len(layer_sizes) - 1):
        for n1 in range(layer_sizes[i]):
            for n2 in range(layer_sizes[i+1]):
                # y-Positionen zentrieren
                y1 = n1 - (layer_sizes[i]-1)/2
                y2 = n2 - (layer_sizes[i+1]-1)/2
                ax.plot([x_pos[i], x_pos[i+1]], [y1, y2], c='gray', alpha=0.3, lw=1)

    # Zeichne Neuronen
    labels = ["Input (Sensoren)", "Hidden Layer", "Output"]
    for i, size in enumerate(layer_sizes):
        for n in range(size):
            y = n - (size-1)/2
            ax.scatter(x_pos[i], y, s=400, c='white', edgecolors='#1f77b4', zorder=3)
        ax.text(x_pos[i], -3, labels[i], ha='center', fontsize=10, fontweight='bold')

    ax.axis('off')
    ax.set_ylim(-3.5, 3.5)
    return fig

# --- Layout-Aufteilung ---
col_net, col_map = st.columns(2)

with col_net:
    st.subheader("Aktuelle Architektur")
    st.pyplot(draw_neural_network(hidden_nodes))
    st.info(f"Dieses Netzwerk hat {2*hidden_nodes + hidden_nodes*1} gewichtete Verbindungen, die im Training optimiert werden.")

# --- Trainings-Logik ---
if train_button:
    with col_map:
        st.subheader("Lernfortschritt")
        X, y = get_factory_data()
        w1 = np.random.uniform(size=(2, hidden_nodes))
        w2 = np.random.uniform(size=(hidden_nodes, 1))
        
        plot_spot = st.empty()

        for i in range(epochs):
            l1 = sigmoid(np.dot(X, w1))
            out = sigmoid(np.dot(l1, w2))
            
            # Backprop
            err = y - out
            d_out = err * d_sigmoid(out)
            d_l1 = d_out.dot(w2.T) * d_sigmoid(l1)
            w2 += l1.T.dot(d_out) * lr
            w1 += X.T.dot(d_l1) * lr

            if i % 250 == 0:
                fig_map, ax_map = plt.subplots()
                res = 40
                _x, _y = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
                grid = np.c_[_x.ravel(), _y.ravel()]
                z = sigmoid(np.dot(sigmoid(np.dot(grid, w1)), w2)).reshape(_x.shape)
                
                ax_map.contourf(_x, _y, z, levels=20, cmap='RdYlGn', alpha=0.8)
                ax_map.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdYlGn', edgecolors='k')
                ax_map.set_xlabel("Temperatur")
                ax_map.set_ylabel("Vibration")
                plot_spot.pyplot(fig_map)
                plt.close()
        
        st.success("Training erfolgreich abgeschlossen!")
