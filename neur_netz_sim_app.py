import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Seite & Layout ---
st.set_page_config(page_title="KI-Produktions-Check", layout="wide")
st.title("🛠️ KI-Qualitätsprüfung: Das 'Gehirn' in der Fabrik")

# Initialisierung der Gewichte im Session-State, damit sie nach dem Training für Tests verfügbar bleiben
if 'w1' not in st.session_state:
    st.session_state.w1 = None
    st.session_state.w2 = None

# --- Funktionen ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)

def get_factory_data():
    # X: [Temperatur, Vibration] | y: 1=OK, 0=Ausschuss
    X = np.array([[0.2,0.2], [0.3,0.1], [0.5,0.4], [0.8,0.8], [0.9,0.9], 
                  [0.1,0.8], [0.2,0.9], [0.9,0.1], [0.8,0.2], [0.5,0.1]])
    y = np.array([[1],[1],[1],[1],[1], [0],[0], [0],[0], [0]])
    return X, y

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Trainings-Setup")
    nodes = st.slider("Neuronen im Hidden Layer", 2, 16, 8)
    lr = st.select_slider("Lern-Geschwindigkeit", options=[0.01, 0.1, 0.5], value=0.1)
    epochs = st.slider("Trainings-Durchläufe", 500, 5000, 2000)
    train_now = st.button("Training starten 🚀")

# --- Hauptbereich: Training ---
col_map, col_test = st.columns([2, 1])

if train_now:
    X, y = get_factory_data()
    w1 = np.random.uniform(size=(2, nodes))
    w2 = np.random.uniform(size=(nodes, 1))
    
    plot_spot = col_map.empty()
    status_spot = col_test.empty()

    for i in range(epochs):
        l1 = sigmoid(np.dot(X, w1))
        out = sigmoid(np.dot(l1, w2))
        err = y - out
        d_out = err * d_sigmoid(out)
        d_l1 = d_out.dot(w2.T) * d_sigmoid(l1)
        w2 += l1.T.dot(d_out) * lr
        w1 += X.T.dot(d_l1) * lr

        if i % 200 == 0:
            fig, ax = plt.subplots()
            res = 40
            _x, _y = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
            grid = np.c_[_x.ravel(), _y.ravel()]
            z = sigmoid(np.dot(sigmoid(np.dot(grid, w1)), w2)).reshape(_x.shape)
            ax.contourf(_x, _y, z, levels=20, cmap='RdYlGn', alpha=0.8)
            ax.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='RdYlGn', edgecolors='k')
            ax.set_title(f"KI lernt... (Schritt {i})")
            plot_spot.pyplot(fig)
            plt.close()

    st.session_state.w1, st.session_state.w2 = w1, w2
    st.success("Training beendet! Die KI hat die Qualitätsregeln verinnerlicht.")

# --- Interaktive Test-Station ---
if st.session_state.w1 is not None:
    st.divider()
    st.header("🔍 Interaktive Bauteil-Prüfung")
    st.write("Simuliere hier ein neues Bauteil und schau, was die KI sagt:")
    
    c1, c2, c3 = st.columns(3)
    t_input = c1.slider("Temperatur (Simuliert)", 0, 100, 50) / 100
    v_input = c2.slider("Vibration (Simuliert)", 0, 10, 2) / 10
    
    # Vorhersage für das neue Bauteil
    test_data = np.array([[t_input, v_input]])
    l1_test = sigmoid(np.dot(test_data, st.session_state.w1))
    prediction = sigmoid(np.dot(l1_test, st.session_state.w2))[0][0]
    
    if prediction > 0.5:
        c3.metric("KI-Urteil", "✅ GUTTEIL")
        st.info(f"Sicherheit der KI: {prediction*100:.1f}%")
    else:
        c3.metric("KI-Urteil", "❌ AUSSCHUSS")
        st.warning(f"Sicherheit der KI: {(1-prediction)*100:.1f}%")
