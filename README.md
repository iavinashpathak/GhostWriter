# ✍️ GhostWriter: Context-Aware LSTM Autocomplete Engine

**GhostWriter** is a real-time, deep learning-powered typing assistant designed to combat "blank page syndrome" for technical writers. Unlike standard autocomplete systems that rely on static dictionaries or simple N-grams, GhostWriter utilizes a stacked **Long Short-Term Memory (LSTM)** neural network to understand sequence context and predict the next most probable words with high relevance.

Built as a solo sprint to demonstrate the practical application of Recurrent Neural Networks (RNNs) in Natural Language Processing (NLP), this project decouples the inference engine from the frontend to achieve low-latency performance suitable for live usage.


### 🚀 Key Features
* **Contextual Intelligence:** Uses a Word-Level LSTM trained on a corpus of technical articles to provide domain-specific jargon and phrasing suggestions.

* **Real-Time Inference:** Engineered with Streamlit's resource caching (@st.cache_resource) to load the heavy model once and deliver sub-millisecond predictions during typing.

* **Explainable AI (XAI) Interface:** Goes beyond black-box text generation by visualizing model uncertainty, displaying dynamic confidence probability bars for the top 3 word suggestions.

* **Scalable Architecture:** Modular design allows for easy swapping of datasets (e.g., switching from "Tech" to "Creative Writing") without rewriting the core application logic.


### 🛠️ Tech Stack

* **Deep Learning:** Python, TensorFlow, Keras (Sequential LSTM)
* **Data Processing:** NumPy, Pickle
* **Frontend / Deployment:** Streamlit
* **Version Control:** Git


### 🎯 Project Goal

To bridge the gap between **NLP theory** and **practical usability** by building a next-word prediction engine that is not only accurate but also interactive, transparent, and fast enough for real-time writing.


### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ghostwriter.git
cd ghostwriter

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```


### 🧠 Model Architecture
The core model is a sequential LSTM network designed for next-word prediction:

1. **Embedding Layer:** Converts input tokens into dense vectors.
1. **LSTM Layers:** Two stacked LSTM layers (128 units) with dropout to prevent overfitting.
1. **Dense Layer:** A softmax output layer mapping to the vocabulary size.

## 📄 License

Distributed under the **MIT License**.
See the `LICENSE` file for details.
