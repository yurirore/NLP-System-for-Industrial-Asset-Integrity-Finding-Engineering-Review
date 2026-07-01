# Reproducing the Trained Models

The trained model artifacts (`models/classifiers/*.pkl`, `models/fine_tuned/*`, `models/all-MiniLM-L6-v2/`, `models/flan-t5-small/`, `models/bert-base-uncased/`) are excluded from git due to their size (~1.5GB).

This guide documents how to reproduce each artifact.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

---

## 1. Pre-trained Models (Download from HuggingFace)

These are the base models that need to be downloaded before fine-tuning or inference:

```bash
python -c "
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download all-MiniLM-L6-v2
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save_pretrained('models/all-MiniLM-L6-v2-hf')
print('✅ MiniLM downloaded')

# Download FLAN-T5-Small
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer.save_pretrained('models/flan-t5-small-hf')
model.save_pretrained('models/flan-t5-small-hf')
print('✅ FLAN-T5-Small downloaded')
"

# Download BERT-base-uncased (for BERTScore evaluation)
python -c "
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('models/bert-base-uncased')
model.save_pretrained('models/bert-base-uncased')
print('✅ BERT-base-uncased downloaded')
"
```

---

## 2. Prepare Offline MiniLM

The SentenceTransformer loads from manual module paths (0_Transformer, 1_Pooling).
After downloading, copy the model structure:

```bash
python -c "
from sentence_transformers import SentenceTransformer
import shutil
import os

# Load and save locally
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Save modules explicitly
os.makedirs('models/all-MiniLM-L6-v2/0_Transformer', exist_ok=True)
os.makedirs('models/all-MiniLM-L6-v2/1_Pooling', exist_ok=True)

model[0].save_pretrained('models/all-MiniLM-L6-v2/0_Transformer')
model[1].save_pretrained('models/all-MiniLM-L6-v2/1_Pooling')
model.save_pretrained('models/all-MiniLM-L6-v2')
print('✅ Offline MiniLM ready')
"
```

---

## 3. NLTK Data (Offline)

```bash
python -c "
import nltk
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')
nltk.download('omw-1.4', download_dir='./nltk_data')
print('✅ NLTK data downloaded')
"
```

---

## 4. Hierarchical Clustering → Topic Centroids

Run the clustering notebook or script:

```python
# See notebooks/02_embedding_clustering.ipynb for full walkthrough
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import joblib
import numpy as np

model = SentenceTransformer('models/all-MiniLM-L6-v2')
embeddings = model.encode(descriptions, show_progress_bar=True)

clustering = AgglomerativeClustering(n_clusters=7, linkage='ward')
labels = clustering.fit_predict(embeddings)

centroids = {}
for i in range(7):
    mask = labels == i
    centroids[i] = embeddings[mask].mean(axis=0)

joblib.dump(centroids, 'models/classifiers/topic_centroids.pkl')
print('✅ Centroids saved')
```

---

## 5. Random Forest Classifier

```python
# See notebooks/ for details
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# embeddings, labels prepared from training data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(embeddings, recommendation_labels)

le = LabelEncoder()
le.fit(recommendation_labels)

joblib.dump(rf, 'models/classifiers/random_forest_model.pkl')
joblib.dump(le, 'models/classifiers/label_encoder.pkl')
```

---

## 6. FLAN-T5 Fine-Tuning

This is the most resource-intensive step. Run `notebooks/04_flan_t5_finetuning.ipynb`.

**Key parameters used:**
- Base model: `google/flan-t5-small`
- Epochs: 30 (best checkpoint at epoch 27)
- Batch size: 4 (per device, CPU)
- Learning rate warmup: 500 steps
- Weight decay: 0.01
- Input max length: 128 tokens
- Output max length: 64 tokens
- Evaluation strategy: epoch
- Save strategy: epoch
- Load best model at end: True (metric: eval_loss)

**Estimated time:**
- CPU (Intel Xeon, 16GB RAM): ~8 hours
- GPU (T4, 16GB VRAM): ~45 minutes

**Output:** The fine-tuned model is saved as `models/fine_tuned/summarization_model_joblib.pkl` using:

```python
import joblib
joblib.dump({'model': model, 'tokenizer': tokenizer}, 'models/fine_tuned/summarization_model_joblib.pkl')
```

---

## Quick Verify

After reproducing all artifacts, verify the pipeline works:

```bash
python -m src.pipeline
```

Expected output: single sample inference + batch test both run without errors.
