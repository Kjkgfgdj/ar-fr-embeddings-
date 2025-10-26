# ar-fr-embeddings

Cross‑lingual **Arabic ↔ French word embeddings** trained with a **skip‑gram** model (window size 5, 300 dimensions).  
Includes notebooks for training, intrinsic/extrinsic evaluation, and model visualization.

**Author:** Abdulaziz Almakdhdhoub



## Repository contents

```
.
├── ArbFrVec.ipynb                       # Preprocessing + training the Word2Vec model
├── instrintic_evaluation.ipynb          # Intrinsic evaluation (similarity, nearest neighbors, etc.)
├── ArbFrVec_Extrinsic_Evaluation.ipynb  # Extrinsic evaluation (simple downstream checks)
├── models_visualization.ipynb           # 2‑D projections and qualitative analysis
├── google_ar2fr_single.pkl              # AR→FR seed dictionary / pairs for evaluation
├── google_fr2ar_single.pkl              # FR→AR seed dictionary / pairs for evaluation
└── randomshuffle_5window_skipgram_300size.model  # Trained model (download link below)
```
> The `.pkl` files are small helper artifacts used by the evaluation notebooks.



## Setup

- Python 3.9+ recommended.

Install core dependencies:

```bash
pip install -U gensim numpy pandas scikit-learn matplotlib jupyter umap-learn
```



## Download the pre‑trained model

The trained 300‑dimensional skip‑gram model (window=5) is available on Google Drive:

**Download:** https://drive.google.com/file/d/17--g6koL3QA5W6DN93AKrA0SySmX7ZSG/view?usp=drive_link

Command‑line download (optional):

```bash
pip install -U gdown
gdown --id 17--g6koL3QA5W6DN93AKrA0SySmX7ZSG -O randomshuffle_5window_skipgram_300size.model
```

Place the file at the repository root (or adjust the path in your code).



## Quick usage

```python
# If the model was saved with gensim's Word2Vec.save(...)
from gensim.models import Word2Vec
model = Word2Vec.load("randomshuffle_5window_skipgram_300size.model")

# Example query (replace with a token present in your vocab)
model.wv.most_similar("كتاب", topn=10)
```

If you only have vectors (KeyedVectors):

```python
from gensim.models import KeyedVectors
kv = KeyedVectors.load("randomshuffle_5window_skipgram_300size.model", mmap="r")
```



## Notebooks

- **Training:** `ArbFrVec.ipynb`  
  - Loads/preprocesses sentence‑aligned AR↔FR data (e.g., MultiUN; not included).  
  - Trains a 300‑dimensional skip‑gram model with window size 5.  
  - Uses simple random word shuffling as augmentation.

- **Intrinsic evaluation:** `instrintic_evaluation.ipynb`  
  - Nearest neighbors, word similarity probes, frequency effects.

- **Extrinsic evaluation:** `ArbFrVec_Extrinsic_Evaluation.ipynb`  
  - Simple downstream checks and cross‑lingual lookup.  
  - Uses `google_ar2fr_single.pkl` and `google_fr2ar_single.pkl`.

- **Visualization:** `models_visualization.ipynb`  
  - t‑SNE/UMAP projections and cluster inspection.



## Model details (summary)

- Algorithm: Word2Vec **skip‑gram**  
- Window size: **5**  
- Dimensions: **300**  
- Augmentation: Random word shuffling within sentences  
- Training data: Sentence‑aligned Arabic–French text (e.g., MultiUN).  
  The corpus is **not** redistributed here; obtain it from the original source and follow its license.

For exact hyper‑parameters (epochs, `min_count`, negative sampling, subsampling, etc.), see `ArbFrVec.ipynb`.

