# 🎬 **AutoRec: Movie Recommendation System (MovieLens)**
A masked autoencoder–based recommender system trained on the MovieLens dataset to predict missing user–item ratings and generate Top-N personalized movie recommendations.

## 🧩 **Problem Statement**
Online platforms must recommend a small set of items a user is most likely to enjoy from a very large catalog.

This problem is challenging due to:
- Extremely sparse user–item interactions
- Strong popularity bias
- The need to rank relevant items rather than accurately reconstruct all ratings

This project tackles the problem using AutoRec, a neural collaborative filtering approach that learns user preference representations directly from the user–item matrix.

## 🛠 Solution Overview
### **Input**
- MovieLens explicit ratings dataset
- Sparse user × movie rating matrix

### **Model**
- User-based AutoRec (masked autoencoder)
- Encoder compresses a user’s rating vector into a latent preference representation
- Decoder reconstructs ratings for all items
- Missing ratings are masked during training

### **Output**
- Predicted ratings for unseen movies
- Top-N movie recommendations per user

### **Evaluation**
- Offline ranking metric: Recall@K
- Evaluated only on unseen test interactions

# 📊 **Baseline**
To ground performance, a Matrix Factorization baseline was implemented.
- Metric: Recall@3
- Result: 0.0024
This established a realistic lower bound and justified exploring neural models.

## 🧠 **Modeling Decisions & Experimental Reasoning**

### 🔬 **Experiment Summary**
| Iteration |	Key Change |	Recall@3 | Insight
| --------- | ---------- | --------- | ------- |
| 1 |	Single-layer AutoRec (100 epochs)	| 0.0060 |	Underpowered model |
| 2	| Deeper model + longer training | 0.021	| Capacity matters |
| 3 |	Dropout (late encoder) | 0.001 | Misplaced regularization hurts
| 4 |	Introspection-driven redesign |	0.015 |	Better loss–metric alignment |

### 📈 **Key Results (At a Glance)**
- AutoRec outperformed matrix factorization by ~6×
- Performance gains came from:
  - Increased representational capacity
  - Careful regularization placement
  - Aligning loss function with ranking metric
- Experimental failures directly informed better design decisions

## 🔮 **Future Improvements**
- Ranking-aware objectives (BPR / pairwise losses)
- Denoising AutoRec
- Implicit feedback modeling (Mult-VAE)
- Confidence weighting & user normalization
