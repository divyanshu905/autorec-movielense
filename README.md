# AutoRec on MovieLens Dataset
A Masked Autoencoder-based Recommender System built on the MovieLens dataset. This project demonstrates how to reconstruct user-item ratings and generate top-N recommendations using an autoencoder architecture.

## 🚀Project Overview
**Goal**: Predict missing ratings in the MovieLens dataset and provide top-N recommendations for each user.  
**Approach**: Use an autoencoder to learn latent representations of users, masking out missing ratings during training.  
**Evaluation**: Recall@K metric on unseen ratings (test set).  
**Baseline**: Matrix factorization gave a recall @ 3 of 0.0024

## ✨Features
- Converts the MovieLens dataset into a user-item rating matrix.
- Handles sparse ratings using a mask to ignore missing entries during training.
- Trains a shallow autoencoder with ReLU activation to reconstruct ratings.
- Provides top-N recommendations for each user.
- Evaluates the model using Recall@K metric.

## 🏗 Model Details

### First Iteration:
  -**Architecture**: Single-layer autoencoder  
  -**Hidden Dimension**: 64  
  -**Loss**: Masked Mean Squared Error (only considers observed ratings)  
  -**Optimizer**: Adam  
  -**Learning Rate**: 0.001  
  -**Epochs**: 100  

  -# 📊 Evaluation
   -**Metric**: Recall@3
   -**Result**: 0.0060 (on test set)

⚠️ **Note**: This low recall indicates the model is underperforming, which is common for a simple autoencoder on highly sparse datasets. Possible improvements include:
- Adding deeper layers to the autoencoder
- Using regularization (dropout, weight decay)
- Normalizing ratings per user
