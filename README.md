AutoRec on MovieLens Dataset

A Masked Autoencoder-based Recommender System built on the MovieLens dataset. This project demonstrates how to reconstruct user-item ratings and generate top-N recommendations using an autoencoder architecture.

Project Overview

Goal: Predict missing ratings in the MovieLens dataset and provide top-N recommendations for each user.

Approach: Use an autoencoder to learn latent representations of users, masking out missing ratings during training.

Evaluation: Recall@K metric on unseen ratings (test set).

Baseline: User mean ratings and matrix factorization.

Features

Converts the MovieLens dataset into a user-item rating matrix.

Handles sparse ratings using a mask to ignore missing entries during training.

Trains a shallow autoencoder with ReLU activation to reconstruct ratings.

Provides top-N recommendations for each user.

Evaluates the model using Recall@K metric.
