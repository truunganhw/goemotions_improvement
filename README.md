# goemotions_improvement
# Topic title: Improving Multi-label Emotion Classification on GoEmotions via Knowledge Distillation and Imbalance-Aware Learning

## Installation
[Data and all trained model files are saved in this link](https://drive.google.com/drive/folders/1ielKhI1DRtTO0A8jduh_ZaGZB8dJ-Xq6?usp=sharing)
1. Open file .ipynb to check the results on code
2. Open streamlit_interface file to see the Sreamlit interface

## Overview
This study focuses on addressing the problem of multi-label emotion classification on the severely imbalanced GoEmotions dataset. It compares and replaces methods using single models such as RoBERTa-base, DeBERTa-v3, RoBERTa-large combined with the traditional Binary Cross Entropy function with a 0.5 decision threshold by employing ensemble learning techniques based on three fine-tuned versions of RoBERTa-large combined with Focal Loss, while also performing knowledge distillation from the large model to a smaller RoBERTa-base model to optimise computational cost and emotion classification capability. 
- Highlights the advantages of tuning hyperparameters
- Handling data imbalance through loss function approaches
- Applying multi-parameter ensemble learning, and demonstrates the feasibility of resource savings through knowledge distillation techniques.

Keywords: Multi-label Classification, GoEmotions, RoBERTa, DeBERTa-v3, RoBERTa Large, Focal Loss, Ensemble Learning, Knowledge Distillation, Custom Threshold.

## Objective
- Fine tune pre-trained models for specific multi-label emotion classification problem 
- Combine imbalanced data handling method via loss algorithms approach
- Choose the best performance model can detect the complex emotions
- Implement knowledge distillation, transfer teacher model knowledge to student model
- Intergrate all of phases into an interactive platform Streamlit

## Dataset
- GoEmotions Dataset, original 211.255 rows, 37 attributes, no null
- A typical imbalanced dataset, need merge data and use majority voting technique
- After processing: 57,731 entries, preprocess: replaced with [URL], [SUBREDDIT], [USER], “>”, whitespace

## Methodology
1. Business Understanding: Develop a multi-label classification system to identify 28 emotional for GoEmotions dataset
2. Data Understanding: EDA, emotion distribution, words in each sentence length, merge data and do majority voting
3. Data Preparation: clean text, tokenizer, maxlength, data type
4. Modelling: Apply BERT family models: RoBERTa, DeBERTa-v3, intergrate Focal Loss, implement Hyperparameter Diversity Ensemble model, deploy Multi-label Knowledge Distillation to RoBERTa-base student model
5. Evaluation: Compare default and custom threshold for each label, cross evaluate Hamming Loss, Macro F1-score, and each label metrics
6. Deployment: Store file and deploy on Streamlit
