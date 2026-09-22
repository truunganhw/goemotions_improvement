# goemotions_improvement
# Topic title: Improving Multi-label Emotion Classification on GoEmotions via Knowledge Distillation and Imbalance-Aware Learning

## Installation
[Data and all trained model files are saved in this link](https://drive.google.com/drive/folders/16IEfd5eFQFBw7LRxPhVlQNrZJG4M4d5Y)
1. Open file .ipynb to check the results on code
2. Open goemotion.py file to see the Sreamlit interface

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

Figure 1: The main interface of multilabel classification, this text is "neutral", it means there is no emotion was detected in this sentence
<img width="2810" height="1704" alt="image" src="https://github.com/user-attachments/assets/05cbeb34-730d-4c84-b7a6-bebacfdfed1e" />

Figure 2: This model turned the result of multilabel in this text that included sad, happy and curiosity
<img width="2812" height="1702" alt="image" src="https://github.com/user-attachments/assets/6eddda17-f12b-4c34-ad38-2193fe83f3c3" />

Figure 3: This figure shows the threshold status of the model when it returns the results in the backbone or blackbox system
<img width="2808" height="1704" alt="image" src="https://github.com/user-attachments/assets/d001658a-353d-47e7-aa3d-2be04a81eee1" />

Figure 4: This is EDA example
<img width="2804" height="1706" alt="image" src="https://github.com/user-attachments/assets/4aabb5ad-9491-4d19-b6b8-0c8011020855" />

Figure 5: Emotions check in the one sentence
<img width="2808" height="1702" alt="image" src="https://github.com/user-attachments/assets/4171f05b-f8fa-46a1-b2ad-f244f2559193" />

Figure 6: Hardware efficiency after applying knowledge distillation comparison (1)
<img width="2802" height="1702" alt="image" src="https://github.com/user-attachments/assets/f66fb482-2537-4b70-921f-40888d33cf2c" />

Figure 7: Hardware efficiency after applying knowledge distillation comparison (2)
<img width="2806" height="1704" alt="image" src="https://github.com/user-attachments/assets/3a24b256-14ea-45e0-9010-0a5051fd7dfd" />



