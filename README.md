# Predicting-constructiveness-of-Steam-Game-Reviews

## Introduction:
This study aims to develop machine learning models capable of predicting whether a Steam game review is constructive or not constructive. The goal is to utilize review text, game titles, and engineered features to create predictive models that can distinguish helpful feedback from less useful commentary. By applying multiple models and evaluating text preprocessing techniques (with and without special characters), the most effective pipeline for classification is identified.

## Dataset Overview:
### id	
Unique identifier for each review (dropped later in preprocessing)
### game	
Title of the game being reviewed
### review	
The user-written text of the review
### author_playtime_at_review	
Hours played by the user at the time of review
### voted_up	
Boolean indicating whether the review is positive (Steam's metric)
### votes_up	
Number of users who found the review helpful
### votes_funny	
Number of users who found the review funny
### constructive	
Target label: 1 if review is constructive, 0 otherwise

## Data Preparation:
### Dropped Unused Columns:
The id column was removed as it did not contribute to the analysis.

### Label Skew Analysis:
The voted_up column was evaluated using a pie chart and found to be highly imbalanced (82.55% True, 17.45% False).
Due to its correlation with sentiment (rather than constructiveness), it was removed to prevent bias.

### WordCloud Analysis:
Separate word clouds were generated for constructive and non-constructive reviews to highlight word frequency differences.

### Feature Engineering:
A new column review_info was created by merging game and review fields for richer context.
This step was repeated for datasets with and without special characters, using regex for cleaning ([^a-zA-Z\s]).

### Vectorization:
CountVectorizer was applied to transform review_info into a document-term matrix.
This matrix was then joined back to the cleaned dataset (excluding the original review_info text column) for model input.

## Modeling Approaches:
### Random Forest Classifier (With Special Characters)
#### Accuracy: 0.795

#### Observations:
Performs relatively well; however, performance for the "Constructive" class was lower (Recall: 0.56).

#### Classification Report:
Not Constructive: Precision = 0.78, Recall = 0.93
Constructive:     Precision = 0.83, Recall = 0.56

### Random Forest Classifier (Without Special Characters)
#### Accuracy: 0.804

#### Observations:
Slight improvement in accuracy and F1-score for both classes after removing special characters.

#### Classification Report:
Not Constructive: Precision = 0.79, Recall = 0.94
Constructive:     Precision = 0.85, Recall = 0.57


### Bernoulli Naive Bayes (Without Special Characters)
#### Accuracy: 0.749

#### Observations:
Performs well in detecting "Not Constructive" reviews but struggles with identifying constructive ones (Recall: 0.39).

#### Classification Report:
Not Constructive: Precision = 0.72, Recall = 0.98
Constructive:     Precision = 0.92, Recall = 0.39

## Insights and Recommendations:
Removing special characters from review text slightly improved model performance, suggesting that punctuation might introduce noise in token-based models like CountVectorizer.
Bernoulli Naive Bayes showed poor recall on the minority class, making it less ideal for this use case despite high precision.
Random Forest remains the most balanced and effective model for the current dataset, especially when paired with cleaned review text.


