import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("weather_forecast.csv")
print(df.head())

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(df.drop('Play', axis=1)).toarray()
y = df['Play']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def train_evaluate_tree(criterion):
    # Train the decision tree classifier
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    
    # Visualize the tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=encoder.get_feature_names_out(), class_names=['No', 'Yes'])
    plt.show()
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"{criterion.upper()} Algorithm Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_encoded, y, cv=5)
    print(f"Cross-Validation Scores ({criterion.upper()}):", cv_scores)
    print(f"Mean CV Accuracy ({criterion.upper()}):", cv_scores.mean())

# Train and evaluate ID3 (entropy) and CART (gini) classifiers
train_evaluate_tree('entropy')
train_evaluate_tree('gini')
