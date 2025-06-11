#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# Load the data
accs = pd.read_csv("accidents_final_data.csv")
# %%
accs.head()
# %%
# Step 2: Overview of Data
print("Shape of the dataset:", accs.shape)
print("Columns in the dataset:\n", accs.columns)
print("Summary of data:\n", accs.describe())
print("Columns and data types:\n")
accs.info()
# %%
# Step 3: Check for Missing Values
missing_values = accs.isnull().sum()
print("Missing values per column:\n", missing_values)
# %%
# Step 4: Univariate Analysis
numerical_columns = accs.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = accs.select_dtypes(include=['object']).columns
# %%
# Numerical data distributions
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(accs[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
# %%
# Limit to top 20 categories for each column
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    top_categories = accs[col].value_counts().nlargest(20).index
    sns.countplot(y=accs[accs[col].isin(top_categories)][col], order=top_categories)
    plt.title(f'Frequency of Top 20 Categories in {col}')
    plt.show()
# %%
# Step 5: Correlation Analysis
correlation_matrix = accs[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()
# %%
# Step 6: Feature Engineering
# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    accs[col] = label_encoder.fit_transform(accs[col])


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import xgboost as xgb
#from xgboost import plot_importance
from sklearn import set_config
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

#%%
accs2 = accs

#%%

# Independent variables
independent_vars = [
    'Equipment Type', 
    'Incident Number', 
    'View Obstruction', 
    'Driver In Vehicle', 
    'Track Type', 
    'Incident Year', 
    'Visibility'
]

# Target variable
target_var = 'Crossing Warning Location'

#%%
# Load your dataset (ensure accs2 is defined)
# accs2 = pd.read_csv("your_data.csv")  # Uncomment and update with the correct file path

# Encode the target variable if it's categorical (multiclass)
#label_encoder = LabelEncoder()
accs2[target_var] = label_encoder.fit_transform(accs2[target_var])

#%%
# Handle missing values (optional: based on your dataset)
accs2 = accs2.dropna()

#%%
from imblearn.over_sampling import SMOTE
# Split features and target
X = accs2[independent_vars]
y = accs2[target_var]

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize SMOTE (to oversample the minority classes)
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Resample the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#%%
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

#%%
# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
from sklearn.metrics import classification_report, precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
print("Precision for each class:", precision)
print("Recall for each class:", recall)
print("F1-Score for each class:", f1)

#%%
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print("Weighted Precision:", precision_avg)
print("Weighted Recall:", recall_avg)
print("Weighted F1-Score:", f1_avg)
#%%
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)
y_score = rf_model.predict_proba(X_test)
from itertools import cycle
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%%
# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
         label=f"Micro-average ROC curve (AUC = {roc_auc['micro']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

#%%

# %%
