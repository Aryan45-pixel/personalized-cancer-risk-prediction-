import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
import joblib

#step 2 load the data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


#step 3 Apply Smote

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

#step 4 train Autoencoder

from autoencoder import build_autoencoder

autoencoder, encoder = build_autoencoder(X_resampled.shape[1])

autoencoder.fit(
    X_resampled,
    X_resampled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Generate embeddings
X_embedded = encoder.predict(X_resampled)

#step 5 Train Random Forest

X_train, X_test, y_train, y_test = train_test_split(
    X_embedded, y_resampled, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))


# save model
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(encoder, "model/encoder.pkl")

#Improve embedding quality
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

import os
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/trained_model.pkl")

encoder.save("model/encoder.h5")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import os
os.makedirs("model", exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")

from sklearn.ensemble import RandomForestClassifier

# Train RF directly on scaled gene features
rf_gene_model = RandomForestClassifier()
rf_gene_model.fit(X_scaled, y)

joblib.dump(rf_gene_model, "model/rf_gene_model.pkl")