import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess():
    # Load dataset
    raw_path = Path("heartdataset_raw/heart_raw.csv")
    df = pd.read_csv(raw_path)

    # Pisahin fitur & target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Fitur numerik dan kategorikal
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Column Transfor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit Transform
    X_processed = preprocessor.fit_transform(X)

    # Convert sparse to dense array (universal)
    if hasattr(X_processed, "toarray"):
        X_processed_dense = X_processed.toarray()
    else:
        X_processed_dense = X_processed

    # Convert DataFrame   
    processed_df = pd.DataFrame(X_processed_dense)
    processed_df["target"] = y.values

    # Simpan Dataset yg sudah diproses
    out_dir = Path("./preprocessing")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "heartdataset_preprocessing.csv"
    processed_df.to_csv(out_path, index=False)

    print(f"Preprocessing selesai! Dataset disimpan ke: {out_path}")
    return processed_df


if __name__ == "__main__":
    df_processed = preprocess()
    print("Shape dataset:", df_processed.shape)
