from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


class Preprocessor:
    def __init__(self, scaling: str = "standard", encoding: str = "onehot", impute_strategy: str = "mean"):
        self.scaling = scaling
        self.encoding = encoding
        self.impute_strategy = impute_strategy
        self.scaler = None
        self.encoder = None
        self.imputer = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()

        # Impute missing values
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_processed = pd.DataFrame(self.imputer.fit_transform(X_processed), columns=X.columns)

        # Encode categorical
        cat_cols = X_processed.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.encoder.fit_transform(X_processed[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(cat_cols))
            X_processed = X_processed.drop(columns=cat_cols)
            X_processed = pd.concat([X_processed.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Scale numerical
        num_cols = X_processed.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0:
            if self.scaling == "standard":
                self.scaler = StandardScaler()
            elif self.scaling == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling}")
            X_processed[num_cols] = self.scaler.fit_transform(X_processed[num_cols])

        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        if self.imputer:
            X_processed = pd.DataFrame(self.imputer.transform(X_processed), columns=X.columns)

        # Encode categorical
        cat_cols = X_processed.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0 and self.encoder:
            encoded = self.encoder.transform(X_processed[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(cat_cols))
            X_processed = X_processed.drop(columns=cat_cols)
            X_processed = pd.concat([X_processed.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Scale numerical
        num_cols = X_processed.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0 and self.scaler:
            X_processed[num_cols] = self.scaler.transform(X_processed[num_cols])

        return X_processed
