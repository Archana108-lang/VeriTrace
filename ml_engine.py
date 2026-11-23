import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.neural_network import MLPRegressor

class FraudDetectorEngine:
    def __init__(self):
        self.mean = None
        self.std = None
        self.autoencoder = None
        self.model = None
        self.explainer = None
        self._train_models()

    def _generate_synthetic_data(self, n_samples=4000):
        np.random.seed(42)
        amount = np.random.exponential(scale=6000, size=n_samples)
        hour = np.random.randint(0, 24, size=n_samples)
        distance = np.random.exponential(scale=12, size=n_samples)
        foreign = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        tx_velocity = np.random.choice(
            [1, 2, 3, 6, 8, 10],
            size=n_samples,
            p=[0.55, 0.25, 0.08, 0.05, 0.04, 0.03]
        )

        risk_points = (
            (amount > 40000).astype(int) * 0.40 +
            ((hour <= 4) | (hour >= 23)).astype(int) * 0.25 +
            (distance > 50).astype(int) * 0.35 +
            (foreign == 1).astype(int) * 0.35 +
            (tx_velocity >= 5).astype(int) * 0.75 -
            (distance <= 15).astype(int) * 0.25
        )
        is_fraud = (risk_points >= 0.70).astype(int)

        df = pd.DataFrame({
            'amount': amount,
            'hour': hour,
            'distance': distance,
            'foreign': foreign,
            'tx_velocity': tx_velocity
        })
        return df, is_fraud

    def _train_models(self):
        df, is_fraud = self._generate_synthetic_data()

        # 1. Train Autoencoder on Normal Transactions
        normal_data = df[is_fraud == 0].values
        self.mean = normal_data.mean(axis=0)
        self.std = normal_data.std(axis=0) + 1e-7
        normal_scaled = (normal_data - self.mean) / self.std

        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(4, 2, 4),
            activation='relu',
            max_iter=250,
            random_state=42
        )
        self.autoencoder.fit(normal_scaled, normal_scaled)

        # Compute Anomaly Score
        all_scaled = (df.values - self.mean) / self.std
        reconstructions = self.autoencoder.predict(all_scaled)
        df['anomaly_score'] = np.mean(np.square(all_scaled - reconstructions), axis=1)

        # 2. Train Balanced Cost-Sensitive XGBoost
        fraud_ratio = (len(is_fraud) - sum(is_fraud)) / max(sum(is_fraud), 1)
        self.model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=np.sqrt(fraud_ratio),
            eval_metric='logloss'
        )
        self.model.fit(df, is_fraud)

        # 3. Initialize SHAP Explainer
        self.explainer = shap.TreeExplainer(self.model)

    def analyze_transaction(self, amount: float, hour: int, distance: float, foreign: int, tx_velocity: int):
        raw_input = np.array([[amount, hour, distance, foreign, tx_velocity]])
        scaled_input = (raw_input - self.mean) / self.std
        recon = self.autoencoder.predict(scaled_input)
        anomaly_score = float(np.mean(np.square(scaled_input - recon)))

        features = pd.DataFrame([{
            'amount': amount,
            'hour': hour,
            'distance': distance,
            'foreign': foreign,
            'tx_velocity': tx_velocity,
            'anomaly_score': anomaly_score
        }])

        risk_prob = float(self.model.predict_proba(features)[0][1]) * 100

        shap_values = self.explainer.shap_values(features)[0]
        feature_names = [
            'Transaction Amount (₹)',
            'Hour of Day',
            'Distance from Home',
            'Foreign Transaction',
            'Swipes in Last 1 Hr',
            'Behavior Anomaly'
        ]

        explanations = [
            {"feature": name, "impact": round(float(val), 3)}
            for name, val in zip(feature_names, shap_values)
        ]
        explanations.sort(key=lambda x: abs(x['impact']), reverse=True)

        return {
            "risk_score": round(risk_prob, 1),
            "status": "HIGH RISK - BLOCK" if risk_prob > 60 else "APPROVED",
            "explanations": explanations
        }