"""
models.py — Model A (XGBoost) + Model B (LSTM)
════════════════════════════════════════════════
Model A: XGBoost / LightGBM برای سیگنال + احتمال
Model B: LSTM برای pattern زمانی
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_score, recall_score)
import joblib, os


# ═══════════════════════════════════════════════════════════════════════════
#  Model A — XGBoost / LightGBM
# ═══════════════════════════════════════════════════════════════════════════

class ModelA_GBM:
    """
    Gradient Boosting برای classification سیگنال:
      0 = no trade
      1 = long
      2 = short
    
    خروجی: احتمال هر کلاس (prob_long, prob_short, prob_none)
    """
    
    def __init__(self, use_lightgbm: bool = True):
        self.use_lgbm = use_lightgbm
        self.model    = None
        self.scaler   = StandardScaler()
        self.feature_importance_ = None
        self._fitted  = False
    
    def _build_model(self):
        if self.use_lgbm:
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators     = 500,
                    max_depth        = 6,
                    learning_rate    = 0.05,
                    num_leaves       = 63,
                    subsample        = 0.8,
                    colsample_bytree = 0.8,
                    min_child_samples= 50,
                    reg_alpha        = 0.1,
                    reg_lambda       = 0.1,
                    class_weight     = 'balanced',
                    n_jobs           = -1,
                    verbose          = -1,
                )
            except ImportError:
                print("  LightGBM نصب نیست، از XGBoost استفاده میشه")
        
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators        = 500,
            max_depth           = 6,
            learning_rate       = 0.05,
            subsample           = 0.8,
            colsample_bytree    = 0.8,
            min_child_weight    = 5,
            reg_alpha           = 0.1,
            reg_lambda          = 0.1,
            scale_pos_weight    = 1,
            use_label_encoder   = False,
            eval_metric         = 'mlogloss',
            n_jobs              = -1,
            verbosity           = 0,
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("  [Model A] آموزش GBM...", end="", flush=True)
        
        X_tr = self.scaler.fit_transform(X_train)
        self.model = self._build_model()
        
        if X_val is not None:
            X_v = self.scaler.transform(X_val)
            # Early stopping با validation
            try:
                import lightgbm as lgb
                if isinstance(self.model, lgb.LGBMClassifier):
                    self.model.fit(
                        X_tr, y_train,
                        eval_set         = [(X_v, y_val)],
                        callbacks        = [lgb.early_stopping(50, verbose=False),
                                            lgb.log_evaluation(-1)],
                    )
                else:
                    raise ImportError
            except (ImportError, Exception):
                try:
                    self.model.fit(
                        X_tr, y_train,
                        eval_set         = [(X_v, y_val)],
                        early_stopping_rounds = 50,
                        verbose          = False,
                    )
                except Exception:
                    self.model.fit(X_tr, y_train)
        else:
            self.model.fit(X_tr, y_train)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        self._fitted = True
        print(" ✓")
        
        # ارزیابی روی train
        if X_val is not None:
            preds = self.model.predict(X_v)
            proba = self.model.predict_proba(X_v)
            
            # AUC برای long و short
            try:
                if len(np.unique(y_val)) > 1:
                    auc = roc_auc_score(
                        (y_val > 0).astype(int),
                        proba[:, 1:].sum(axis=1),
                    )
                    print(f"  [Model A] Val AUC (signal vs no-signal): {auc:.3f}")
            except Exception:
                pass
    
    def predict_proba(self, X) -> np.ndarray:
        """خروجی: [prob_none, prob_long, prob_short]"""
        if not self._fitted:
            raise RuntimeError("Model آموزش ندیده")
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)
    
    def predict_signal(self, X, thresh_long=0.55, thresh_short=0.55) -> np.ndarray:
        """
        خروجی: 0=no trade, 1=long, -1=short
        thresh: حداقل احتمال برای ورود
        """
        proba = self.predict_proba(X)
        # proba[:, 0] = none, proba[:, 1] = long, proba[:, 2] = short
        sig = np.zeros(len(proba), dtype=int)
        sig[proba[:, 1] > thresh_long]  =  1
        sig[proba[:, 2] > thresh_short] = -1
        # اگر هر دو > thresh → بزرگتر برنده
        both = (proba[:, 1] > thresh_long) & (proba[:, 2] > thresh_short)
        sig[both & (proba[:, 1] >= proba[:, 2])] =  1
        sig[both & (proba[:, 2] >  proba[:, 1])] = -1
        return sig
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path: str):
        d = joblib.load(path)
        self.model = d['model']; self.scaler = d['scaler']
        self._fitted = True


# ═══════════════════════════════════════════════════════════════════════════
#  Model B — LSTM
# ═══════════════════════════════════════════════════════════════════════════

class ModelB_LSTM:
    """
    LSTM برای یادگیری pattern زمانی.
    ورودی: sequence از آخرین `lookback` کندل
    خروجی: احتمال [none, long, short]
    """
    
    def __init__(self, lookback: int = 48, n_features: int = 30):
        self.lookback   = lookback
        self.n_features = n_features
        self.model      = None
        self.scaler     = StandardScaler()
        self._fitted    = False
        self._selected_features = None
    
    def _select_features(self, X: pd.DataFrame, top_n: int = 30) -> list:
        """انتخاب مهم‌ترین featureها برای کاهش complexity"""
        # featureهایی که Z-score و divergence هستن (مرتبط‌ترین)
        priority = [c for c in X.columns if any(k in c for k in
                    ['z_ratio', 'div_', 'rsi', 'corr', 'atr', 'ret_e', 'ret_g'])]
        rest     = [c for c in X.columns if c not in priority]
        selected = (priority + rest)[:top_n]
        return selected
    
    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """تبدیل داده به sequences برای LSTM"""
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i-self.lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    
    def _build_model(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                                  BatchNormalization, Bidirectional)
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True),
                              input_shape=(self.lookback, self.n_features)),
                Dropout(0.3),
                BatchNormalization(),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(3, activation='softmax'),  # [none, long, short]
            ])
            
            model.compile(
                optimizer = Adam(learning_rate=1e-3),
                loss      = 'sparse_categorical_crossentropy',
                metrics   = ['accuracy'],
            )
            return model, True
        
        except ImportError:
            print("  ⚠️ TensorFlow نصب نیست — Model B غیرفعاله")
            return None, False
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        
        print("  [Model B] آموزش LSTM...", end="", flush=True)
        
        # انتخاب feature
        self._selected_features = self._select_features(X_train, self.n_features)
        X_tr_sel = X_train[self._selected_features].values
        X_tr_sc  = self.scaler.fit_transform(X_tr_sel)
        X_tr_seq, y_tr_seq = self._make_sequences(X_tr_sc, y_train)
        
        model, ok = self._build_model()
        if not ok:
            self._fitted = False
            return
        
        callbacks = []
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
            ]
        except ImportError:
            pass
        
        val_data = None
        if X_val is not None:
            X_v_sel = X_val[self._selected_features].values
            X_v_sc  = self.scaler.transform(X_v_sel)
            X_v_seq, y_v_seq = self._make_sequences(X_v_sc, y_val)
            val_data = (X_v_seq, y_v_seq)
        
        # class weights برای imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_tr_seq)
        cw = compute_class_weight('balanced', classes=classes, y=y_tr_seq)
        class_weight_dict = dict(zip(classes.astype(int), cw))
        
        model.fit(
            X_tr_seq, y_tr_seq,
            epochs          = 50,
            batch_size      = 256,
            validation_data = val_data,
            class_weight    = class_weight_dict,
            callbacks       = callbacks,
            verbose         = 0,
        )
        
        self.model   = model
        self._fitted = True
        print(" ✓")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.model is None:
            # اگر LSTM نصب نیست، uniform probability برمیگردونه
            return np.ones((len(X), 3)) / 3
        
        X_sel = X[self._selected_features].values
        X_sc  = self.scaler.transform(X_sel)
        
        # padding برای اولین lookback کندلها
        result = np.ones((len(X), 3)) / 3
        
        for i in range(self.lookback, len(X_sc)):
            seq = X_sc[i-self.lookback:i].reshape(1, self.lookback, self.n_features)
            result[i] = self.model.predict(seq, verbose=0)[0]
        
        return result
    
    def predict_proba_batch(self, X: pd.DataFrame) -> np.ndarray:
        """نسخه batch برای سرعت بیشتر"""
        if not self._fitted or self.model is None:
            return np.ones((len(X), 3)) / 3
        
        X_sel = X[self._selected_features].values
        X_sc  = self.scaler.transform(X_sel)
        X_seq, _ = self._make_sequences(X_sc, np.zeros(len(X_sc)))
        
        proba = self.model.predict(X_seq, batch_size=512, verbose=0)
        
        # pad اول با 1/3
        pad = np.ones((self.lookback, 3)) / 3
        return np.vstack([pad, proba])
    
    def save(self, path: str):
        if self._fitted and self.model:
            self.model.save(path + '.keras')
            joblib.dump({
                'scaler': self.scaler,
                'features': self._selected_features,
                'lookback': self.lookback,
            }, path + '_meta.pkl')
    
    def load(self, path: str):
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path + '.keras')
            meta = joblib.load(path + '_meta.pkl')
            self.scaler = meta['scaler']
            self._selected_features = meta['features']
            self.lookback = meta['lookback']
            self._fitted = True
        except Exception as e:
            print(f"  ⚠️ بارگذاری LSTM ناموفق: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Feature importance reporter
# ═══════════════════════════════════════════════════════════════════════════

def print_feature_importance(model_a: ModelA_GBM, feature_cols: list, top_n: int = 20):
    if model_a.feature_importance_ is None:
        return
    imp = pd.Series(model_a.feature_importance_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  مهم‌ترین {top_n} feature (Model A):")
    for name, val in imp.items():
        bar = '█' * int(val / imp.max() * 20)
        print(f"    {name:<30} {bar} {val:.4f}")


if __name__ == '__main__':
    print("models.py — تست سریع")
    # تست با داده مصنوعی
    np.random.seed(42)
    n = 1000
    X = pd.DataFrame(np.random.randn(n, 40),
                     columns=[f'f_{i}' for i in range(40)])
    y = np.random.randint(0, 3, n)
    
    ma = ModelA_GBM(use_lightgbm=False)
    ma.fit(X[:800], y[:800], X[800:], y[800:])
    proba = ma.predict_proba(X[800:])
    sig   = ma.predict_signal(X[800:])
    print(f"  Model A OK — signals: {np.unique(sig, return_counts=True)}")
