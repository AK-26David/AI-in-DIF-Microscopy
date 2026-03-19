import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DIFEnsembleClassifier:
    def __init__(self, csv_path, random_state=42):
        self.csv_path = csv_path
        self.random_state = random_state
        self.df = pd.read_csv(csv_path)
        self._prepare_data()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def _prepare_data(self):
        feature_cols = [c for c in self.df.columns if c not in ['filename', 'label']]
        self.feature_names = feature_cols
        self.X = self.df[feature_cols].values
        self.y = self.df['label'].values
        self.filenames = self.df['filename'].values
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
    def split_data(self, test_size=0.2, val_size=0.1):
        X_temp, self.X_test, y_temp, self.y_test, files_temp, self.files_test = train_test_split(self.X, self.y, self.filenames, test_size=test_size, random_state=self.random_state, stratify=self.y)
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val, self.files_train, self.files_val = train_test_split(X_temp, y_temp, files_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp)
        
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_val_encoded = self.label_encoder.transform(self.y_val)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        smote = SMOTE(random_state=self.random_state)
        self.X_train_scaled, self.y_train_encoded = smote.fit_resample(self.X_train_scaled, self.y_train_encoded)
        
    def train_svm(self):
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=self.random_state, class_weight='balanced')
        svm.fit(self.X_train_scaled, self.y_train_encoded)
        y_val_pred = svm.predict(self.X_val_scaled)
        y_val_proba = svm.predict_proba(self.X_val_scaled)
        val_acc = accuracy_score(self.y_val_encoded, y_val_pred)
        val_f1 = f1_score(self.y_val_encoded, y_val_pred, average='macro')
        
        print(f"SVM:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation F1-score: {val_f1:.4f}")
        
        self.models['svm'] = svm
        self.results['svm'] = {'val_accuracy': val_acc, 'val_f1': val_f1, 'y_val_pred': y_val_pred, 'y_val_proba': y_val_proba}

    def train_random_forest(self, n_estimators=200):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=self.random_state, class_weight='balanced', n_jobs=-1)
        rf.fit(self.X_train_scaled, self.y_train_encoded)
        y_val_pred = rf.predict(self.X_val_scaled)
        y_val_proba = rf.predict_proba(self.X_val_scaled)
        val_acc = accuracy_score(self.y_val_encoded, y_val_pred)
        val_f1 = f1_score(self.y_val_encoded, y_val_pred, average='macro')

        self.feature_importance = pd.DataFrame({'feature': self.feature_names, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

        for i, row in self.feature_importance.head(10).iterrows():
            print(f"{row['feature']:40s}: {row['importance']:.4f}")
        
        print(f"RF:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation F1-score: {val_f1:.4f}")
        
        self.models['rf'] = rf
        self.results['rf'] = {'val_accuracy': val_acc, 'val_f1': val_f1, 'y_val_pred': y_val_pred, 'y_val_proba': y_val_proba}
        
    def train_xgboost(self, n_estimators=200):
        xgb = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,eval_metric='mlogloss', use_label_encoder=False)
        xgb.fit(self.X_train_scaled, self.y_train_encoded, eval_set=[(self.X_val_scaled, self.y_val_encoded)], verbose=False)
        
        y_val_pred = xgb.predict(self.X_val_scaled)
        y_val_proba = xgb.predict_proba(self.X_val_scaled)
        
        val_acc = accuracy_score(self.y_val_encoded, y_val_pred)
        val_f1 = f1_score(self.y_val_encoded, y_val_pred, average='macro')
        
        print(f"XGB:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation F1-score: {val_f1:.4f}")
        
        self.models['xgb'] = xgb
        self.results['xgb'] = {'val_accuracy': val_acc, 'val_f1': val_f1, 'y_val_pred': y_val_pred, 'y_val_proba': y_val_proba}
        
    def train_all_models(self):
        self.train_svm()
        self.train_random_forest()
        self.train_xgboost()
        
    def create_ensemble(self, weights=None):
        
        # Majority Voting
        # svm_pred = self.results['svm']['y_val_pred']
        # rf_pred = self.results['rf']['y_val_pred']
        # xgb_pred = self.results['xgb']['y_val_pred']
        # predictions = np.array([svm_pred, rf_pred, xgb_pred])
        # ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        # ensemble_proba = None
            
        # Weighted Avg
        svm_proba = self.results['svm']['y_val_proba']
        rf_proba = self.results['rf']['y_val_proba']
        xgb_proba = self.results['xgb']['y_val_proba']
        
        if weights is None:
            weights = [self.results['svm']['val_accuracy'], self.results['rf']['val_accuracy'], self.results['xgb']['val_accuracy']]
        
        weights = np.array(weights) / np.sum(weights)
        print(f"Weights: SVM={weights[0]:.3f}, RF={weights[1]:.3f}, XGB={weights[2]:.3f}")
        
        ensemble_proba = (weights[0] * svm_proba + weights[1] * rf_proba + weights[2] * xgb_proba)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        val_acc = accuracy_score(self.y_val_encoded, ensemble_pred)
        val_f1 = f1_score(self.y_val_encoded, ensemble_pred, average='weighted')
        
        print(f"Ensemble:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation F1-score: {val_f1:.4f}")
        
        self.ensemble_weights = weights
        self.results['ensemble'] = {'val_accuracy': val_acc, 'val_f1': val_f1, 'y_val_pred': ensemble_pred, 'y_val_proba': ensemble_proba, 'weights': weights}
        
    def evaluate_on_test(self):
        results_summary = []
        
        for model_name in ['svm', 'rf', 'xgb']:
            model = self.models[model_name]
            y_test_pred = model.predict(self.X_test_scaled)
            y_test_proba = model.predict_proba(self.X_test_scaled)
            acc = accuracy_score(self.y_test_encoded, y_test_pred)
            f1 = f1_score(self.y_test_encoded, y_test_pred, average='macro')
            self.results[model_name]['test_accuracy'] = acc
            self.results[model_name]['test_f1'] = f1
            self.results[model_name]['y_test_pred'] = y_test_pred
            self.results[model_name]['y_test_proba'] = y_test_proba
            
            results_summary.append({'Model': model_name.upper(), 'Accuracy': acc, 'F1-Score': f1})
            print(f"{model_name.upper():12s} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        
        # Majority Voting
        # predictions = np.array([self.results['svm']['y_test_pred'], self.results['rf']['y_test_pred'], self.results['xgb']['y_test_pred']# ])
        # ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        
        # Weighted Average
        ensemble_proba = (self.ensemble_weights[0] * self.results['svm']['y_test_proba'] + self.ensemble_weights[1] * self.results['rf']['y_test_proba'] + self.ensemble_weights[2] * self.results['xgb']['y_test_proba'])
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        ens_acc = accuracy_score(self.y_test_encoded, ensemble_pred)
        ens_f1 = f1_score(self.y_test_encoded, ensemble_pred, average='macro')
        
        self.results['ensemble']['test_accuracy'] = ens_acc
        self.results['ensemble']['test_f1'] = ens_f1
        self.results['ensemble']['y_test_pred'] = ensemble_pred
        
        results_summary.append({'Model': 'ENSEMBLE', 'Accuracy': ens_acc, 'F1-Score': ens_f1})
        
        print(f"ENSEMBLE - Accuracy: {ens_acc:.4f}, F1-score: {ens_f1:.4f}")
        self.test_results_df = pd.DataFrame(results_summary)
        
    def plot_results(self):
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        models = ['SVM', 'RF', 'XGB', 'ENSEMBLE']
        val_accs = [self.results['svm']['val_accuracy'], self.results['rf']['val_accuracy'], self.results['xgb']['val_accuracy'], self.results['ensemble']['val_accuracy']]
        test_accs = [self.results['svm']['test_accuracy'], self.results['rf']['test_accuracy'], self.results['xgb']['test_accuracy'], self.results['ensemble']['test_accuracy']]
        
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width/2, val_accs, width, label='Validation', alpha=0.8)
        ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (v, t) in enumerate(zip(val_accs, test_accs)):
            ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
            ax1.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', fontsize=8)
        
        ax2 = fig.add_subplot(gs[0, 2:])
        cm = confusion_matrix(self.y_test_encoded, self.results['ensemble']['y_test_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        ax2.set_title('Ensemble - Confusion Matrix (Test Set)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        ax3 = fig.add_subplot(gs[1, :])
        top_features = self.feature_importance.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax3.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'], fontsize=8)
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 20 Feature Importances (Random Forest)')
        ax3.invert_yaxis()
        
        ax4 = fig.add_subplot(gs[2, 0])
        for model_name, label in [('svm', 'SVM'), ('rf', 'RF'), ('xgb', 'XGB'), ('ensemble', 'Ensemble')]:
            y_true = self.y_test_encoded
            y_pred = self.results[model_name]['y_test_pred']
            f1_per_class = f1_score(y_true, y_pred, average=None)
            ax4.plot(self.label_encoder.classes_, f1_per_class, 'o-', label=label, linewidth=2)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Per-Class F1-Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        for idx, (model_name, title) in enumerate([('svm', 'SVM'), ('rf', 'Random Forest'), ('xgb', 'XGBoost')]):
            ax = fig.add_subplot(gs[2, idx+1])
            cm = confusion_matrix(self.y_test_encoded, self.results[model_name]['y_test_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar=False, xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            ax.set_title(f'{title}\nAcc: {self.results[model_name]["test_accuracy"]:.3f}')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
            ax.tick_params(labelsize=7)
        
        plt.suptitle('DIF Classification', fontsize=16, fontweight='bold')
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_classification_report(self):
        report = classification_report(self.y_test_encoded, self.results['ensemble']['y_test_pred'], target_names=self.label_encoder.classes_, digits=4)
        print(report)
        
    def save_models(self, output_dir='models'):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for name, model in self.models.items():
            filepath = output_path / f'{name}_model.pkl'
            joblib.dump(model, filepath)
        
        joblib.dump(self.scaler, output_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, output_path / 'label_encoder.pkl')
        
        ensemble_config = {'weights': self.ensemble_weights.tolist() if self.ensemble_weights is not None else None, 'feature_names': self.feature_names, 'classes': self.label_encoder.classes_.tolist()}
        
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)        
        self.feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
        
        self.test_results_df.to_csv(output_path / 'test_results.csv', index=False)
        
    def cross_validate(self, n_folds=5):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        for model_name, model in self.models.items():
            scores = cross_val_score(model, self.X_train_scaled, self.y_train_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
            print(f"{model_name.upper():12s} - Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

classifier = DIFEnsembleClassifier('dif_features.csv')
classifier.split_data(test_size=0.2, val_size=0.1)
classifier.train_all_models()
classifier.cross_validate(n_folds=5)
classifier.create_ensemble(weights=None)
classifier.evaluate_on_test()
classifier.generate_classification_report()
classifier.plot_results()
classifier.save_models(output_dir='models')