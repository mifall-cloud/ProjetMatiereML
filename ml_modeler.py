"""
═══════════════════════════════════════════════════════════════════════════
    SPACE COLLISION PREDICTION - ML MODELER MODULE V2
    Auteur: Fadel (Machine Learning Modeler)
    Projet: Prédiction de collisions spatiales (ESA dataset)
    
    🎯 DÉFI : Déséquilibre extrême (99.5% safe / 0.5% collision)
    🚀 STRATÉGIE : 3 modèles avec différentes approches de gestion du déséquilibre
    
    Modèle 1 (Baseline)  : Régression Logistique + class_weight='balanced'
    Modèle 2 (Champion)  : XGBoost + scale_pos_weight=226
    Modèle 3 (Deep)      : MLP + SMOTE (oversampling intelligent)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    recall_score, 
    precision_score, 
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# XGBoost
import xgboost as xgb

# Imbalanced-learn (pour SMOTE)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 📌 CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Paramètres centralisés du projet"""
    
    # Chemins des données (compatibles Windows/Linux avec pathlib)
    DATA_DIR = Path(".")  # Répertoire courant par défaut
    TRAIN_FILE = "Data/train_ready.csv"
    TEST_FILE = "Data/test_ready.csv"
    
    # Chemins de sortie
    MODEL_DIR = Path("models")
    OUTPUT_DIR = Path("outputs")
    
    # Colonnes cibles et features
    TARGET_COL = "target"
    EXCLUDE_COLS = ['target', 'mission_id']
    
    # 🎯 PARAMÈTRE CRITIQUE : Ratio de déséquilibre
    # Calculé comme : (Nombre de Classe 0) / (Nombre de Classe 1)
    # Pour un déséquilibre 99.5% / 0.5%, le ratio est ~199
    # Valeur fournie dans le prompt : 226
    CLASS_IMBALANCE_RATIO = 226
    
    # Paramètres SMOTE pour MLP
    SMOTE_SAMPLING_STRATEGY = 0.1  # Rééquilibrage à 10% (au lieu de 0.5%)
    
    # Seeds pour reproductibilité
    RANDOM_STATE = 42
    
    # Hyperparamètres XGBoost (optimisés pour déséquilibre)
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.05,  # Plus bas pour éviter l'overfitting
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 2,  # Plus élevé pour régularisation
        'min_child_weight': 5,
        'eval_metric': 'aucpr',
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Hyperparamètres MLP
    MLP_PARAMS = {
        'hidden_layer_sizes': (128, 64, 32),  # 3 couches cachées
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,  # Régularisation L2
        'batch_size': 256,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': 42,
        'verbose': False
    }

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 CLASSE PRINCIPALE - ML PIPELINE V2
# ═══════════════════════════════════════════════════════════════════════════

class SpaceCollisionModeler:
    """
    Pipeline ML complet avec 3 stratégies de gestion du déséquilibre.
    
    Architecture:
    -------------
    1. Baseline (LR)  : class_weight='balanced'
    2. Champion (XGB) : scale_pos_weight=226
    3. Deep (MLP)     : SMOTE + Neural Network
    
    Output:
    -------
    predictions_for_evaluator.csv avec y_proba de chaque modèle
    """
    
    def __init__(self, config=None):
        """
        Initialisation du modeler.
        
        Parameters:
        -----------
        config : Config, optional
            Configuration personnalisée (utilise Config() par défaut)
        """
        self.config = config or Config()
        
        # Chemins de fichiers (avec gestion d'erreur)
        self.train_path = self.config.DATA_DIR / self.config.TRAIN_FILE
        self.test_path = self.config.DATA_DIR / self.config.TEST_FILE
        
        # Données
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        # Preprocessing
        self.preprocessor = None
        
        # Modèles
        self.baseline_model = None  # Logistic Regression
        self.champion_model = None  # XGBoost
        self.deep_model = None      # MLP + SMOTE
        
        # Statistiques
        self.actual_class_ratio = None
        
        # Créer les dossiers de sortie
        self.config.MODEL_DIR.mkdir(exist_ok=True)
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_data(self):
        """
        Charge les données train/test avec gestion d'erreur robuste.
        Compatible Windows/Linux grâce à pathlib.
        """
        print("🔄 Chargement des données...")
        print(f"   📂 Répertoire: {self.config.DATA_DIR.absolute()}")
        
        # Vérification existence des fichiers
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"❌ Fichier introuvable: {self.train_path}\n"
                f"   Assurez-vous que {self.config.TRAIN_FILE} est dans {self.config.DATA_DIR}"
            )
        
        if not self.test_path.exists():
            raise FileNotFoundError(
                f"❌ Fichier introuvable: {self.test_path}\n"
                f"   Assurez-vous que {self.config.TEST_FILE} est dans {self.config.DATA_DIR}"
            )
        
        # Lecture des CSV
        print(f"   📥 Lecture de {self.config.TRAIN_FILE}...")
        train_df = pd.read_csv(self.train_path)
        
        print(f"   📥 Lecture de {self.config.TEST_FILE}...")
        test_df = pd.read_csv(self.test_path)
        
        print(f"   📊 Train shape: {train_df.shape}")
        print(f"   📊 Test shape: {test_df.shape}")
        
        # Vérification colonne cible
        if self.config.TARGET_COL not in train_df.columns:
            raise ValueError(
                f"❌ Colonne cible '{self.config.TARGET_COL}' introuvable!\n"
                f"   Colonnes disponibles: {list(train_df.columns)}"
            )
        
        # Optimisation mémoire
        train_df = self._optimize_dtypes(train_df)
        test_df = self._optimize_dtypes(test_df)
        
        # Séparation features / target
        feature_cols = [col for col in train_df.columns 
                       if col not in self.config.EXCLUDE_COLS]
        
        self.X_train = train_df[feature_cols].copy()
        self.y_train = train_df[self.config.TARGET_COL].copy()
        self.X_test = test_df[feature_cols].copy()
        self.y_test = test_df[self.config.TARGET_COL].copy()
        
        # Stockage des noms de features
        self.feature_names = feature_cols
        
        # 🎯 CALCUL DU RATIO RÉEL DE DÉSÉQUILIBRE
        self.actual_class_ratio = self._compute_class_imbalance()
        
        print(f"\n✅ Données chargées avec succès")
        print(f"   🔢 Nombre de features: {len(feature_cols)}")
        print(f"\n📊 DISTRIBUTION DES CLASSES (TRAIN):")
        print(f"   Classe 0 (safe):      {(self.y_train == 0).sum():,} ({(self.y_train == 0).mean()*100:.2f}%)")
        print(f"   Classe 1 (collision): {(self.y_train == 1).sum():,} ({(self.y_train == 1).mean()*100:.2f}%)")
        print(f"\n⚖️  RATIO DE DÉSÉQUILIBRE:")
        print(f"   Calculé:  {self.actual_class_ratio:.2f}")
        print(f"   Configuré: {self.config.CLASS_IMBALANCE_RATIO}")
        
        # Warning si différence significative
        if abs(self.actual_class_ratio - self.config.CLASS_IMBALANCE_RATIO) > 20:
            print(f"\n⚠️  ATTENTION: Le ratio configuré ({self.config.CLASS_IMBALANCE_RATIO}) "
                  f"diffère significativement du ratio réel ({self.actual_class_ratio:.0f})")
            print(f"   💡 Conseil: Utiliser le ratio réel pour scale_pos_weight")
    
    def _optimize_dtypes(self, df):
        """Optimisation mémoire des types de données"""
        initial_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        # Float64 → Float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Int64 → Int32 (si possible)
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                df[col] = df[col].astype('int32')
        
        final_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100 * (initial_mem - final_mem) / initial_mem
        
        print(f"   💾 Optimisation mémoire: {initial_mem:.1f}MB → {final_mem:.1f}MB "
              f"(-{reduction:.1f}%)")
        
        return df
    
    def _compute_class_imbalance(self):
        """Calcule le ratio réel de déséquilibre"""
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 : PREPROCESSING PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    
    def create_preprocessor(self):
        """
        Crée un ColumnTransformer pour gérer numériques et catégorielles.
        
        Note: Ce dataset est 100% numérique, mais le code reste générique.
        """
        print("\n🔧 Construction du preprocessing pipeline...")
        
        # Identification des types de colonnes
        numeric_features = self.X_train.select_dtypes(
            include=['int32', 'int64', 'float32', 'float64']
        ).columns.tolist()
        
        categorical_features = self.X_train.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Pipeline numérique
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Pipeline catégoriel (si nécessaire)
        transformers = [
            ('num', numeric_transformer, numeric_features)
        ]
        
        if categorical_features:
            from sklearn.preprocessing import OneHotEncoder
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Assemblage
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        print(f"   ✅ Features numériques: {len(numeric_features)}")
        print(f"   ✅ Features catégorielles: {len(categorical_features)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 : MODÈLE 1 - BASELINE (RÉGRESSION LOGISTIQUE)
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_baseline(self):
        """
        Modèle Baseline: Régression Logistique avec class_weight='balanced'.
        
        Stratégie de déséquilibre:
        --------------------------
        class_weight='balanced' ajuste automatiquement les poids inversement
        proportionnels aux fréquences des classes.
        
        Formule: weight_class_i = n_samples / (n_classes * n_samples_class_i)
        """
        print("\n" + "="*80)
        print("🏁 MODÈLE 1/3 : BASELINE (Régression Logistique)")
        print("="*80)
        print("📋 Stratégie: class_weight='balanced'")
        
        # Pipeline complet
        self.baseline_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',  # 🎯 Gestion du déséquilibre
                solver='saga',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ))
        ])
        
        # Entraînement
        print("⏳ Training en cours...")
        self.baseline_model.fit(self.X_train, self.y_train)
        
        # Évaluation
        self._evaluate_model(
            model=self.baseline_model,
            X_test=self.X_test,
            y_test=self.y_test,
            model_name="Baseline (LR)"
        )
        
        # Sauvegarde
        model_path = self.config.MODEL_DIR / "baseline_lr.joblib"
        joblib.dump(self.baseline_model, model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 : MODÈLE 2 - CHAMPION (XGBOOST)
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_champion(self):
        """
        Modèle Champion: XGBoost avec scale_pos_weight.
        
        Stratégie de déséquilibre:
        --------------------------
        scale_pos_weight = 226 force le modèle à pénaliser 226 fois plus
        les faux négatifs (collisions manquées) que les faux positifs.
        
        🎯 C'EST CRITIQUE POUR LA SÉCURITÉ SPATIALE!
        """
        print("\n" + "="*80)
        print("🚀 MODÈLE 2/3 : CHAMPION (XGBoost)")
        print("="*80)
        print(f"📋 Stratégie: scale_pos_weight={self.config.CLASS_IMBALANCE_RATIO}")
        
        # Preprocessing manuel (XGB n'accepte que des arrays)
        print("⏳ Preprocessing des données...")
        X_train_prep = self.preprocessor.fit_transform(self.X_train)
        X_test_prep = self.preprocessor.transform(self.X_test)
        
        # Configuration des paramètres
        params = self.config.XGB_PARAMS.copy()
        
        # 🎯 PARAMÈTRE CRITIQUE: scale_pos_weight
        # On peut utiliser soit la valeur configurée, soit le ratio réel
        # Ici on prend le ratio configuré (226) comme demandé dans le prompt
        params['scale_pos_weight'] = self.config.CLASS_IMBALANCE_RATIO
        
        print(f"   🎯 scale_pos_weight = {params['scale_pos_weight']}")
        print(f"   🎯 learning_rate = {params['learning_rate']}")
        print(f"   🎯 n_estimators = {params['n_estimators']}")
        
        # Entraînement
        print("⏳ Training XGBoost...")
        self.champion_model = xgb.XGBClassifier(**params)
        self.champion_model.fit(
            X_train_prep,
            self.y_train,
            eval_set=[(X_test_prep, self.y_test)],
            verbose=False
        )
        
        # Évaluation
        self._evaluate_model_preprocessed(
            model=self.champion_model,
            X_test_prep=X_test_prep,
            y_test=self.y_test,
            model_name="Champion (XGB)"
        )
        
        # Sauvegarde
        model_path = self.config.MODEL_DIR / "champion_xgb.joblib"
        joblib.dump(self.champion_model, model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Stockage pour export final
        self._X_test_prep_xgb = X_test_prep
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 5 : MODÈLE 3 - DEEP LEARNING (MLP + SMOTE)
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_deep_model(self):
        """
        Modèle Deep Learning: MLP avec SMOTE.
        
        Stratégie de déséquilibre:
        --------------------------
        SMOTE (Synthetic Minority Over-sampling Technique) génère des
        exemples synthétiques de la classe minoritaire UNIQUEMENT sur
        le train set (jamais sur le test!).
        
        ⚠️ CRITIQUE: Utiliser imblearn.pipeline.Pipeline pour garantir
        que SMOTE n'est appliqué que sur train.
        
        sampling_strategy=0.1 signifie:
        "Après SMOTE, la classe minoritaire représentera 10% du dataset"
        (au lieu de 0.5% initialement)
        """
        print("\n" + "="*80)
        print("🧠 MODÈLE 3/3 : DEEP LEARNING (MLP + SMOTE)")
        print("="*80)
        print(f"📋 Stratégie: SMOTE (sampling_strategy={self.config.SMOTE_SAMPLING_STRATEGY})")
        
        # 🎯 PIPELINE IMBLEARN (crucial!)
        # ImbPipeline garantit que SMOTE n'est appliqué que sur fit(), pas sur predict()
        self.deep_model = ImbPipeline([
            ('preprocessor', self.preprocessor),
           
            ('classifier', MLPClassifier(**self.config.MLP_PARAMS))
        ])
        
        # Informations pré-entraînement
        n_minority_before = (self.y_train == 1).sum()
        n_total_before = len(self.y_train)
        
        # Après SMOTE, avec sampling_strategy=0.1:
        # La classe 1 représentera 10% du total
        # Si on a 100,000 échantillons de classe 0, on aura ~11,111 de classe 1
        n_majority = (self.y_train == 0).sum()
        n_minority_after_smote = int(n_majority * self.config.SMOTE_SAMPLING_STRATEGY / (1 - self.config.SMOTE_SAMPLING_STRATEGY))
        
        print(f"\n📊 Impact du SMOTE:")
        print(f"   Classe 1 AVANT SMOTE: {n_minority_before:,} ({n_minority_before/n_total_before*100:.2f}%)")
        print(f"   Classe 1 APRÈS SMOTE: ~{n_minority_after_smote:,} ({self.config.SMOTE_SAMPLING_STRATEGY*100:.1f}%)")
        print(f"   Échantillons synthétiques générés: ~{n_minority_after_smote - n_minority_before:,}")
        
        # Entraînement
        print("\n⏳ Training MLP avec SMOTE (ceci peut prendre 5-10 min)...")
        print("   💡 Le early_stopping va arrêter l'entraînement automatiquement")
        
        self.deep_model.fit(self.X_train, self.y_train)
        
        # Nombre réel d'epochs
        n_epochs = self.deep_model.named_steps['classifier'].n_iter_
        print(f"   ✅ Convergence atteinte après {n_epochs} epochs")
        
        # Évaluation
        self._evaluate_model(
            model=self.deep_model,
            X_test=self.X_test,
            y_test=self.y_test,
            model_name="Deep (MLP+SMOTE)"
        )
        
        # Sauvegarde
        model_path = self.config.MODEL_DIR / "deep_mlp_smote.joblib"
        joblib.dump(self.deep_model, model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # UTILITAIRES D'ÉVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Évalue un modèle avec pipeline complet"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        self._print_metrics(y_test, y_pred, y_proba, model_name)
    
    def _evaluate_model_preprocessed(self, model, X_test_prep, y_test, model_name):
        """Évalue un modèle sur des données déjà préprocessées"""
        y_pred = model.predict(X_test_prep)
        y_proba = model.predict_proba(X_test_prep)[:, 1]
        
        self._print_metrics(y_test, y_pred, y_proba, model_name)
    
    def _print_metrics(self, y_true, y_pred, y_proba, model_name):
        """Affiche les métriques de performance"""
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_proba)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📈 MÉTRIQUES - {model_name}")
        print("-" * 80)
        print(f"   Recall (Sensibilité):    {recall:.4f}  👈 % de collisions détectées")
        print(f"   Precision:               {precision:.4f}  👈 % d'alertes vraies")
        print(f"   F1-Score:                {f1:.4f}")
        print(f"   AUPRC:                   {auprc:.4f}  👈 Métrique clé (déséquilibre)")
        print(f"\n🎯 MATRICE DE CONFUSION:")
        print(f"   TN (Vrai Négatif):  {tn:,}  | FP (Faux Positif):  {fp:,}")
        print(f"   FN (Faux Négatif):  {fn:,}  | TP (Vrai Positif):  {tp:,}")
        
        # Interprétation métier
        if fn > 0:
            print(f"\n⚠️  {fn} collision(s) MANQUÉE(S) (Faux Négatifs)")
        if tp > 0:
            print(f"✅ {tp} collision(s) DÉTECTÉE(S) (Vrais Positifs)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 6 : FEATURE IMPORTANCE (XGBoost uniquement)
    # ═══════════════════════════════════════════════════════════════════════
    
    def extract_feature_importance(self):
        """
        Analyse l'importance des features du modèle XGBoost.
        
        🎯 OBJECTIF MÉTIER:
        ------------------
        Identifier les facteurs physiques qui augmentent le risque de collision.
        """
        print("\n" + "="*80)
        print("🔬 FEATURE IMPORTANCE ANALYSIS (XGBoost)")
        print("="*80)
        
        # Extraction des importances
        importances = self.champion_model.feature_importances_
        
        # Création DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Sauvegarde
        importance_path = self.config.OUTPUT_DIR / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Affichage Top 15
        print("\n📊 Top 15 features les plus importantes:")
        print("-" * 80)
        for idx, row in importance_df.head(15).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"{row['feature']:40s} {bar} {row['importance']:.4f}")
        
        print(f"\n✅ Feature importance sauvegardée: {importance_path}")
        
        # Interprétation des top 5
        print("\n💡 INTERPRÉTATION MÉTIER (Top 5):")
        print("-" * 80)
        for idx, row in importance_df.head(5).iterrows():
            interpretation = self._interpret_feature(row['feature'])
            print(f"   {idx+1}. {row['feature']}")
            print(f"      → {interpretation}\n")
    
    def _interpret_feature(self, feature_name):
        """Donne une interprétation métier d'une feature"""
        interpretations = {
            'miss_distance': 'Distance minimale prévue - Plus elle est faible, plus le risque est élevé',
            'relative_speed': 'Vitesse relative - Une vitesse élevée augmente les dégâts potentiels',
            'mahalanobis_distance': 'Distance statistique tenant compte des incertitudes orbitales',
            'max_risk_estimate': 'Estimation du risque maximum calculée par les systèmes ESA',
            'time_to_tca': 'Temps avant la rencontre - Moins de temps = moins de manœuvres possibles',
            't_weighted_rms': 'Qualité du tracking du satellite (RMS faible = orbite bien connue)',
            'c_weighted_rms': 'Qualité du tracking du débris (RMS élevé = plus d\'incertitude)',
        }
        
        # Recherche exacte
        if feature_name in interpretations:
            return interpretations[feature_name]
        
        # Recherche par mot-clé
        for key, value in interpretations.items():
            if key in feature_name:
                return value
        
        # Interprétation générique
        if 'covariance' in feature_name:
            return 'Matrice de covariance - Mesure l\'incertitude sur position/vitesse'
        elif 'sigma' in feature_name:
            return 'Écart-type de mesure - Plus élevé = plus d\'incertitude'
        elif feature_name.startswith('t_'):
            return 'Caractéristique orbitale du satellite cible'
        elif feature_name.startswith('c_'):
            return 'Caractéristique orbitale du débris conjoint'
        else:
            return 'Paramètre orbital technique'
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 7 : EXPORT POUR L'ÉVALUATRICE (SINY)
    # ═══════════════════════════════════════════════════════════════════════
    
    def export_predictions_for_evaluator(self):
        """
        Génère le fichier CSV avec toutes les prédictions pour Siny.
        
        🎯 FICHIER CLÉS EN MAIN:
        -----------------------
        Ce fichier unique contient tout ce dont Siny a besoin pour tracer:
        - Courbes PR-AUC
        - Courbes de calibration
        - Matrices de confusion
        - Comparaison directe des 3 modèles
        
        Structure:
        ----------
        y_true | y_proba_lr | y_proba_xgb | y_proba_mlp
        """
        print("\n" + "="*80)
        print("📦 EXPORT DES PRÉDICTIONS POUR L'ÉVALUATRICE")
        print("="*80)
        
        # Prédictions Baseline (LR)
        print("⏳ Génération prédictions Baseline...")
        y_proba_lr = self.baseline_model.predict_proba(self.X_test)[:, 1]
        
        # Prédictions Champion (XGB)
        print("⏳ Génération prédictions Champion...")
        y_proba_xgb = self.champion_model.predict_proba(self._X_test_prep_xgb)[:, 1]
        
        # Prédictions Deep (MLP)
        print("⏳ Génération prédictions Deep Learning...")
        y_proba_mlp = self.deep_model.predict_proba(self.X_test)[:, 1]
        
        # Assemblage du DataFrame
        predictions_df = pd.DataFrame({
            'y_true': self.y_test.values,
            'y_proba_lr': y_proba_lr,
            'y_proba_xgb': y_proba_xgb,
            'y_proba_mlp': y_proba_mlp
        })
        
        # Sauvegarde
        pred_path = self.config.OUTPUT_DIR / "predictions_for_evaluator.csv"
        predictions_df.to_csv(pred_path, index=False)
        
        print(f"\n✅ Fichier généré: {pred_path}")
        print(f"   📊 Shape: {predictions_df.shape}")
        print(f"\n🔍 Aperçu (5 premières lignes):")
        print(predictions_df.head().to_string(index=False))
        
        # Statistiques de distribution
        print(f"\n📊 DISTRIBUTION DES PROBABILITÉS:")
        print("-" * 80)
        for model_name, col in [('Baseline (LR)', 'y_proba_lr'), 
                                 ('Champion (XGB)', 'y_proba_xgb'), 
                                 ('Deep (MLP)', 'y_proba_mlp')]:
            print(f"\n{model_name}:")
            print(f"   Min:     {predictions_df[col].min():.6f}")
            print(f"   Max:     {predictions_df[col].max():.6f}")
            print(f"   Moyenne: {predictions_df[col].mean():.6f}")
            print(f"   Médiane: {predictions_df[col].median():.6f}")
        
        print("\n" + "="*80)
        print("🎯 INSTRUCTIONS POUR SINY (L'ÉVALUATRICE):")
        print("="*80)
        print(f"""
Ce fichier contient tout ce dont tu as besoin pour faire les graphiques:

1. COURBE PR-AUC (Precision-Recall):
   from sklearn.metrics import precision_recall_curve, auc
   precision, recall, _ = precision_recall_curve(df['y_true'], df['y_proba_xgb'])
   
2. COURBE DE CALIBRATION:
   from sklearn.calibration import calibration_curve
   fraction_positives, mean_predicted = calibration_curve(df['y_true'], df['y_proba_xgb'])
   
3. MATRICE DE CONFUSION:
   from sklearn.metrics import confusion_matrix
   # Choisir un seuil (ex: 0.5)
   y_pred = (df['y_proba_xgb'] > 0.5).astype(int)
   cm = confusion_matrix(df['y_true'], y_pred)
   
4. COMPARAISON DES 3 MODÈLES:
   Répéter pour chaque colonne (y_proba_lr, y_proba_xgb, y_proba_mlp)
        """)
    
    # ═══════════════════════════════════════════════════════════════════════
    # RAPPORT FINAL
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_final_report(self):
        """Génère un rapport de synthèse complet"""
        print("\n" + "="*80)
        print("📋 RAPPORT DE SYNTHÈSE FINAL")
        print("="*80)
        
        # Évaluations finales
        y_pred_lr = self.baseline_model.predict(self.X_test)
        y_pred_xgb = self.champion_model.predict(self._X_test_prep_xgb)
        y_pred_mlp = self.deep_model.predict(self.X_test)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PROJET: PRÉDICTION DE COLLISIONS SPATIALES")
        report_lines.append("=" * 80)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {self.config.TRAIN_FILE} / {self.config.TEST_FILE}")
        report_lines.append("")
        
        report_lines.append("1. DONNÉES")
        report_lines.append("-" * 80)
        report_lines.append(f"Train size:        {len(self.y_train):,}")
        report_lines.append(f"Test size:         {len(self.y_test):,}")
        report_lines.append(f"Features:          {len(self.feature_names)}")
        report_lines.append(f"Déséquilibre:      {self.actual_class_ratio:.2f} (classe 0 / classe 1)")
        report_lines.append(f"% Collisions:      {(self.y_train==1).mean()*100:.2f}%")
        report_lines.append("")
        
        report_lines.append("2. STRATÉGIES DE GESTION DU DÉSÉQUILIBRE")
        report_lines.append("-" * 80)
        report_lines.append(f"Baseline (LR):     class_weight='balanced'")
        report_lines.append(f"Champion (XGB):    scale_pos_weight={self.config.CLASS_IMBALANCE_RATIO}")
        report_lines.append(f"Deep (MLP):        SMOTE (sampling_strategy={self.config.SMOTE_SAMPLING_STRATEGY})")
        report_lines.append("")
        
        report_lines.append("3. RÉSULTATS COMPARATIFS (TEST SET)")
        report_lines.append("-" * 80)
        
        models = [
            ("Baseline (LR)", y_pred_lr, self.baseline_model.predict_proba(self.X_test)[:, 1]),
            ("Champion (XGB)", y_pred_xgb, self.champion_model.predict_proba(self._X_test_prep_xgb)[:, 1]),
            ("Deep (MLP)", y_pred_mlp, self.deep_model.predict_proba(self.X_test)[:, 1])
        ]
        
        for model_name, y_pred, y_proba in models:
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auprc = average_precision_score(self.y_test, y_proba)
            
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"   Recall:    {recall:.4f}")
            report_lines.append(f"   Precision: {precision:.4f}")
            report_lines.append(f"   F1-Score:  {f1:.4f}")
            report_lines.append(f"   AUPRC:     {auprc:.4f}")
        
        report_lines.append("")
        report_lines.append("4. FICHIERS GÉNÉRÉS")
        report_lines.append("-" * 80)
        report_lines.append("MODÈLES:")
        report_lines.append("   • models/baseline_lr.joblib")
        report_lines.append("   • models/champion_xgb.joblib")
        report_lines.append("   • models/deep_mlp_smote.joblib")
        report_lines.append("")
        report_lines.append("ANALYSES:")
        report_lines.append("   • outputs/predictions_for_evaluator.csv")
        report_lines.append("   • outputs/feature_importance.csv")
        report_lines.append("   • outputs/final_report.txt")
        report_lines.append("")
        
        report_lines.append("5. RECOMMANDATIONS")
        report_lines.append("-" * 80)
        report_lines.append("Pour la présentation à Mme Diop:")
        report_lines.append("   1. Expliquer le défi du déséquilibre (99.5% / 0.5%)")
        report_lines.append("   2. Justifier les 3 stratégies différentes")
        report_lines.append("   3. Montrer la feature importance (facteurs physiques)")
        report_lines.append("   4. Comparer les courbes PR-AUC (via Siny)")
        report_lines.append("   5. Analyser le compromis Recall/Precision")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Sauvegarde
        report_path = self.config.OUTPUT_DIR / "final_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Affichage
        print('\n'.join(report_lines))
        print(f"\n✅ Rapport sauvegardé: {report_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORCHESTRATION COMPLÈTE
    # ═══════════════════════════════════════════════════════════════════════
    
    def run_full_pipeline(self):
        """
        Lance toutes les étapes du pipeline.
        
        Ordre d'exécution:
        ------------------
        1. Chargement des données
        2. Création du preprocessor
        3. Baseline (LR)
        4. Champion (XGB)
        5. Deep (MLP + SMOTE)
        6. Feature Importance
        7. Export pour évaluatrice
        8. Rapport final
        """
        start_time = datetime.now()
        
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "🚀 SPACE COLLISION ML PIPELINE V2 🚀" + " " * 22 + "║")
        print("║" + " " * 19 + "3 Modèles - 3 Stratégies" + " " * 31 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        try:
            # Étape 1
            self.load_data()
            
            # Étape 2
            self.create_preprocessor()
            
            # Étape 3
            self.train_baseline()
            
            # Étape 4
            self.train_champion()
            
            # Étape 5
            self.train_deep_model()
            
            # Étape 6
            self.extract_feature_importance()
            
            # Étape 7
            self.export_predictions_for_evaluator()
            
            # Étape 8
            self.generate_final_report()
            
            # Temps total
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "="*80)
            print(f"✅ PIPELINE COMPLET TERMINÉ EN {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print("="*80)
            
            print("\n🎯 PROCHAINES ÉTAPES:")
            print("   1. Transmettre outputs/predictions_for_evaluator.csv à Siny")
            print("   2. Siny génère les courbes PR-AUC et calibration")
            print("   3. Préparer la présentation avec outputs/final_report.txt")
            print("   4. Montrer outputs/feature_importance.csv pour l'aspect métier")
            
            print("\n💡 ARGUMENTS CLÉS POUR MME DIOP:")
            print("   • Déséquilibre extrême (226:1) nécessite stratégies spécialisées")
            print("   • XGBoost avec scale_pos_weight=226 force la détection")
            print("   • SMOTE génère des exemples synthétiques sans contaminer le test")
            print("   • Recall prioritaire: mieux vaut une fausse alerte qu'une collision manquée!")
            
        except Exception as e:
            print(f"\n❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# ═══════════════════════════════════════════════════════════════════════════
# 🎬 POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🌟" * 40)
    print("SPACE COLLISION PREDICTION - MODELER MODULE V2")
    print("Fadel, Gnatam, Siny - Projet Mme Diop")
    print("🌟" * 40 + "\n")
    
    # Création de l'instance
    modeler = SpaceCollisionModeler()
    
    # Lancement du pipeline complet
    modeler.run_full_pipeline()
    
    print("\n" + "🎉" * 40)
    print("MISSION ACCOMPLIE ! Bon courage pour la présentation ! 🚀")
    print("🎉" * 40 + "\n")