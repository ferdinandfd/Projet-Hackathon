import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib pour éviter les problèmes de compatibilité
import matplotlib
matplotlib.use('Agg' if 'matplotlib.backends' in str(matplotlib.get_backend()) else 'TkAgg')

class BiasDetector:
    def __init__(self, train_path, valid_path):
        """
        Initialise le détecteur de biais avec les chemins des fichiers CSV
        
        Args:
            train_path (str): Chemin vers le fichier CSV du train set
            valid_path (str): Chemin vers le fichier CSV du validation set
        """
        print("🔍 Chargement des données...")
        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        
        print(f"📊 Train set: {self.train_df.shape[0]} lignes, {self.train_df.shape[1]} colonnes")
        print(f"✅ Validation set: {self.valid_df.shape[0]} lignes, {self.valid_df.shape[1]} colonnes")
        
        # Vérifier que les colonnes sont identiques
        self.common_columns = list(set(self.train_df.columns) & set(self.valid_df.columns))
        if len(self.common_columns) != len(self.train_df.columns):
            print("⚠️  Attention: Les datasets n'ont pas exactement les mêmes colonnes")
        
        self.bias_report = {}
        
    def detect_column_types(self):
        """Détecte automatiquement les types de colonnes"""
        numerical_cols = []
        categorical_cols = []
        
        for col in self.common_columns:
            if self.train_df[col].dtype in ['int64', 'float64']:
                # Vérifier si c'est vraiment numérique ou catégoriel avec peu de valeurs
                unique_values = len(self.train_df[col].unique())
                if unique_values <= 20:  # Seuil arbitraire
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                
        return numerical_cols, categorical_cols
    
    def analyze_numerical_bias(self, numerical_cols):
        """Analyse le biais pour les variables numériques"""
        print("\n📈 Analyse des variables numériques...")
        
        n_cols = min(3, len(numerical_cols))
        if n_cols == 0:
            print("Aucune variable numérique trouvée")
            return
            
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Histogrammes superposés
            ax.hist(self.train_df[col].dropna(), alpha=0.7, label='Train', bins=30, density=True)
            ax.hist(self.valid_df[col].dropna(), alpha=0.7, label='Validation', bins=30, density=True)
            ax.set_title(f'{col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Densité')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Test de Kolmogorov-Smirnov
            try:
                ks_stat, p_value = ks_2samp(
                    self.train_df[col].dropna(), 
                    self.valid_df[col].dropna()
                )
                
                # Classification du biais
                if p_value < 0.01:
                    bias_level = "ÉLEVÉ"
                    color = "red"
                elif p_value < 0.05:
                    bias_level = "MODÉRÉ"
                    color = "orange"
                else:
                    bias_level = "FAIBLE"
                    color = "green"
                
                ax.text(0.05, 0.95, f'KS p-value: {p_value:.4f}\nBiais: {bias_level}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       color=color, fontweight='bold')
                
                self.bias_report[col] = {
                    'type': 'numerical',
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'bias_level': bias_level,
                    'train_mean': self.train_df[col].mean(),
                    'valid_mean': self.valid_df[col].mean(),
                    'train_std': self.train_df[col].std(),
                    'valid_std': self.valid_df[col].std()
                }
                
            except Exception as e:
                print(f"Erreur lors du test KS pour {col}: {e}")
        
        # Masquer les axes vides
        for j in range(len(numerical_cols), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.suptitle('Distribution des Variables Numériques - Train vs Validation', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def analyze_categorical_bias(self, categorical_cols):
        """Analyse le biais pour les variables catégorielles"""
        print("\n📊 Analyse des variables catégorielles...")
        
        if len(categorical_cols) == 0:
            print("Aucune variable catégorielle trouvée")
            return
        
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Compter les valeurs
            train_counts = self.train_df[col].value_counts(normalize=True)
            valid_counts = self.valid_df[col].value_counts(normalize=True)
            
            # Créer un DataFrame pour la visualisation
            all_categories = set(train_counts.index) | set(valid_counts.index)
            comparison_df = pd.DataFrame({
                'Train': [train_counts.get(cat, 0) for cat in all_categories],
                'Validation': [valid_counts.get(cat, 0) for cat in all_categories]
            }, index=list(all_categories))
            
            # Graphique en barres
            comparison_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{col}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Catégories')
            ax.set_ylabel('Proportion')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Test du Chi-carré
            try:
                # Créer la table de contingence
                train_counts_abs = self.train_df[col].value_counts()
                valid_counts_abs = self.valid_df[col].value_counts()
                
                all_cats = set(train_counts_abs.index) | set(valid_counts_abs.index)
                contingency_table = np.array([
                    [train_counts_abs.get(cat, 0) for cat in all_cats],
                    [valid_counts_abs.get(cat, 0) for cat in all_cats]
                ])
                
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Classification du biais
                if p_value < 0.01:
                    bias_level = "ÉLEVÉ"
                    color = "red"
                elif p_value < 0.05:
                    bias_level = "MODÉRÉ"
                    color = "orange"
                else:
                    bias_level = "FAIBLE"
                    color = "green"
                
                ax.text(0.05, 0.95, f'Chi² p-value: {p_value:.4f}\nBiais: {bias_level}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       color=color, fontweight='bold')
                
                self.bias_report[col] = {
                    'type': 'categorical',
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'bias_level': bias_level,
                    'train_distribution': train_counts.to_dict(),
                    'valid_distribution': valid_counts.to_dict()
                }
                
            except Exception as e:
                print(f"Erreur lors du test Chi² pour {col}: {e}")
        
        # Masquer les axes vides
        for j in range(len(categorical_cols), len(axes)):
            if j < len(axes):
                axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.suptitle('Distribution des Variables Catégorielles - Train vs Validation', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def create_summary_report(self):
        """Crée un rapport de synthèse"""
        print("\n" + "="*60)
        print("📋 RAPPORT DE SYNTHÈSE - DÉTECTION DE BIAIS")
        print("="*60)
        
        total_vars = len(self.bias_report)
        high_bias_vars = [col for col, info in self.bias_report.items() 
                         if info['bias_level'] == 'ÉLEVÉ']
        medium_bias_vars = [col for col, info in self.bias_report.items() 
                           if info['bias_level'] == 'MODÉRÉ']
        low_bias_vars = [col for col, info in self.bias_report.items() 
                        if info['bias_level'] == 'FAIBLE']
        
        print(f"\n📊 Variables analysées: {total_vars}")
        print(f"🔴 Biais ÉLEVÉ: {len(high_bias_vars)} variables")
        print(f"🟡 Biais MODÉRÉ: {len(medium_bias_vars)} variables") 
        print(f"🟢 Biais FAIBLE: {len(low_bias_vars)} variables")
        
        if high_bias_vars:
            print(f"\n⚠️  Variables avec biais ÉLEVÉ:")
            for var in high_bias_vars:
                info = self.bias_report[var]
                print(f"   • {var} (p-value: {info['p_value']:.4f})")
        
        if medium_bias_vars:
            print(f"\n⚡ Variables avec biais MODÉRÉ:")
            for var in medium_bias_vars:
                info = self.bias_report[var]
                print(f"   • {var} (p-value: {info['p_value']:.4f})")
        
        # Graphique de synthèse
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique en camembert des niveaux de biais
        bias_counts = [len(high_bias_vars), len(medium_bias_vars), len(low_bias_vars)]
        labels = ['Biais Élevé', 'Biais Modéré', 'Biais Faible']
        colors = ['#ff6b6b', '#feca57', '#48dbfb']
        
        wedges, texts, autotexts = ax1.pie(bias_counts, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Répartition des Niveaux de Biais', fontsize=14, fontweight='bold')
        
        # Graphique des p-values
        variables = list(self.bias_report.keys())
        p_values = [self.bias_report[var]['p_value'] for var in variables]
        colors_bar = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'green' for p in p_values]
        
        bars = ax2.bar(range(len(variables)), p_values, color=colors_bar, alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil α=0.05')
        ax2.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='Seuil α=0.01')
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('P-value')
        ax2.set_title('P-values des Tests de Biais', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(variables)))
        ax2.set_xticklabels(variables, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        if len(high_bias_vars) > 0:
            print("   🔴 Action urgente requise! Plusieurs variables montrent un biais élevé.")
            print("   🔧 Considérez un rééchantillonnage ou une stratification.")
        elif len(medium_bias_vars) > 0:
            print("   🟡 Attention modérée requise pour certaines variables.")
            print("   🔍 Surveillez ces variables lors de l'évaluation du modèle.")
        else:
            print("   🟢 Excellent! Aucun biais significatif détecté.")
            print("   ✅ Votre validation set semble bien représentatif.")
    
    def run_full_analysis(self):
        """Lance l'analyse complète"""
        print("🚀 Début de l'analyse de biais...")
        
        # Configuration du style des graphiques
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Détection des types de colonnes
        numerical_cols, categorical_cols = self.detect_column_types()
        
        print(f"\n🔢 Variables numériques détectées: {len(numerical_cols)}")
        print(f"📝 Variables catégorielles détectées: {len(categorical_cols)}")
        
        # Analyse des variables numériques
        if numerical_cols:
            self.analyze_numerical_bias(numerical_cols)
        
        # Analyse des variables catégorielles  
        if categorical_cols:
            self.analyze_categorical_bias(categorical_cols)
        
        # Rapport de synthèse
        self.create_summary_report()
        
        print("\n✅ Analyse terminée!")
        return self.bias_report

# Fonction utilitaire pour lancer l'analyse
def analyze_bias(train_csv_path, validation_csv_path):
    """
    Fonction principale pour analyser le biais entre train et validation set
    
    Args:
        train_csv_path (str): Chemin vers le fichier CSV du train set
        validation_csv_path (str): Chemin vers le fichier CSV du validation set
    
    Returns:
        dict: Rapport détaillé de l'analyse de biais
    """
    detector = BiasDetector(train_csv_path, validation_csv_path)
    return detector.run_full_analysis()

# Exemple d'utilisation
if __name__ == "__main__":
    # 🔧 MODIFIEZ CES LIGNES AVEC VOS CHEMINS DE FICHIERS 🔧
    TRAIN_PATH = "waiting_times_train.csv"  # ← LIGNE À MODIFIER : Chemin vers votre train set
    VALIDATION_PATH = "waiting_times_X_test_final.csv"  # ← LIGNE À MODIFIER : Chemin vers votre validation set
    
    # Lancer l'analyse
    try:
        bias_report = analyze_bias(TRAIN_PATH, VALIDATION_PATH)
        print("\n🎉 Analyse complétée avec succès!")
    except FileNotFoundError as e:
        print(f"❌ Erreur: Fichier non trouvé - {e}")
        print("💡 Assurez-vous que les chemins vers vos fichiers CSV sont corrects")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")