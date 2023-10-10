
##################################################
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from config import Config, Constant
import warnings
warnings.filterwarnings('ignore')
##################################################
df=pd.read_csv(str(Config.DATASET_DIR) + '/' + str(Config.DATA_1DEK_EXTR))
##################################################
df.head()
##################################################
# Définition de la fonction to_numeric_with_nan
def to_numeric_with_nan(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return pd.NA  # Retourne une valeur manquante
##################################################
def label_change(value):
    if pd.isna(value):
        return pd.NA
    elif value == 0.0:
        return '0'
    elif value == 1.0:
        return '1'
    else:
        return value  # Si la valeur est différente de NaN, 0.0 et 1.0, la renvoyer telle quelle
##################################################
def load_data_dek(file_path):
    df = pd.read_csv(file_path)    
    # Supprimer les lignes contenant des valeurs NaN
    df.dropna(axis=0, inplace=True)   
    # Convertir les colonnes Year, Month et Decade en numérique (si nécessaire)
    df[['Year', 'Month', 'Decade']] = df[['Year', 'Month', 'Decade']].applymap(to_numeric_with_nan)    
    # Appliquer la fonction label_change à la colonne 'Label Secheresse'
    df['Label Secheresse'] = df['Label Secheresse'].apply(label_change)  
    # Encodage des valeurs qualitatives
    ordinal_columns = ['Station', 'Saison_Pluie']  # Liste des colonnes catégorielles ordinales
    encoder = LabelEncoder()    
    for col in ordinal_columns:
        df[col] = encoder.fit_transform(df[col])   
    # Décaler la sécheresse de deux mois en avant pour la prédiction
    df['Secheresse_future'] = df['Label Secheresse'].shift(2)
    df.dropna(axis=0, inplace=True)  
    return df

##################################################
# Prétraitement des données et entraînement du modèle
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grid):
    # Prétraitement des données
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    # Créer le sélecteur de caractéristiques basé sur l'importance des fonctionnalités
    feature_selector = SelectFromModel(model, threshold='median')

    # Créer le pipeline de prétraitement, de sélection de caractéristiques et de modèle
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

    # Créer le modèle GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')

    # Entraîner le modèle GridSearchCV
    grid_search.fit(X_train, y_train)

    # Obtenir les meilleures valeurs d'hyperparamètres
    best_params = grid_search.best_params_

    # Obtenir le modèle avec les meilleurs hyperparamètres
    best_model = grid_search.best_estimator_

    # Prédire la sécheresse deux mois à l'avance sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return best_params, best_model, accuracy, report
##################################################
features=['Station','v_wind_975','u_wind_700','u_wind_100','eau_precipitable','t_point_rosee','h_vol_sol_wat','anom_lef_dek','anom_nino_dek','Saison_Pluie']
##################################################
# Définir les modèles et leurs grilles d'hyperparamètres respectives
models = [
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [None, 10, 20, 30]
        }
    },
    {
        'name': 'Gradient Boosting',
        'model': GradientBoostingRegressor(random_state=42),
        'param_grid': {'model__n_estimators':range(1,200,5)   
        }
    },
    {
        'name': 'SVM',
        'model': SVC(random_state=42),
        'param_grid': {
        #    'model__C': [0.1, 1, 10],
        #    'model__kernel': ['linear', 'rbf','poly'],
        #    'model__gamma': [0.1, 1, 'scale','auto']
        
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf', 'poly'],
        #'model__gamma': [0.1, 1, 'scale', 'auto'],
        #'model__degree': [2, 3, 4],
        #'model__coef': [0.0, 0.5, 1.0]

        
        
        }
    }
    # Ajoutez d'autres modèles avec leurs paramètres ici
]
##################################################

data_ext_1dek=load_data_dek(str(Config.DATASET_DIR) + '/' + str(Config.DATA_1DEK_EXTR))
##################################################
data_ext_1dek.head()
##################################################
X_train, X_test, y_train, y_test = train_test_split(data_ext_1dek[features], data_ext_1dek['Secheresse_future'], test_size=0.2, random_state=42)

##################################################
for model_info in models:
    print(f'--- {model_info["name"]} ---')
    best_params, best_model, accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_info['model'], model_info['param_grid'])
    print(f'Best Hyperparameters: {best_params}')
    print(f'Accuracy: {accuracy}')
    print(report)