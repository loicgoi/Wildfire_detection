# Wildfire_detection
Développer un modèle de détection d’objets capable d’identifier et de localiser les zones qui brûlent sur des images satellites pour permettre une détection et un suivi des incendies, faire de la prévention, etc.

## 1. Présentation du projet
Ce projet vise à développer un modèle de **détection d’objets** capable d’identifier et de localiser les zones brûlées sur des images satellites.  
Objectifs :  
- Détection rapide et précise des zones brûlées  
- Aide à la prévention et suivi des incendies  
- Analyse et visualisation des résultats  

---

## 2. Structure du projet

```
Wildfire_detection/
├── data/ # Dossier des images et annotations
├── models/ # Scripts pour l'entraînement et la comparaison des modèles
│ ├── init.py
│ ├── compare.py
│ ├── evaluate.py
│ ├── model.py
│ └── train.py
├── prepare_data/ # Scripts pour nettoyage, préparation et exploration des données
│ ├── init.py
│ ├── data_cleaner.py
│ ├── data_explorer.py
│ ├── data_loader.py
│ └── data_preparation.py
├── tests/ # Tests unitaires
│ ├── init.py
│ ├── test_data_cleaner.py
│ ├── test_data_explorer.py
│ └── test_data_loader.py
├── main.py # Script principal pour lancer entraînement et évaluation
├── pipeline.py # Pipeline global du projet
├── pyproject.toml
├── README.md
├── requirements.txt
├── uv.lock
├── visualize.py # Visualisation du jeu de données dans FiftyOne
├── .gitignore
├── .python-version
└── data.yaml # Fichier de configuration dataset YOLO
```


---

## 3. Installation


### 1. Cloner le dépôt :
```bash
git clone https://github.com/loicgoi/Wildfire_detection.git
cd Wildfire_detection
```  


### 2. Créer un environnement virtuel :
```bash
uv venv
```  


### 3. Installer les dépendances :
```bash
uv sync
```  


## 4. Structure des données

- data/ : images satellites

- _annotations.coco.json : annotations au format COCO

Chaque annotation contient :

- image_id

- category_id

- bbox (x, y, largeur, hauteur)  
  

## 5. Usage

Possibilité de lancer main.py avec des parser en paramètres :

- --preprocess : Import, traitement et split des données
- --train : Entrainer le modèle
- --evaluate : Evaluer le meilleur modèle
- --compare : Comparer les modèles entre-eux
- --visualize : Lancer la visualisation FiftyOne

Exemple :
```bash
python main.py --compare
```
Génère un tableau comparatif (comparison_results.csv) :  


## 6. Comparaison des modèles

Exemple de tableau comparatif obtenu :

| Model   | Precision | Recall | mAP50  | mAP50-95 | Fitness |
| ------- | --------- | ------ | ------ | -------- | ------- |
| yv8s    | 0.4861    | 0.3947 | 0.4252 | 0.2179   | 0.2179  |
| y11s_DA | 0.5171    | 0.4211 | 0.4310 | 0.2230   | 0.2230  |
| y11n   | 0.6118    | 0.4842 | 0.4940 | 0.2660   | 0.2660  |
| yv8n    | 0.5146    | 0.4632 | 0.4641 | 0.2597   | 0.2597  |
| yv8s_DA | 0.5074    | 0.4737 | 0.4464 | 0.2578   | 0.2578  |

Analyse des résultats

- Meilleur modèle global : y11n (Fitness = 0.266)

- Effet de la Data Augmentation : légère amélioration de Recall

- YOLO11n dépasse les modèles YOLOv8 en précision et rappel

- Compromis vitesse/précision : YOLOv8 reste plus léger mais légèrement moins performant  


## 7. Remarques

- Ne pas versionner les données lourdes dans Git

- Utiliser le pipeline pour garantir la cohérence du traitement des données
