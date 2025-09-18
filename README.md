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

## 4. Structure des données :

images/ : 
- images satellites  

- _annotations.coco.json : annotations au format COCO  

- Chaque annotation contient : image_id, category_id, bbox (x, y, largeur, hauteur)  

## 5. Usage

Exploration des données
```bash
jupyter notebook notebooks/exploration.ipynb
```

Entraînement du modèle
```bash
jupyter notebook notebooks/train_model.ipynb
```

Évaluation
- Visualisation des bounding boxes prédites vs. annotations  

- Calcul des métriques : IoU, mAP, precision, recall  

## 6. Métriques
- IoU (Intersection over Union) : qualité de localisation  

- Precision / Recall : qualité de détection  

- mAP : moyenne de la précision pour toutes les classes  

## 7. Remarques
- Ne pas versionner les données lourdes dans Git  

- Utiliser le pipeline pour garantir la cohérence du traitement des données  

- Les notebooks sont utilisés pour l’exploration et le test des modèles  
