# Résolveur d'Équations Différentielles Ordinaires

Ce projet est une application Streamlit permettant de calculer des dérivées partielles et fractionnaires grâce à SymPy.

## Accès en ligne

L’application est disponible en ligne à l’adresse suivante :  
[derivees-partielles-pidr.streamlit.app](https://derivees-partielles-pidr.streamlit.app/)

Aucune installation n'est nécessaire pour utiliser la version en ligne.

## Exécution locale

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)

### Installation

1. Clonez ce dépôt
   ```bash
   git clone https://github.com/hcosserat/derivees-partielles-streamlit.git
   cd ode-solver-streamlit
   ```
2. Installez les dépendences
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application
   ```bash
   streamlit run main.py
   ```

### Fonctionnalités

Cette application permet de :

- Calculer des dérivées partielles, gradients, matrices jaconbiennes et hessiennes de fonctions multivariées
- Indiquer des points où la fonction s'annule et des points où la dérivée s'annule
- Calculer des dérivées fractionnaires par méthode de Riemann-Liouville ou par transformée de Mellin
- Visualiser les dérivées graphiquement

Développé avec Streamlit, SymPy et Matplotlib dans le cadre du PIDR.
