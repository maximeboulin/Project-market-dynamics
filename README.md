# Market Dynamics — Modélisation VAR du flux d'ordres d'un carnet d'ordres

> Reconstruction du flux d'ordres (ordres limites, annulations, exécutions) à partir de données haute fréquence du **Limit Order Book** de **TotalEnergies (TOTF, Euronext Paris)**, puis modélisation de sa dynamique jointe avec les prix par un **modèle vectoriel autorégressif VAR(p)**.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![pandas](https://img.shields.io/badge/pandas-numpy-150458) ![statsmodels](https://img.shields.io/badge/statsmodels-VAR-green) ![Status](https://img.shields.io/badge/projet-académique-lightgrey)

<p align="center">
  <img src="figures/lob_evolution.png" width="85%" alt="Évolution du mid-price, des volumes au meilleur bid/ask et du spread — 14 avril 2014">
</p>

---

## Sommaire

- [Objectif](#objectif)
- [Données](#données)
- [Méthodologie](#méthodologie)
- [Résultats clés](#résultats-clés)
- [Visualisations du carnet d'ordres](#visualisations-du-carnet-dordres)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation et exécution](#installation-et-exécution)
- [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)
- [Compétences mobilisées](#compétences-mobilisées)
- [Équipe](#équipe)

---

## Objectif

Le prix d'un actif est le résultat d'un équilibre permanent entre l'offre et la demande inscrites dans le carnet d'ordres. Ce projet cherche à répondre à la question suivante :

**Comment les différents flux d'ordres (passifs et agressifs) interagissent-ils entre eux et avec le prix, à l'échelle des changements de prix significatifs ?**

Pour cela, nous :

1. reconstruisons, à partir des snapshots du carnet et des transactions, les volumes d'**ordres limites**, d'**annulations** et d'**exécutions** de chaque côté du carnet ;
2. agrégeons ces flux entre deux **changements de prix significatifs** (event time plutôt que temps calendaire) ;
3. estimons un **VAR(p)** sur le vecteur

$$
Y_n = \left(r_n,\ \Delta t_n,\ V^{lo,b}_n,\ V^{lo,a}_n,\ V^{c,b}_n,\ V^{c,a}_n,\ V^{ex,b}_n,\ V^{ex,a}_n\right)
$$

| Variable | Définition |
|---|---|
| $r_n$ | log-rendement du mid-price entre deux changements de prix significatifs |
| $\Delta t_n$ | durée (s) entre deux changements de prix significatifs |
| $V^{lo,b}$ / $V^{lo,a}$ | volume d'ordres limites ajoutés côté bid / ask |
| $V^{c,b}$ / $V^{c,a}$ | volume d'annulations côté bid / ask |
| $V^{ex,b}$ / $V^{ex,a}$ | volume exécuté (market orders) côté bid / ask |

---

## Données

| Fichier | Contenu | Période |
|---|---|---|
| `TOTF_book_2014-04-13_to_2014-10-12.csv.gz` | Snapshots du carnet sur **5 niveaux** (`bid_i`, `bidQ_i`, `ask_i`, `askQ_i`) | avril → octobre 2014 (**126 jours** de bourse) |
| `TOTF_trade_2014_2017.csv.gz` | Transactions exécutées (prix, volume, horodatage) | 2014 → 2017 |

- **~500 000 mises à jour du carnet par jour** (2,5 M sur 5 jours) ;
- tick de 0,01 € (tolérance d'un demi-tick utilisée dans les comparaisons de prix) ;
- spread moyen ≈ **1,1 centime**, mid-price moyen ≈ 49 €.

> Les données brutes ne sont pas versionnées (`csv/` dans le `.gitignore`) pour des raisons de taille et de licence.

---

## Méthodologie

### 1. Extraction et mise en forme

- Lecture des fichiers compressés **par chunks** (`chunksize=200_000`) pour tenir en mémoire ;
- reconstruction des timestamps (`date` + `time`) ;
- filtrage sur les heures de marché (9h30 → 15h25) et suppression des snapshots corrompus (`bid = 0` / `ask = 0` à la clôture).

### 2. Classification des événements du carnet

En comparant deux états consécutifs du carnet au meilleur niveau :

| Variation | Trade détecté au prix ? | Événement |
|---|---|---|
| `ΔbidQ > 0` | — | **Ordre limite** (achat) |
| `ΔbidQ < 0` | oui | **Market order** (vente agressive) |
| `ΔbidQ < 0` | non | **Annulation** (achat) |

La logique est symétrique côté ask. Sur la journée du 14 avril 2014 :

| No change | Ordres limites | Market orders |
|:---:|:---:|:---:|
| **72,2 %** | **14 %** | **3,5 %** |

→ Le marché est dominé par la fourniture de liquidité ; les ordres agressifs restent rares.

### 3. Construction des variables du VAR (version vectorisée)

- **Changement de prix significatif** : le nouveau bid atteint l'ancien ask, ou le nouvel ask atteint l'ancien bid (le prix a traversé le spread). Chaque mise à jour du carnet est affectée à son bloc via `np.searchsorted`.
- **Flux passifs** (carnet) : différences de quantités sur les 5 niveaux, séparées en ajouts (`np.clip(·, 0, None)` → ordres limites) et retraits (`np.clip(·, None, 0)` → annulations), puis `groupby(bloc).sum()`.
- **Flux agressifs** (trades) : direction de chaque transaction par **tick test / zero-tick rule** (logique Lee-Ready), +1 = initié à l'achat, −1 = initié à la vente.
- **Gain de performance** : la vectorisation ramène le traitement à **≈ 0,2 s par jour** contre plusieurs minutes avec une approche ligne par ligne → **151 125 changements de prix significatifs** sur les 126 jours.

### 4. Pré-traitement pour la stationnarité

| Étape | But |
|---|---|
| Correction d'une anomalie structurelle (juin–juillet 2014) | pic simultané sur les 8 variables, lié à `Δt` |
| **Normalisation intraday** : $\tilde V_n = V_n / \lambda_V(\text{bin}_n)$ (100 bins de position dans la journée, moyennés sur 126 jours) | retirer la saisonnalité intraday (pic d'ouverture) |
| **Transformation Box-Cox** (λ estimé par maximum de vraisemblance) ; `log` pour `Δt` | rapprocher les distributions d'une gaussienne |
| **Standardisation** (moyenne 0, écart-type 1) | mettre les variables à la même échelle et stabiliser l'estimation |
| Retrait de la 1ʳᵉ observation de chaque jour | éviter les sauts overnight |

<p align="center">
  <img src="figures/compare_methods.png" width="85%" alt="Profil intraday de Vlo_b : agrégation par moyenne vs par somme">
</p>

### 5. Test de stationnarité

Test **ADF (Augmented Dickey-Fuller)** sur chaque série transformée : **toutes les séries sont stationnaires** (p-value < 0,05) → le VAR est applicable.

### 6. Sélection de l'ordre p

Comparaison des critères **AIC / BIC / HQIC** pour p = 1…10 → **p = 3** retenu.

### 7. Estimation du VAR(3)

$$
Y_n = c + A_1 Y_{n-1} + A_2 Y_{n-2} + A_3 Y_{n-3} + \varepsilon_n
$$

Estimation par `statsmodels.tsa.vector_ar.VAR`, visualisation des matrices $A_1, A_2, A_3$ en heatmaps.

### 8. Diagnostic du modèle

- **Stabilité** : racines de la matrice compagnon par rapport au cercle unité ;
- **Bruit blanc des résidus** : ACF des résidus, test de **Ljung-Box**, statistique de **Durbin-Watson**.

---

## Résultats clés

- **Un VAR(3) permet de capturer une partie de la dynamique jointe des flux et des prix** sur 126 jours de données tick-by-tick.
- **Variables les plus influentes : les ordres limites et les annulations côté bid** — la liquidité passive côté acheteur porte l'essentiel de l'information.
- **Symétrie limit / exécution** autour des changements de prix : les market makers ajustent simultanément offre et liquidité en réaction aux mouvements de prix.
- **Saisonnalité intraday marquée** (pic de volume à l'ouverture, ~9h35) et **volatility clustering** persistant sur les rendements ; asymétrie bid/ask des volumes.
- **Beaucoup de coefficients ne sont pas significatifs** (t-stats) : le modèle est en réalité plus parcimonieux qu'il n'y paraît.
- **Limites du modèle** : certaines racines sortent du cercle unité (instabilité) et le test de Ljung-Box rejette l'hypothèse de bruit blanc pour une partie des résidus → **signes de non-linéarité** que le VAR ne capte pas.

> **Conclusion** : le carnet d'ordres « dicte » le prix — modéliser explicitement ses flux apporte une meilleure compréhension de la microstructure, mais un cadre linéaire reste insuffisant pour en décrire toute la dynamique.

---

## Visualisations du carnet d'ordres

Le notebook exploratoire reproduit aussi plusieurs **faits stylisés** de la microstructure (autocorrélation du signe des rendements, queues épaisses, profil intraday, etc.) et propose des outils de visualisation du LOB :

| Snapshot du carnet (5 niveaux) | Depth charts au fil de la journée |
|:---:|:---:|
| <img src="figures/lob_snapshot.png" width="420"> | <img src="figures/lob_depth_charts.png" width="420"> |

<p align="center">
  <img src="figures/fact1_long_memory.png" width="60%" alt="ACF du signe des rendements">
</p>

---

## Structure du dépôt

```
market-dynamics/
├── main.ipynb              # Pipeline complet : classification des ordres → variables → VAR(p) → diagnostics
├── main_executed.ipynb     # Analyse exploratoire : faits stylisés + visualisations du LOB (avec sorties)
├── Market_Dynamics.pdf     # Présentation du projet (22 slides)
├── figures/                # Graphiques exportés
└── csv/                    # Données brutes (non versionnées)
```

---

## Installation et exécution

```bash
git clone <url-du-depot>
cd market-dynamics
pip install pandas numpy matplotlib scipy statsmodels scikit-learn jupyter
```

1. Placer les deux fichiers de données dans le dossier pointé par `DATA_PATH` (par défaut `../Data MDS/`) ;
2. lancer `jupyter notebook main.ipynb` et exécuter les cellules dans l'ordre.

> Le chargement des 126 jours de carnet nécessite plusieurs Go de RAM ; les étapes ADF / sélection du lag / estimation travaillent sur un échantillon (5 000 à 20 000 observations) pour garder des temps de calcul raisonnables.

---

## Limites et pistes d'amélioration

- **Non-linéarités** : tester des modèles de type **processus de Hawkes multivariés**, VAR à seuil / à changement de régime, ou modèles à base de réseaux de neurones (DeepLOB).
- **Instabilité** : ré-estimer sur fenêtres glissantes, réduire la dimension (VAR parcimonieux, LASSO-VAR) pour ne conserver que les coefficients significatifs.
- **Analyse causale** : fonctions de réponse impulsionnelle (IRF), décomposition de variance et tests de causalité de Granger entre flux et rendements.
- **Échantillon** : estimer sur l'ensemble des 151 125 observations et étendre à d'autres actifs / périodes (les trades couvrent 2014–2017).

---

## Compétences mobilisées

- **Microstructure de marché** : limit order book, classification d'ordres, règle de Lee-Ready / tick test, faits stylisés
- **Économétrie des séries temporelles** : stationnarité (ADF), VAR(p), critères d'information, diagnostics (stabilité, Ljung-Box, Durbin-Watson)
- **Data engineering haute fréquence** : traitement par chunks de millions de lignes, vectorisation NumPy (de plusieurs minutes à ~0,2 s par jour)
- **Python** : pandas, NumPy, SciPy, statsmodels, scikit-learn, Matplotlib

---

## Équipe

Projet réalisé par **Joseph Lemoine, Nathan Rocco, Maxime Boulin, Ivan Djeuchou et Mathieu Rémi**.
