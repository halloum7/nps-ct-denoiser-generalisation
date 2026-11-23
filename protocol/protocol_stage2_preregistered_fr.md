# Article 9 — Protocole expérimental Stage 2 (pré-enregistré)

2026-11-14 · Kamal Halloum

## Statut et acquis du Stage 1

Le Stage 1 est validé : un filtre radial linéaire appris en validation croisée par patient (LOPO) reproduit la texture du bruit B30 à partir de D45 chez 10/10 patients, aux deux épaisseurs. Ces résultats ont été exécutés sur Kaggle (version 2, run 369518568) et recontrôlés indépendamment.

| Mesure (médiane sur 10 patients) | 1 mm | 3 mm |
| --- | --- | --- |
| Variance du bruit émulé / réel, estimateur ROI [IC 95 %] | 0,995 [0,978 ; 1,022] | 1,005 [0,979 ; 1,022] |
| W1 texture émulé vs B30 / D45 vs B30 | 0,0034 / 0,118 | 0,0033 / 0,119 |
| Erreur image QD émulé / non filtré (× σ bruit B30) | 0,21 / 1,97 | 0,24 / 2,00 |
| f_mean NPS B30 / D45 (cycles/mm) | 0,324 / 0,488 | 0,324 / 0,488 |

Faits établis à reprendre dans l'article :

- Taille de pixel 0,664 à 0,820 mm selon le patient : toute analyse fréquentielle se fait en cycles/mm, par patient.
- 1,8 à 3,4 % des pixels sont écrêtés à −1024 HU (air, poumon) : l'estimation sur image entière est biaisée ; l'estimateur sur ROI de tissu ne l'est pas.
- Les champs dose des en-têtes QD (mA, mAs, CTDIvol) sont copiés du FD : inutilisables comme mesure de dose.
- 2 scanners (AS+ ×8, Flash ×2), 2 tensions (100 kVp ×5, 120 kVp ×5).
- Le décalage FD/QD mesuré (≤ 0,04 px) est sous la limite de détection de la méthode (0,08 px).

## Résultats du Stage 2a (Kaggle, version 3, run 369528906)

La famille de kernels est validée : les trois contrôles pré-enregistrés réussissent. H5 échoue en revanche : l'estimateur par coupes adjacentes n'est pas assez fidèle.

| Contrôle | Résultat | Critère | Verdict |
| --- | --- | --- | --- |
| 1 – réalisme du filtre hybride (LOPO, α = 1) | C1, C2, C3 : 10/10 ; variance émulé / réel médiane 0,998 (0,952–1,096) | ≥ 9/10 | Réussi |
| 2 – monotonie de W1 en α | 10/10 patients strictement croissants | 10/10 | Réussi |
| 3 – kernels hors famille distincts | T_h(f/0,8) : 0,053 ; Butterworth (coupure 0,531 /mm) : 0,0129 | ≥ 0,01 | Réussi |
| H5 – W1 estimé vs référence | CCC 0,70 [0,62 ; 0,79] ; biais +0,024 | CCC ≥ 0,90 et borne basse ≥ 0,80 | Échec |

Points à reporter dans l'article :

- Le vrai B30 est à W1 = 0,0016 de α = 1 : sa texture est quasi identique au B30 émulé.
- Le Butterworth a la même fréquence moyenne que α = 0,5 (0,374 contre 0,374 cycle/mm) mais une forme différente, comme prévu par l'amendement A2.
- W1 et f_mean sont calculés sur la bande commune 0,01–0,59 cycle/mm (Nyquist du plus grand pixel). Les f_mean ne sont donc pas comparables à ceux du Stage 1 (bande propre à chaque patient) ; les deux définitions seront nommées.
- L'espacement des coupes « 1 mm » est de 0,8 mm : les coupes adjacentes se chevauchent. Cela explique probablement l'échec de H5 (bruit corrélé entre coupes, plus le changement d'anatomie). C'est une hypothèse, non vérifiée.
- Cache d'entraînement : 2 970 coupes (une sur deux), en HU entiers exacts, avec empreinte SHA-256 par fichier.

Décision : le critère de H5 n'est pas modifié et l'échec sera rapporté tel quel. Toute variante de l'estimateur (coupes espacées de 1,6 mm, bande de fréquence restreinte) sera marquée « post hoc ». H1, l'hypothèse principale, n'est pas affectée.

## Résultats du pilote Stage 2b (Kaggle, GPU T4, 8 h 47)

N est fixé à 20 000 itérations pour les deux architectures, par la règle pré-enregistrée : c'est le premier multiple de 10 000 après lequel la RMSE de validation moyenne (3 graines) gagne moins de 0,5 % sur les 10 000 itérations suivantes.

| Mesure (pli 1, α = 1, β = 1) | REDCNN | DnCNN |
| --- | --- | --- |
| RMSE validation moyenne à 10k / 20k / 30k / 40k (HU) | 30,68 / 30,26 / 30,24 / 30,16 | 31,33 / 30,86 / 30,83 / 30,68 |
| Gain 10k→20k / 20k→30k / 30k→40k | 1,34 % / 0,10 % / 0,23 % | 1,48 % / 0,12 % / 0,47 % |
| Meilleure RMSE validation par graine (HU), écart-type | 30,15 ; 30,13 ; 30,10 — 0,027 | 30,63 ; 30,62 ; 30,65 — 0,013 |
| RMSE test L067 / L291, B30 émulé (moyenne 3 graines) | 28,97 / 29,57 | 29,40 / 30,20 |
| Rapport vrai B30 / B30 émulé (L067 ; L291) | 0,999–1,000 ; 0,987–0,989 | 0,999–1,001 ; 0,985–0,988 |
| Temps par itération | 0,124 s | 0,133 s |

Lecture :

- Tous les tests automatiques ont réussi (filtre GPU = numpy à 0,0007 HU ; métrique exacte ; sur-apprentissage d'un lot).
- La variabilité due à la graine est très faible (≤ 0,1 % en validation, ≤ 0,4 % en test) : un seul entraînement par condition suffit, comme prévu.
- Après 20 000 itérations, le gain restant observé avec le calendrier de 40 000 est de 0,3 à 0,6 %. C'est un plancher accepté par la règle ; il est identique pour toutes les conditions et ne biaise donc pas les comparaisons.
- V0, première lecture sur 2 patients : écart vrai/émulé ≤ 1,5 %, dans la marge de ± 5 %. Pour L291, l'entrée émulée est déjà 2,3 % plus bruitée que la vraie : l'écart vient de l'émulation, pas du débruiteur. Le test formel se fera sur les 10 patients.
- Budget : 0,124 s par itération donne 41 min par entraînement, soit environ 55 h pour les 80 runs. Le Stage 2c utilise donc un pipeline accéléré (coupes filtrées calculées une fois, 2 GPU), vérifié identique au pilote avant tout entraînement.

Amendement A3 (15/11/2026, avant tout entraînement du Stage 2c) : précision mixte pour l'entraînement. Le pipeline accéléré produit des données identiques au pilote (écart 0,0 HU), mais la vitesse est restée de 0,124 s par itération : REDCNN est limité par le calcul en float32 sur T4 (environ 1 TFLOP par itération). Mesure sur T4 : la précision mixte n'accélère que DnCNN (0,122 → 0,063 s/it), pas REDCNN (0,102 → 0,104 s/it). Règle retenue : la précision mixte (float16 et mise à l'échelle dynamique de la perte) n'est utilisée pour une architecture que si elle est au moins 20 % plus rapide que float32 ; l'évaluation reste toujours en float32. cuDNN choisit ses algorithmes les plus rapides (tailles d'entrée fixes). Contrôle pré-spécifié : la même tâche (REDCNN, pli 1, α = 1, β = 1, même graine) est entraînée en float32 et en float16. La précision mixte n'est adoptée que si l'écart de RMSE de validation est ≤ 0,3 % (écart-type entre graines au pilote : 0,09 %). Sinon, arrêt.

## Question scientifique, lacune et hypothèses

Question : la distance de texture du bruit (W1 entre NPS radiaux normalisés) entre données d'entraînement et de déploiement prédit-elle, de façon quantitative, la perte de performance d'un débruiteur CT par apprentissage profond ?

Lacune : Zeng et al. (Med Phys 2022) montrent que REDCNN généralise mal d'un kernel à l'autre et formulent, en conclusion, l'hypothèse que la généralisation pourrait être fortement corrélée à la similarité des NPS. Leur protocole compare deux kernels réels ; il ne permet pas d'estimer une relation dose–réponse. Les travaux récents (NADD, ENCORE) conditionnent le réseau sur le contexte de bruit ; ils ne testent pas la valeur prédictive d'une mesure de texture pour un modèle fixe.

Contribution visée : (1) premier test contrôlé de cette hypothèse sur une famille continue de kernels émulés et validés physiquement ; (2) séparation de l'effet texture et de l'effet niveau de bruit ; (3) un contrôle pré-déploiement calculable sur les seules images du site cible. La formulation « premier » sera revérifiée par une recherche bibliographique systématique avant la rédaction.

| Code | Hypothèse | Statut |
| --- | --- | --- |
| H1 | La perte de généralisation augmente de façon monotone avec W1(train, test) | Primaire |
| H2 | La texture (W1) explique la perte au-delà de l'écart de niveau de bruit | Secondaire |
| H3 | La perte est asymétrique : entraîner sur une texture plus fine généralise mieux vers une texture plus lisse que l'inverse | Secondaire |
| H4 | Un modèle W1 → erreur ajusté sur la famille émulée prédit l'erreur sur le vrai B30 et sur des kernels hors famille | Secondaire |
| H5 | Le W1 estimé sans paires FD/QD (différence de coupes adjacentes) concorde avec le W1 de référence | Secondaire, condition du contrôle de déploiement |

Validation préalable (V0) : un débruiteur entraîné sur le B30 émulé doit avoir la même erreur sur le vrai B30 que sur le B30 émulé (test d'équivalence). Si V0 échoue, la famille émulée n'est pas un substitut valide et H1–H4 ne sont pas interprétées.

## Données, découpage patient et contrôle des fuites

Données : Mayo 2016 via Kaggle (andrewmvd/ct-low-dose-reconstruction), coupes de 1 mm, DICOM originaux uniquement. Apprentissage et émulation en D45 (FD et QD) ; le vrai B30 ne sert qu'au test (V0, H4). Le 3 mm est réservé à une analyse de sensibilité.

Validation croisée à 5 plis par patient, fixée avant tout entraînement (tirage avec graine 2026, contraintes : un patient 100 kVp et un 120 kVp par pli ; les deux patients Flash dans des plis différents).

| Pli | Patients test | Patient validation (arrêt, sélection) | Patients entraînement |
| --- | --- | --- | --- |
| 1 | L067, L291 | L143 | les 7 autres |
| 2 | L109, L143 | L310 | les 7 autres |
| 3 | L192, L310 | L096 | les 7 autres |
| 4 | L333, L096 | L286 | les 7 autres |
| 5 | L506, L286 | L291 | les 7 autres |

Règle de validation : patient 120 kVp du pli suivant (cyclique). Échantillonnage : une coupe sur deux pour l'entraînement ; 30 coupes réparties pour la validation ; 50 coupes réparties (hors 2 coupes d'extrémité) par patient test.

Contrôles de fuite :

- Aucun patient test ou validation ne contribue à l'entraînement du débruiteur du même pli.
- Les hyperparamètres sont fixés a priori (section Débruiteurs) ; aucun réglage sur les patients test.
- Le filtre de la famille de kernels est un opérateur physique sans étiquette de débruitage ; il est appris une fois sur les 10 patients. Analyse de sensibilité : refaire l'émulation avec un filtre appris sur les seuls patients d'entraînement du pli 1.
- Les patients du pli test ne sont jamais utilisés pour l'arrêt précoce ni le choix du checkpoint.

## Famille de kernels émulés et niveaux de bruit

La famille va de D45 (α = 0) à B30 émulé (α = 1) par puissance du filtre hybride, appliqué en fréquence physique avec la taille de pixel de chaque patient :

```latex
T_\alpha(f) = \max\left(T_h(f),\,10^{-3}\right)^{\alpha}, \qquad T_h(f) = \bigl(1-w(f)\bigr)\,T_{\mathrm{img}}(f) + w(f)\,T_{\mathrm{roi}}(f), \qquad w(f) = \frac{1}{1+e^{-(f-0.06)/0.015}}
```

Justification du filtre hybride : l'estimateur ROI est exact mais sa résolution fréquentielle est limitée aux basses fréquences (≈ 2/(taille ROI × pixel)) ; l'estimateur image couvre ces basses fréquences mais il est biaisé par l'écrêtage au-delà. Amendement A1 (14/11/2026, avant tout run réel) : transition déplacée de 0,15 à 0,06 cycle/mm (largeur 0,015), car le test synthétique a montré un biais de l'estimateur image entre 0,05 et 0,15 cycle/mm ; erreur maximale du filtre hybride réduite de 0,031 à 0,015.

Entrée et cible du débruiteur, avec β le facteur de niveau de bruit (variance du bruit inséré ∝ β²) :

```latex
x_{\alpha,\beta} = K_\alpha\!\left[\mathrm{FD} + \beta\,(\mathrm{QD}-\mathrm{FD})\right], \qquad y_\alpha = K_\alpha[\mathrm{FD}]
```

β n'est pas une dose au sens strict : le bruit propre du FD n'est pas réduit. Il sera nommé « niveau de bruit » dans l'article.

| Ensemble | α | β | Nombre |
| --- | --- | --- | --- |
| Entraînement, série texture (H1) | 0 ; 0,25 ; 0,5 ; 0,75 ; 1 | 1 | 5 |
| Entraînement, série niveau de bruit (H2) | 0 ; 0,5 ; 1 | 0,5 | 3 |
| Grille de test sur la famille | 0 ; 0,25 ; 0,5 ; 0,75 ; 1 | 0,5 ; 0,75 ; 1 | 15 |
| Vrai B30 (V0, H4) | — | 0,5 ; 1 | 2 |
| Hors famille (H4) : T_h(f/0,8) à α = 1 ; Butterworth d'ordre 8 sur D45, coupure fixée pour une f_mean égale à celle de α = 0,5 | — | 0,5 ; 1 | 4 |

Contrôles obligatoires en Stage 2a, sans GPU, avant tout entraînement :

1. À α = 1, T_h doit satisfaire C1, C2 et C3 du Stage 1 chez au moins 9/10 patients (LOPO).
2. W1(α, 0) doit croître de façon strictement monotone avec α chez chaque patient.
3. Les kernels hors famille doivent avoir un NPS distinct de la famille : W1 au point le plus proche de la famille ≥ 0,01 cycle/mm, sinon ils ne testent rien de nouveau. Amendement A2 (14/11/2026, avant tout run réel) : le kernel T_h(f/1,25) a été retiré, car le test synthétique l'a placé sur le chemin de la famille (W1 = 0,0035). Il est remplacé par un Butterworth de même f_mean que α = 0,5 mais de forme différente, qui teste si W1 apporte plus qu'une fréquence moyenne.

## Débruiteurs et entraînement

Deux architectures primaires, choisies pour leur statut de référence et non pour leur performance : REDCNN (le modèle étudié par Zeng et al.) et DnCNN (apprentissage résiduel avec normalisation par lot). Un U-Net léger reste optionnel, seulement si le budget mesuré au pilote le permet ; il ne change pas les hypothèses.

| Élément | REDCNN | DnCNN |
| --- | --- | --- |
| Structure | 5 conv + 5 déconv symétriques, 96 filtres 5×5, pas 1, sans padding, raccourcis entre couches appariées | 17 couches, 64 filtres 3×3, BatchNorm, sortie = résidu |
| Raccourcis | 3 (choix de l'implémentation de référence courante ; l'article ne donne pas le nombre) | 1 résidu global |
| Source | Chen et al., IEEE TMI 2017 | Zhang et al., IEEE TIP 2017 |

Entraînement identique pour toutes les conditions, fixé a priori :

- Perte MSE ; Adam ; taux 10⁻⁴ décroissant en cosinus jusqu'à 10⁻⁵ ; lot de 32 patchs 64×64.
- Nombre d'itérations N identique partout, fixé après le pilote (convergence de la perte de validation) puis gelé.
- Checkpoint retenu = meilleure RMSE sur le patient de validation, évaluée toutes les 2 000 itérations.
- Normalisation : HU / 1000. Centres de patch tirés dans le masque du corps (non érodé).
- Augmentation : retournements et rotations de 90° uniquement. Pas de changement d'échelle ni de rotation libre : ils modifieraient le NPS, donc la variable étudiée. (C'est une différence assumée avec l'article REDCNN, qui utilisait un facteur d'échelle.)
- Émulation à la volée sur GPU : filtre K_α appliqué à la coupe entière 512×512 (FFT, taille de pixel du patient), puis découpe des patchs. Aucun fichier filtré n'est stocké.
- Graine fixée par (architecture, pli, condition). Variabilité due à l'entraînement : 3 graines sur une condition du pli 1 au pilote, pour la comparer aux effets mesurés.

## Métriques et critères d'évaluation

Critère principal : la perte de généralisation relative L, par patient test, d'un modèle entraîné en condition c et appliqué en condition c′, rapportée au modèle apparié (entraîné en c′) :

```latex
L_p(c \to c') = \frac{\mathrm{RMSE}_p\bigl(f_c(x_{c'}),\,y_{c'}\bigr)}{\mathrm{RMSE}_p\bigl(f_{c'}(x_{c'}),\,y_{c'}\bigr)} - 1
```

L est défini sur les 8 conditions d'entraînement, soit 56 paires non appariées par architecture et par patient ; le test principal H1 n'utilise que les 20 paires de la série texture (β = 1), où seule la texture change. RMSE en HU, calculée sur les 50 coupes test regroupées, dans le masque de tissu (corps sans poumon ni air, érodé de 4 px).

Variables explicatives, calculées avant tout entraînement :

- W1(c, c′) : distance de texture entre le NPS normalisé du bruit d'entraînement (regroupé sur les patients d'entraînement du pli) et celui du patient test. W1 ne dépend pas de β, qui ne change que l'amplitude.
- Δlogvar(c, c′) = log(variance du bruit test / variance du bruit d'entraînement) : écart de niveau de bruit.
- Signe de Δα : direction du décalage (vers plus fin ou plus lisse), pour H3.

Métriques secondaires, par patient :

- RMSE absolue (HU) sur toute la grille de test (15 + 6 conditions), y compris sans modèle apparié.
- SSIM, fenêtre [−160 ; 240] HU, plage 400 HU.
- Bruit résiduel : variance et W1 du NPS de (sortie − cible) dans des ROI de tissu, pour voir quelle bande de fréquence n'est pas traitée.
- Pour H5 : W1 estimé à partir de la seule entrée (différence de coupes adjacentes de 1 mm, divisée par √2), comparé au W1 de référence (QD − FD).

Non mesuré : la résolution spatiale (MTF) et la détectabilité à bas contraste. Il faudrait un fantôme, absent du jeu Kaggle ; c'est une limite déclarée.

## Plan d'analyse statistique pré-enregistré

L'unité statistique est le patient (n = 10, chacun testé une fois par la validation croisée). Les coupes ne sont jamais traitées comme des observations indépendantes.

| Test | Analyse | Critère de succès fixé d'avance |
| --- | --- | --- |
| V0 | TOST sur log(RMSE vrai B30 / RMSE B30 émulé), modèle α = 1, β = 1, t de Student apparié | IC 90 % du rapport inclus dans [0,95 ; 1,05] |
| H1 (primaire) | ρ de Spearman entre W1 et L sur les 20 paires non appariées à β = 1 (4 niveaux d'écart de texture), par patient ; Wilcoxon sur les 10 ρ ; Holm sur 2 architectures | p ajusté < 0,05 et ρ médian ≥ 0,5 pour chaque architecture |
| H2 | Modèle mixte L ~ W1 + Δlogvar + signe(Δα) + architecture + (1 \| patient) ; IC par bootstrap sur les patients (B = 10 000) | IC 95 % du coefficient W1 excluant 0 avec Δlogvar dans le modèle |
| H3 | Paires symétriques à β égal : Wilcoxon apparié L(fin → lisse) vs L(lisse → fin) | p < 0,05, sens annoncé par H3 |
| H4 | Régression log(RMSE) ~ W1 + Δlogvar ajustée sur la famille, prédiction sur vrai B30 et hors famille | Erreur absolue relative médiane ≤ 10 % et inférieure à celle d'un prédicteur naïf (RMSE du modèle apparié le plus proche en α) |
| H5 | CCC de Lin et Bland–Altman entre W1 estimé et W1 de référence (patients × conditions) | CCC ≥ 0,90 et borne inférieure de l'IC 95 % ≥ 0,80 |

Règles d'interprétation :

- Seule H1 est confirmatoire. H2 à H5 sont rapportées avec effets et IC, sans conclusion confirmatoire.
- Si V0 échoue, les résultats sur la famille émulée sont présentés comme exploratoires et l'article change de cadre.
- Un résultat négatif sur H1 est publiable tel quel ; aucun critère ne sera modifié après avoir vu les résultats.
- Limite de puissance déclarée : avec 10 patients, le plus petit p possible du Wilcoxon bilatéral est 0,002 ; seuls des effets cohérents chez presque tous les patients seront détectables.
- Toute analyse ajoutée après ce document sera marquée « post hoc » dans l'article.

## Budget de calcul, étapes et points d'arrêt

Le plan complet compte 80 entraînements : 2 architectures × 5 plis × 8 conditions. Le temps par entraînement n'est pas encore connu : il sera mesuré au pilote avant de lancer le reste. Le quota GPU hebdomadaire et la durée maximale d'une session Kaggle seront relevés sur le compte avant le pilote.

1. **Stage 2a, sans GPU.** Construire T_h ; contrôles 1 à 3 de la section famille ; matrices W1 et Δlogvar pour toutes les paires ; données de H5 ; cache des coupes D45 FD/QD (float16, par patient) publié en sortie de notebook privée. Arrêt si un contrôle échoue : correction documentée, puis nouvelle exécution.
2. **Stage 2b, pilote GPU.** Pli 1, REDCNN et DnCNN, condition α = 1, β = 1, 3 graines. Mesure du temps par itération, choix et gel de N, variabilité entre graines, première lecture de V0 sur 2 patients. Arrêt si l'entraînement diverge ou si la variabilité entre graines dépasse la moitié de l'effet attendu le plus faible.
3. **Stage 2c, entraînement complet.** Les 80 runs, plusieurs conditions par session, chaque modèle sauvegardé avec sa graine et son hachage de configuration. Si le budget est dépassé, ordre de réduction fixé d'avance : supprimer le U-Net optionnel, puis espacer l'évaluation de validation, puis retirer la condition α = 0,5, β = 0,5 (H2 restera estimable avec α = 0 et 1). On ne supprime jamais un pli ni une condition de la série texture nécessaire à H1.
4. **Stage 2d, évaluation et statistiques.** Inférence sur toute la grille de test, calcul des métriques, analyses dans l'ordre V0 puis H1 à H5, figures.

Chaque étape produit un fichier de résultats horodaté et recontrôlé indépendamment avant de passer à la suivante.

## Limites connues et références

Limites à déclarer dans l'article :

- Un seul constructeur (Siemens), 10 patients, un seul centre ; deux kernels réels seulement, le reste étant émulé.
- La famille ne peut pas aller au-delà de D45 vers des kernels plus durs (problème inverse mal posé).
- β change le niveau de bruit inséré, pas la dose au sens strict ; la cible FD contient elle-même du bruit.
- Débruitage 2D coupe par coupe ; le W1 radial ignore l'anisotropie et la non-stationnarité du bruit.
- Pas de MTF ni de détectabilité à bas contraste (pas de fantôme dans le jeu de données).
- Faible effectif : seuls des effets cohérents entre patients sont détectables.

Références vérifiées (DOI contrôlés le 14/11/2026) :

1. Zeng R, et al. Performance of a deep learning-based CT image denoising method: generalizability over dose, reconstruction kernel, and slice thickness. Med Phys. 2022;49(2):836–853. [doi:10.1002/mp.15430](https://doi.org/10.1002/mp.15430)
2. McCollough CH, et al. Low-dose CT for the detection and classification of metastatic liver lesions: results of the 2016 Low Dose CT Grand Challenge. Med Phys. 2017;44(10):e339–e352. [doi:10.1002/mp.12345](https://doi.org/10.1002/mp.12345)
3. Chen H, et al. Low-dose CT with a residual encoder-decoder convolutional neural network. IEEE Trans Med Imaging. 2017;36(12):2524–2535. [doi:10.1109/TMI.2017.2715284](https://doi.org/10.1109/TMI.2017.2715284)
4. Zhang K, et al. Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising. IEEE Trans Image Process. 2017;26(7):3142–3155. [doi:10.1109/TIP.2017.2662206](https://doi.org/10.1109/TIP.2017.2662206)
5. Samei E, et al. Performance evaluation of computed tomography systems: summary of AAPM Task Group 233. Med Phys. 2019;46(11):e735–e756. [doi:10.1002/mp.13763](https://doi.org/10.1002/mp.13763)
6. Kristof G, Eulig E, Kachelrieß M. Noise-augmented deep denoising: a method to boost CT image denoising networks. Med Phys. 2025;52(10):e18121. [doi:10.1002/mp.18121](https://doi.org/10.1002/mp.18121)
7. Eulig E, Ommer B, Kachelrieß M. Benchmarking deep learning-based low-dose CT image denoising algorithms. Med Phys. 2024;51(12):8776–8788. [doi:10.1002/mp.17379](https://doi.org/10.1002/mp.17379)
8. Efficient noise context-aware representation for low-dose CT denoising (ENCORE). Prépublication arXiv:2608.10343, non évaluée par les pairs. [arXiv](https://arxiv.org/html/2608.10343)
9. Han et al. Med Phys 2024. [doi:10.1002/mp.16963](https://doi.org/10.1002/mp.16963) — titre complet à reporter lors de la rédaction.
10. Jeu de données : [andrewmvd/ct-low-dose-reconstruction sur Kaggle](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction).
