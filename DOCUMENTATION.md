# Documentation – LBM-PyOpenCL

Simulation d'écoulements 2D par la méthode de Boltzmann sur réseau (LBM, *Lattice Boltzmann Method*), modèle **D2Q9** avec collision **MRT** (*Multiple Relaxation Time*). Le même modèle est implémenté de trois façons (OpenCL sur GPU ou CPU, NumPy, Numba) et se pilote depuis une page web.

Sommaire : [1. Vue d'ensemble](#1-vue-densemble) · [2. Lancer](#2-lancer-et-installer) · [3. Guide de lecture du code](#3-guide-de-lecture-du-code) · [4. La méthode LBM en bref](#4-la-méthode-lbm-en-bref) · [5. Détail des fichiers](#5-détail-des-fichiers) · [6. Protocole client/serveur](#6-protocole-clientserveur) · [7. Tests](#7-tests) · [8. Performances et dépannage](#8-performances-et-dépannage)

---

## 1. Vue d'ensemble

### Ce que fait le programme
On simule un fluide qui s'écoule de gauche à droite dans une boîte 2D, autour d'un obstacle (plaque, cylindre, profil NACA…). Le fluide est décrit sur une grille de `nx × ny` cellules ; à chaque pas de temps, chaque cellule échange des « populations » de particules avec ses voisines (voir section 4). On regarde la norme de la vitesse, le rotationnel, et un colorant passif qui visualise les lignes de courant.

### Les trois solveurs (même modèle, même résultat)

| Solveur | Fichier | Tourne sur | Colorant | Vitesse (512×256, Ryzen 9 5900X / RTX 4080) |
|---|---|---|---|---|
| **OpenCL** | `lbm_opencl.py` + `kernels.cl` | GPU, ou CPU via un runtime OpenCL | oui | ~97 MLUPS (GPU, colorant et rendu compris) |
| **Python NumPy** | `lbm_native.py` | CPU (1 cœur + BLAS) | oui | ~17 MLUPS (~1,3 avec colorant) |
| **Python Numba** | `lbm_numba.py` | CPU, tous les threads | oui | ~160 MLUPS (~70 avec colorant) |

*MLUPS = millions de cellules mises à jour par seconde.* Le solveur NumPy sert de **référence lisible** : les deux autres sont vérifiés contre lui par les tests (section 7).

### Architecture

```
 Navigateur  (web/index.html)                       Serveur Python
 ┌──────────────────────────┐   WebSocket /ws     ┌───────────────────────────────────────────┐
 │ panneau de configuration │ ── start / set ───▶ │ app.py (FastAPI)                          │
 │ canvas (image JPEG)      │ ◀── JPEG + stats ── │   └─ SimulationSession  (simulation_session.py)
 │ FPS, solveur, MLUPS      │ ── ack ───────────▶ │        thread : boucle de calcul          │
 └──────────────────────────┘                     │        └─ un « backend » au choix :        │
                                                  │           OpenCLBackend ─▶ lbm_opencl.py ─▶ ocl_driver.py ─▶ kernels.cl
                                                  │           NativeBackend ─▶ lbm_native.py   │
                                                  │           NumbaBackend  ─▶ lbm_numba.py    │
                                                  │        obstacles : tools.py                │
                                                  └───────────────────────────────────────────┘
```

### Ce qui se passe pour une image affichée
1. La page envoie `start` avec la configuration (solveur, grille, Reynolds, obstacle…).
2. `app.py` crée une `SimulationSession` qui démarre un thread.
3. Le thread construit le backend choisi, calcule l'obstacle (`tools.ObsertacleManager`) et le donne au solveur.
4. Boucle : *appliquer les changements en direct* → *faire `iter_per_frame` pas de temps* → si une image est demandée, *la rendre* (tableau RGB) → *l'encoder en JPEG*.
5. `app.py` envoie le JPEG à la page dès qu'elle a acquitté l'image précédente (`ack`), et les statistiques toutes les 0,5 s.

Le client n'a **aucune connaissance du solveur** : il reçoit des images et envoie des paramètres. C'est ce qui permettra de brancher plus tard un client Unity sur le même protocole (section 6).

---

## 2. Lancer et installer

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py            # ouvre http://localhost:8000
```

- Il faut un pilote OpenCL pour le solveur OpenCL (GPU NVIDIA/AMD/Intel). Testé sur NVIDIA RTX 4080.
- **Device CPU pour OpenCL** : `pip install intel-opencl-rt` (déjà dans `requirements.txt`) ajoute le runtime CPU d'Intel, qui marche aussi sur AMD. La roue pip de PyOpenCL ne lit que le registre Windows (droits administrateur) ; `ocl_driver.py` déclare donc `intelocl64.dll` via la variable `OCL_ICD_FILENAMES` avant d'importer PyOpenCL. Les devices qui ne savent pas exécuter un kernel sont masqués automatiquement.
- Les solveurs Python n'ont besoin d'aucun pilote.
- Test rapide sans navigateur : `python simulation_session.py [opencl|native|numba]` (8 s de simulation, écrit `smoke_test.jpg`).

### Utiliser l'interface
Le panneau du haut règle la simulation, l'image s'affiche en dessous, la barre d'état donne l'état, les **FPS**, les **MLUPS**, le **solveur** (`GPU : <nom>`, `CPU : <nom> sur N threads`…), la grille et l'itération.
- **En direct** : Reynolds, pas par image, affichage (norme / rotationnel), échelles de couleur, obstacle (profil, position, rotation, taille).
- **Au démarrage** (re-cliquer sur « Démarrer ») : solveur, device, threads CPU, grille, speedup, colorant.
- Molette = zoom, glisser = déplacer, double-clic = recentrer.
- **Réinitialiser** remet tous les réglages et la vue à leurs valeurs par défaut (celles de `DEFAULT_CONFIG`, lues via `/api/config`) ; si une simulation tourne, elle est relancée de zéro avec ces valeurs.

---

## 3. Guide de lecture du code

Ordre conseillé pour comprendre le projet :

1. **Section 4 de ce document** (la méthode LBM) : sans elle, les noms `fin`, `fout`, `moments`… restent obscurs.
2. **`lbm_native.py`** (~140 lignes) : tout le modèle en NumPy, de manière lisible. Chaque méthode correspond à une étape du pas de temps : `_bcs`, `_inlet`, `_collision`, `_streaming`.
3. **`simulation_session.py`** : comment un solveur est piloté (backends, thread, paramètres en direct, JPEG).
4. **`app.py`** puis **`web/index.html`** : le serveur et la page (courts).
5. **`lbm_opencl.py` + `kernels.cl`** : la même chose sur GPU. Les kernels portent les mêmes noms que les étapes de `lbm_native.py` (`bcs_sym`, `bcs_inlet`, `moments`, `streaming`). L'en-tête de `kernels.cl` donne les conventions de stockage.
6. **`lbm_numba.py`** : `lbm_native.py` réécrit en boucles par cellule.
7. **`ocl_driver.py`** et **`tools.py`** : utilitaires.

### Correspondance entre les trois implémentations

| Étape du pas de temps | `lbm_native.py` | `lbm_numba.py` | `kernels.cl` |
|---|---|---|---|
| Bords haut/bas/sortie | `_bcs` | `_bcs` | `bcs_sym` |
| Entrée (vitesse imposée) | `_inlet` | `_inlet` | `bcs_inlet` |
| Collision MRT + rebond sur obstacle | `_collision` | `_collision` | `moments` |
| Propagation | `_streaming` | `_streaming` | `streaming` |
| Image de la vitesse | `tools.render_field` | idem | `norm_uv_to_texture`, `curl_to_texture` |

### Conventions communes (à connaître)
- **Indexation `[x, y]`**, `x` vers la droite (indice 0 = entrée), `y` vers le haut. Le tableau d'une grandeur a pour forme `(nx, ny)` ; les populations `(9, nx, ny)`. Dans OpenCL, l'indice plat d'une cellule est `ix * ny + iy`.
- **Ordre des 9 directions D2Q9** : `0` repos, `1` +x, `2` +y, `3` −x, `4` −y, `5` (+1,+1), `6` (−1,+1), `7` (−1,−1), `8` (+1,−1). La direction opposée de `i` est `OPPOSITE[i] = [0,3,4,1,2,7,8,5,6]`.
- **Unités réseau** : une cellule = 1 de longueur, un pas de temps = 1 de temps. `uLB = 0.04·speedup`, `nu = uLB·l_ref/Re` avec `l_ref = ny/2`, `omega = 1/(3·nu + 0.5)`.
- **float32** partout : les solveurs concordent à ~1e-6 près, pas au bit près.
- L'image rendue a sa première ligne en haut (l'axe `y` est retourné au rendu).

---

## 4. La méthode LBM en bref

Au lieu de résoudre directement les équations de Navier-Stokes, la LBM suit des **populations de particules fictives** `f_i` (i = 0..8) sur une grille. Chaque population se déplace d'une cellule dans une des 9 directions `c_i` du réseau **D2Q9** (repos, 4 directions axiales, 4 diagonales). Les grandeurs du fluide se déduisent par sommes :

- densité `ρ = Σ f_i` ;
- quantité de mouvement `ρ·u = Σ f_i · c_i` (donc la vitesse `u = (u, v)`).

Un pas de temps comprend deux étapes locales :

1. **Collision** : dans chaque cellule, les populations relaxent vers un état d'équilibre `f_i^eq = w_i ρ (1 + 3 c_i·u + 4.5 (c_i·u)² − 1.5 u²)` (poids `w = 4/9, 1/9, 1/36`). C'est là que la viscosité du fluide apparaît.
2. **Propagation** (*streaming*) : chaque population `f_i` passe dans la cellule voisine dans la direction `c_i`.

### Collision MRT (utilisée ici)
Au lieu de relaxer les 9 populations au même rythme (BGK), la MRT les transforme d'abord en 9 **moments** `m = M·f` : densité, énergie, flux d'énergie, quantité de mouvement `jx, jy`, contraintes `pxx, pxy`… Chaque moment relaxe vers son équilibre à son propre rythme (matrice diagonale `S`) :

`f ← f − M⁻¹ · S · (m − m_eq)`

| Moments | Taux de relaxation |
|---|---|
| ρ, jx, jy (conservés) | 0 |
| e, ε | 1.4 |
| qx, qy | 1.2 |
| pxx, pxy (fixent la viscosité) | `omega` |

La MRT est plus stable que la BGK à grand Reynolds. `M1S = M⁻¹·S` est calculée une fois (sur CPU) puis réutilisée à chaque pas ; `update_re()` la recalcule quand on change le Reynolds.

### Conditions aux limites
| Bord | Condition |
|---|---|
| Entrée (`x = 0`) | vitesse imposée `(uLB, 0)` ; densité déduite des populations connues ; populations manquantes (1, 5, 8) mises à l'équilibre |
| Sortie (`x = nx−1`) | gradient nul : copie de la colonne voisine |
| Haut et bas | périodiques : les populations manquantes viennent du bord opposé |
| Obstacle | **bounce-back** : les populations rebondissent (direction opposée) |

### Colorant
Un scalaire passif transporté par la vitesse (advection semi-lagrangienne : on remonte la trajectoire, on interpole le colorant à cet endroit). Il sert uniquement à visualiser ; il n'agit pas sur le fluide. Les trois solveurs le calculent avec le même algorithme : à chaque pas, pour chaque cellule fluide, trajectoire rétrograde par Runge-Kutta à trois étages (`k1 = -u`, point à ½·k1, puis à ¾·k2, puis combinaison 2/9·k1 + 1/3·k2 + 4/9·k3) et interpolation bicubique de Lagrange 4×4 du colorant à l'arrivée (0 si un point sort de la grille) ; puis les émetteurs réécrivent leur valeur dans les cellules où ils sont actifs. Le colorant est affiché dans le canal alpha : `alpha = 1 − colorant`, qui fond la couleur vers le fond sombre.

---

## 5. Détail des fichiers

### `simulation_session.py` — le pilote
- `DEFAULT_CONFIG` : tous les paramètres et leurs valeurs par défaut (solveur, device, grille, Reynolds, obstacle, affichage, colorant…).
- `ObstacleBuilder` : construit le masque booléen `(nx, ny)` des cellules solides depuis les paramètres `obstacle_*` (met en cache les profils chargés).
- **Backends** : trois classes avec la même interface — `set_obstacle(mask)`, `update(old_config)` (applique un changement en direct), `step(iters)`, `render()` (→ tableau RGB `uint8`, `(ny, nx, 3)`), `device_text`. `OpenCLBackend`, `NativeBackend` (NumPy), `NumbaBackend` ; `BACKENDS` les référence par nom.
- `SimulationSession` : `start()` / `stop()`, `set_params()` (changements en direct, appliqués avant le prochain lot de pas), `request_frame()` / `get_frame()`, `stats()` (FPS, MLUPS…). Le calcul tourne dans `_run`, un thread dédié.

### `app.py` — le serveur web
FastAPI. `GET /` sert la page ; `GET /api/devices` liste les devices OpenCL ; `GET /api/config` donne les défauts et la liste des profils d'obstacles ; `WS /ws` échange les images et les paramètres (section 6).

### `web/index.html` — la page
HTML/CSS/JS sans bibliothèque. Un formulaire (`readConfig` lit les champs, `sendLive` envoie les changements en direct avec un petit délai anti-rebond), un `<canvas>` où l'on dessine chaque JPEG reçu, le zoom/déplacement à la souris, la barre d'état. `onSolverChange` grise les champs inutiles selon le solveur.

### `lbm_native.py` — le modèle de référence
Classe `LBMNative` (tout en NumPy, `float32`). `run(n)` répète : `_bcs` → `_inlet` → `_collision` → `_streaming`. `get_fields()` renvoie `(u, v, rho)`. Les formules de la collision sont celles du kernel `moments` (équilibre des moments avec `j/ρ ≈ j`, approximation faible compressibilité). Le colorant s'active avec `enable_dye()`, s'alimente avec `add_dye_rect(...)` (même sémantique que `RectDye`), se lit avec `get_dye()` ; l'advection est `_advect_dye` (interpolation par `take` sur indices plats).

### `lbm_numba.py` — le modèle compilé
`LBMNumba` hérite de `LBMNative` (mêmes paramètres, même interface) et remplace le calcul par des fonctions `@njit` : `_collision` et `_streaming` sont parallélisées sur `x` avec `prange`. `warm_up()` compile (ou charge du cache `__pycache__`) avant la première vraie simulation. Le nombre de threads est global au processus (`numba.set_num_threads`).

### `lbm_opencl.py` — le modèle OpenCL
- `LBM` : paramètres physiques et état initial côté CPU (NumPy) : `omega`, matrice `M`, `M1S`, populations d'équilibre initiales, `update_re()`.
- `LBM_OCL` : copie cet état sur le device (`init_data`) et lance les kernels. `run(n)` fait `bcs_sym → inlet → moments → streaming → post_lbm`, puis `post_run`. Les méthodes `get_*` rapatrient les résultats. Les méthodes `collision()`, `equilibrium()`, `macroscopic()` (collision BGK) existent mais **ne sont pas appelées** par `run()`.
- `RectDye`, `DyeEmitter` : sources de colorant.

Disposition mémoire : les 9 populations sont découpées en `f0` `(nx, ny)`, `f14` `(nx, ny, 4)` (directions 1-4) et `f58` `(nx, ny, 4)` (directions 5-8) pour lire 4 valeurs d'un coup (`float4`). Trois jeux : `fin` (après propagation), `fout` (après collision), `feq`. `uvrho` `(nx, ny, 4)` = `(u, v, ρ, inutilisé)`.

### `kernels.cl` — les kernels
| Kernel | Rôle |
|---|---|
| `moments` | Collision MRT (+ calcul de ρ, u, v) ; rebond sur obstacle. `fin → fout`. |
| `streaming` | Propagation ; les populations venant de l'extérieur sont mises à 0. `fout → fin`. |
| `bcs_inlet` | Entrée : vitesse imposée. Un work-item par ligne. |
| `bcs_sym` | Haut/bas périodiques, sortie à gradient nul. |
| `fout`, `equilibrium`, `rho_uv` | Collision BGK, équilibre, macroscopiques (non utilisés par `run()`). |
| `norm_uv_to_texture`, `norm_u_to_buffer` | Norme de la vitesse → couleur (palette bleu → blanc → rouge), obstacles en vert. |
| `curl_to_texture`, `curl_to_buffer` | Rotationnel (signé, centré sur 0). |
| `dye_bipolyinter` | Advection du colorant (Runge-Kutta + interpolation bicubique). |
| `update_dye_emitter` | Injecte le colorant des émetteurs actifs. |
| `dye_to_texture`, `copy_texture` | Mettent le colorant dans le canal alpha de l'image (deux images distinctes : les images `__read_write` ne sont pas disponibles partout). |

Les pointeurs `__global` n'ont aucun qualificateur d'accès (`__read_only`…) : les compilateurs OpenCL récents les refusent.

### `ocl_driver.py`
- `list_devices()` : liste des devices OpenCL **utilisables** (chacun est testé avec un petit kernel), avec `id`, nom, type GPU/CPU, nombre d'unités de calcul, mémoire.
- `OCL_Driver(device_index=…)` : contexte, file de commandes (profilage activé), compilation de `kernels.cl` ; `describe()` renvoie par exemple `GPU : NVIDIA GeForce RTX 4080 (76 compute units)` ; tailles de work-group `2^workers_exp1 × 2^workers_exp2` (32×32, ou 16×16 sur CPU). **`nx` et `ny` doivent en être des multiples** : `SimulationSession` les arrondit à 32.

### `tools.py`
- `ObsertacleManager` : `load_file` lit un profil (`x;y` par ligne, contour fermé), le centre et le met à l'échelle (`scaling · nx`) ; `rotate_and_move` le tourne, le déplace et le rasterise en masque booléen. Pour ajouter une forme : déposer un fichier `.txt` dans `world_objects/`.
- `render_field(u, v, obstacle, mode, vmin, vmax)` : image couleur d'un champ de vitesse (solveurs Python).

---

## 6. Protocole client/serveur

Un seul WebSocket `/ws`. Pour un client Unity (ou autre), il suffit de parler ce protocole :

| Sens | Message | Contenu |
|---|---|---|
| client → serveur | texte JSON `{"type":"start","config":{...}}` | démarre (ou redémarre) ; clés de `simulation_session.DEFAULT_CONFIG` |
| client → serveur | `{"type":"set","params":{...}}` | paramètres en direct (`re`, `display`, `obstacle_*`, `t_max`, `curl_max`, `iter_per_frame`…) |
| client → serveur | `{"type":"stop"}` | arrêt |
| client → serveur | `{"type":"ack"}` | image affichée : le serveur peut envoyer la suivante (contrôle de flux) |
| serveur → client | binaire | une image JPEG (axe `y` déjà vers le haut) |
| serveur → client | texte JSON `{"type":"stats",...}` | `state`, `error`, `fps`, `mlups`, `iteration`, `device`, `grid` (toutes les 0,5 s) |

Pour porter le moteur lui-même dans Unity, les kernels de `kernels.cl` sont du C simple, traduisibles en compute shaders HLSL ; les solveurs Python (et les tests) servent alors de référence numérique.

---

## 7. Tests

```
.venv\Scripts\python -m unittest discover -s tests -v
```
`tests/test_models.py` vérifie, sur une grille 128×64 (Re = 1000, plaque inclinée), après 1, 10, 100 et 300 pas :
- l'écoulement uniforme sans obstacle reste uniforme ; un obstacle crée bien un sillage ;
- **Numba = NumPy** (écart maximal sur `u, v, ρ` < 5e-6) ; idem après un changement de Reynolds en cours de route ;
- **OpenCL = NumPy** sur **chaque** device OpenCL utilisable (GPU et CPU) ; ignoré si aucun device ne fonctionne ;
- **colorant** : une étape d'advection sur un colorant aléatoire (NumPy, Numba et OpenCL concordent à 1e-5), et une simulation avec émetteur sur 1, 10 et 30 pas (écart < 1e-4) ; le colorant reste nul sans émetteur.

---

## 8. Performances et dépannage

- **Mesures** (512×256, Ryzen 9 5900X / RTX 4080) : Numba ~178 MLUPS, OpenCL GPU ~97 MLUPS (rendu et JPEG compris), OpenCL CPU Intel ~68 MLUPS, NumPy ~17 MLUPS. Avec l'IHM, les FPS sont aussi limités par la copie de l'image sur CPU et l'encodage JPEG.
- **Le CPU n'apparaît pas dans la liste OpenCL** : vérifier `pip show intel-opencl-rt` dans le venv. pocl (conda-forge) a été essayé : il est détecté mais ne peut pas lier ses kernels sans Visual Studio, il n'est donc pas utilisé.
- **L'interface se fige au démarrage avec le CPU Intel** : le runtime Intel a bloqué de façon intermittente à la création du contexte OpenCL (observé seulement pendant les premiers tests). Relancer `app.py`.
- **Partage OpenCL/OpenGL** : non utilisé (la roue pip de PyOpenCL est compilée sans). L'image est relue côté CPU. Les restes d'interop dans `lbm_opencl.py` (`add_postrun_texture_interop`, `acquire/release_gl_objects`) ne servent plus qu'à ce chemin.
- **Instabilité** : à Reynolds élevé et vitesse forte, la simulation peut diverger (image uniforme, NaN). Réduire `speedup` ou le Reynolds. La sortie à gradient nul peut produire des réflexions.
- **Colorant** : sensible. Il suit des fronts raides (jusqu'à 1 d'écart entre deux cellules voisines) et l'interpolation bicubique dépasse un peu 1 ; sur plusieurs centaines de pas, les petites différences de float32 entre solveurs s'amplifient près de l'émetteur (jusqu'à ~4e-2 après 300 pas entre OpenCL et NumPy, alors que les vitesses concordent à 1e-6). Les tests comparent donc le colorant sur un pas (strict) et sur quelques dizaines de pas (plus lâche). Il est ~8 fois plus lent avec colorant en NumPy (100 ms par pas en 512×256).
