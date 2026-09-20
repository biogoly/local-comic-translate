# Local Comic Translate

[English](../README.md) | [한국어](README_ko.md) | Français | [简体中文](README_zh-CN.md) | [Español](README_es.md)

<img src="https://i.imgur.com/QUVK6mK.png" alt="Interface de Comic Translate">

Local Comic Translate est un fork communautaire de [Comic Translate](https://github.com/ogkalu2/comic-translate), axé sur la traduction locale et confidentielle ainsi que sur un travail de retouche manuelle plus sûr dans une application de bureau. Il conserve les fonctions de détection, d’OCR, d’inpainting, de rendu, de gestion des projets et des archives, et de prise en charge des webtoons de l’application d’origine. Il y ajoute la traduction avec `llama.cpp` géré par l’application, la retouche artistique avec FLUX.2 Klein, des API directes de fournisseurs en option, ainsi que de nombreuses améliorations de l’édition et de la fiabilité. Aucun compte, abonnement ni crédit de traduction du service d’origine n’est nécessaire.

Le pipeline OCR hérité prend en charge l’anglais, le coréen, le japonais, le français, le chinois simplifié et traditionnel, le russe, l’allemand, le néerlandais, l’espagnol et l’italien comme langues sources. La traduction est possible vers ces langues et d’autres encore.

Ce fork est en développement actif. Les modifications décrites ci-dessous nécessitent actuellement une exécution à partir du code source ; les téléchargements du site de Comic Translate d’origine ne les incluent pas. Les liens de langue ci-dessus donnent accès aux traductions de la documentation actuelle de ce fork. Les libellés des réglages et des boutons sont conservés en anglais dans ces guides pour permettre de les repérer quelle que soit la langue de l’interface.

## Dernière mise à jour — 0.1.0

- **Artistic Edit :** FLUX.2 Klein 4B/9B en local avec des LoRA compatibles, ou l’API Black Forest Labs, avec aperçu avant application, annulation et conservation des retouches dans le projet.
- **Choix du moteur de traduction :** conservez vos réglages de llama-server local tout en passant à OpenAI, Gemini, Claude ou Deepseek avec vos propres clés API.
- **Flux de travail manuel amélioré :** l’OCR fonctionne directement sur les zones tracées à la main ; les traductions manuelles terminées et leur mise en forme sont conservées lors des traitements ultérieurs de la page entière ; les langues source et cible restent sélectionnées d’une page à l’autre.
- **Nettoyage localisé :** restaurez les pixels d’origine au pinceau, clonez un échantillon fixe de couleur et de texture, ou remplissez un masque avec une couleur prélevée sur le fond.
- **Interface soignée :** thèmes Midnight, Parchment, Lavender et Mint en complément de Light/Dark, commandes d’Artistic Edit plus lisibles, nouvel écran de démarrage et catalogues de traduction de l’interface mis à jour.
- **Identité propre au fork :** réglages et données d’application séparés, sans notifications de mise à jour ni recours aux services payants du projet d’origine.

## Points forts de ce fork

### Priorité au local et confidentialité de la traduction par LLM

- Un traducteur **Local LLM** qui ne nécessite ni API de traduction hébergée ni compte Comic Translate.
- Le mode **Managed llama.cpp** lance un processus `llama-server` local, choisit un port libre sur l’interface de boucle locale, réutilise le processus entre les requêtes et l’arrête à la fermeture de l’application.
- Le mode **External server** prend en charge les points de terminaison `/v1` compatibles avec OpenAI, tels qu’Ollama, LM Studio ou un `llama-server` géré séparément.
- Réglages du modèle GGUF, du projecteur multimodal facultatif (`mmproj`), de la taille du contexte, des couches déportées sur le GPU, des délais de démarrage et de requête, du nombre maximal de tokens en sortie, de la température, de Top P et de Top K.
- Envoi facultatif de l’image de la page aux modèles dotés de capacités visuelles. L’OCR existant reste la principale source de texte ; l’image fournit un contexte supplémentaire au modèle multimodal et lui permet de corriger des erreurs évidentes de reconnaissance.
- La validation stricte des identifiants de blocs et les nouvelles tentatives automatiques empêchent qu’une réponse incomplète ou mal formée du modèle décale silencieusement les traductions vers les mauvaises bulles.
- Le cache des traductions de la page utilise l’image source d’origine : un inpainting ultérieur n’invalide donc pas inutilement une traduction mise en cache.

### API directes de fournisseurs, en option

- **Local LLM** reste le choix par défaut. La traduction et l’OCR ne nécessitent et n’utilisent aucune connexion, aucun abonnement ni achat de crédits auprès de Comic Translate d’origine.
- Dans **Settings > Tools > Translator**, choisissez **OpenAI**, **Google Gemini**, **Anthropic Claude** ou **Deepseek** pour utiliser directement un fournisseur cloud.
- Dans **Settings > Provider APIs**, choisissez le fournisseur correspondant, saisissez votre propre clé API, puis sélectionnez ou saisissez l’identifiant de son modèle API. Le fournisseur facture directement votre compte. Les modèles prédéfinis sont des points de départ, et non une liste exhaustive des modèles disponibles.
- Les identifiants Microsoft Azure OCR et la clé Gemini sont également disponibles pour les options OCR cloud existantes. L’OCR par défaut reste local.
- **Save Keys** n’est activé que si vous le choisissez. Les clés sont enregistrées dans les réglages locaux de ce fork, et non dans un coffre chiffré ; elles ne sont jamais stockées dans les fichiers de projet. Désactivez cette option, puis enregistrez les réglages ou fermez l’application pour supprimer les clés de fournisseur conservées ; les choix de modèles restent enregistrés.
- La traduction cloud envoie le texte reconnu et, lorsque **Provide Image as Input to AI** est activé et pris en charge par le modèle, l’image de la page. Les règles et tarifs du fournisseur s’appliquent. Aucun basculement automatique du local vers le cloud n’a lieu.
- L’ancienne configuration **Advanced / Custom** a été supprimée. Un point de terminaison Custom actif et enregistré est migré vers **Local LLM > External server**. Les anciens choix de modèles cloud nommés sont migrés vers leur fournisseur direct en conservant le modèle choisi ; les utilisateurs doivent fournir leur propre clé.
- Les identifiants des fournisseurs et la configuration du serveur local sont indépendants. Changer **Tools > Translator** sélectionne le moteur actif sans effacer l’une ou l’autre configuration ; la liste déroulante de **Provider APIs** sert uniquement à choisir les identifiants à modifier.

### Retouche manuelle plus sûre

- Un flux de travail pratique, page par page, pour vérifier les zones détectées avant tout effacement.
- Les langues source et cible restent sélectionnées lors du passage d’une page à l’autre en mode Manual. Un choix explicite de **Auto** pour la langue source reste lui aussi actif jusqu’à sa modification.
- La commande **OCR** du menu contextuel fonctionne sur une zone tracée à la main, sans devoir d’abord lancer Detect sur toute la page. L’OCR local prépare les limites de lignes manquantes ; une commande OCR explicite relance la reconnaissance de cette zone au lieu de renvoyer une ancienne lecture en cache ou de traiter des cadres sans rapport.
- Detect sur toute la page conserve les zones tracées à la main et supprime les détections en double du même texte. Les opérations Recognize/Translate suivantes préservent leurs textes source et cible déjà finalisés ; Segment et Render conservent leurs calques de texte existants et la mise en forme mot par mot, sans les supprimer ni les dupliquer.
- La conservation d’un calque de texte manuel ne dispense pas du nettoyage du fond : sa zone participe toujours à la segmentation et au nettoyage. Les zones manuelles non traduites suivent le flux de travail normal.
- Commandes Zoom In, Zoom Out et Fit, ainsi que des raccourcis clavier configurables.
- Tri naturel des pages par nom de fichier et tri par date de modification.
- Le texte de remplacement s’affiche immédiatement, sans bordure, dès qu’une nouvelle zone de texte est tracée et que du texte y est saisi.
- La suppression puis la ressaisie du texte conservent la police choisie.
- Le gras, l’italique et le soulignement peuvent s’appliquer à certains mots, plutôt qu’à l’ensemble de l’élément de texte.
- La mise en forme de portions de texte reste compatible avec le déplacement et le redimensionnement des cadres. Le texte cible saisi manuellement est conservé lorsque le focus change, y compris lors d’une saisie japonaise avec un IME.
- La traduction manuelle de toute la page crée les calques de texte manquants. La traduction d’une zone sélectionnée qui n’a pas encore été rendue nettoie cette zone avant de créer son calque de texte traduit.

### Améliorations de l’inpainting et de l’annulation

- Chaque passe de nettoyage manuel peut être annulée ou rétablie avec les commandes Undo/Redo habituelles.
- **Restore brush** restaure les pixels de l’image de base d’origine uniquement sous le pinceau, sans modifier les retouches ailleurs. Le lettrage d’origine est également restauré ; les calques de texte modifiables ne sont pas supprimés. Réglez la taille avec le curseur de pinceau existant. Chaque trait peut être annulé.
- **Clone brush** (icône de tampon) : réglez la taille du pinceau, faites Alt-clic sur une zone propre, puis peignez pour répéter cet échantillon fixe de couleur et de texture. Le cercle cyan en pointillés reste à la source ; le contour noir et blanc continu suit la destination et s’adapte au zoom. **Soft edge** règle la douceur de la transition. **Sample original**, activé par défaut, prélève l’échantillon sous les retouches de nettoyage ; désactivez-le pour copier la page retouchée. Tout changement de taille, de page ou de mode de source nécessite un nouvel échantillonnage par Alt-clic. Chaque trait peut être annulé ; seuls les pixels peints changent. Aucun modèle d’IA n’est utilisé.
- **Pick color**, puis **Fill mask with sampled color**, remplit les zones peintes ou segmentées de la page courante avec une couleur intacte prélevée sur le fond. Cette fonction utilise le masque visible sans extension supplémentaire ni inférence de modèle. Elle convient aux bulles à fond uni ; les dégradés et les dessins texturés nécessitent toujours de l’inpainting ou une retouche artistique. L’annulation restaure à la fois les pixels précédents et le masque.
- **Revert All Inpainting on the Current Page** supprime toutes les retouches de nettoyage enregistrées en une seule opération annulable.
- Les nettoyages successifs partent de la page telle qu’elle est actuellement composée, ce qui empêche le texte déjà supprimé de réapparaître dans les retouches suivantes.
- Le nettoyage manuel au pinceau gère les clics isolés comme les traits et utilise des limites d’image converties de manière sûre en nombres entiers.
- Les masques de secours adaptés à la géométrie distinguent les cartouches rectangulaires des bulles rondes. Ils réduisent ainsi les restes de texte dans les coins des rectangles tout en limitant les interventions près du dessin.
- Les modifications automatiques de la page sont regroupées en étapes d’annulation prévisibles.

### Retouche artistique avec FLUX.2 Klein

- Un processus Diffusers facultatif et isolé peut modifier un cadre de texte sélectionné, une zone peinte ou une page entière avec FLUX.2 Klein 4B ou 9B.
- L’éditeur utilise la précision BF16 native du modèle, avec des modes configurables d’exécution sur GPU ou de déchargement vers le CPU.
- Les LoRA compatibles peuvent être sélectionnés dans `models/loras/flux2-klein`, avec un réglage de leur intensité pour chaque retouche et une validation de la taille du modèle.
- **Use selected translation** peut construire une consigne **Artistic lettering**, ou une consigne **Empty speech bubble** pour y placer ensuite du texte modifiable.
- **Context padding** fournit au modèle le dessin environnant ; **Edit margin** agrandit la zone autorisée à changer autour d’un cadre sélectionné afin qu’un lettrage traduit plus long ne soit pas tronqué aux limites d’origine. Vérifiez l’aperçu avant d’appliquer une retouche élargie.
- Les résultats générés sont présentés dans un aperçu Before/After et sont appliqués au projet sous forme de retouches non destructives et annulables.
- Le processus FLUX reste chargé ; un changement de modèle ou de mode d’utilisation des périphériques entraîne volontairement son rechargement.
- Les modes de déchargement vers le CPU permettent de sélectionner l’indice physique du GPU, ce qui est utile lorsqu’un autre modèle local occupe déjà une carte.
- Un moteur **Black Forest Labs API** facultatif donne accès aux modèles Klein 4B et 9B hébergés, sans environnement FLUX local ni GPU. Configurez sa clé dans **Settings > Provider APIs**, puis sélectionnez ce moteur en haut d’Artistic Edit. **Local Diffusers** reste le choix par défaut à chaque lancement.
- Les retouches cloud reprennent le même fonctionnement pour la sélection, le contexte, la marge de retouche, l’aperçu, Apply et l’annulation. Chaque génération, y compris une nouvelle génération, demande l’autorisation d’envoyer la zone sélectionnée avec son contexte, ou la page entière si elle est sélectionnée. Un aperçu constitue déjà une génération payante, même s’il est rejeté.
- Les LoRA locaux et les réglages du nombre d’étapes, du guidage et du GPU ne sont pas disponibles avec ce moteur hébergé. Le traitement cloud est limité à moins de 4 mégapixels ; les résultats sont validés avant d’être replacés dans les coordonnées de la page et limités au masque de retouche.
- Cancel arrête l’attente du résultat cloud ; il ne garantit ni l’annulation de la tâche chez le fournisseur ni celle de la facturation. Les envois échoués ou dont le résultat est incertain ne sont jamais relancés automatiquement. Après un dépassement de délai, consultez le tableau de bord du fournisseur avant de réessayer. Les restrictions de contenu peuvent différer de celles des modèles locaux.

Documentation de l’API : [retouche d’images BFL](https://docs.bfl.ai/flux_2/flux2_image_editing). L’intégration cloud est couverte par des tests automatisés simulant les communications et les contrôleurs ; de véritables requêtes aux API Flux et LLM ont également été validées pendant le développement. Cela ne constitue pas une garantie pour chaque fournisseur ou modèle : l’accès au compte, les règles de contenu et les frais dépendent toujours du service choisi.

### Apparence, langues et séparation du fork

- Six thèmes : **Dark**, **Light**, **Midnight**, **Parchment**, **Lavender** et **Mint**. Les boutons d’Artistic Edit ont un fond adapté au thème, des libellés lisibles et des bordures visibles.
- Écran de démarrage et identité visuelle du fork mis à jour.
- L’interface propose l’anglais, le coréen, le français, le chinois simplifié, le russe, le japonais, l’allemand, l’espagnol et l’italien. Les nouvelles fonctions sont traduites dans les neuf catalogues Qt, y compris le catalogue turc supplémentaire, qui n’est pas actuellement proposé dans le sélecteur de langue. Les contrôles de couverture valident les paramètres de substitution et les catalogues compilés ; les langues de l’interface sont indépendantes des traductions du README accessibles ci-dessus.
- Le fork utilise ses propres réglages, chemins de données et identité d’application afin d’éviter les interférences avec une version du projet d’origine installée séparément. La migration des anciennes données ne modifie pas l’ancien profil, qu’elle utilise uniquement en lecture.
- Les vérifications automatiques de mises à jour au démarrage sont désactivées, et l’interface commerciale de compte et d’abonnement a été supprimée. La commande manuelle **Check for Updates** recherche désormais les versions publiées sur le GitHub de ce fork, et non celles du projet d’origine.

### GPU et fiabilité

- Les dépendances Windows comprennent ONNX Runtime GPU et les bibliothèques d’exécution CUDA 13/cuDNN 9 correspondantes ; TensorRT n’est plus recherché, sauf demande explicite.
- Un échec de CUDA entraîne un repli propre sur le CPU, sans exiger TensorRT.
- Les corrections couvrent la navigation asynchrone entre pages, l’identification et la conservation des retouches dans les projets, la composition de nettoyages successifs et les rendus qui ne se mettaient pas à jour.
- Une suite de tests de non-régression couvre les réponses du LLM local, les classes de détection, la traduction automatique et manuelle, l’édition de texte, le tri des images, la géométrie de l’inpainting et son annulation, le cache de traduction et la sélection des périphériques.

## Flux de travail recommandé

La traduction automatique reste disponible, mais une restauration soignée des bandes dessinées gagne généralement à se faire page par page :

1. Chargez la bande dessinée et utilisez **Sort** si les pages ne sont pas dans l’ordre de lecture.
2. Sélectionnez le mode **Manual**, définissez les langues source et cible, puis cliquez sur **Detect**. Utilisez **Auto** pour la source lorsque l’œuvre mélange réellement plusieurs langues ou que sa langue est inconnue.
3. Vérifiez les zones avant le traitement : supprimez les détections qui appartiennent au dessin et tracez des cadres autour des dialogues ou des cartouches oubliés.
4. Cliquez sur **Recognize**, puis corrigez le texte source si l’OCR s’est trompé.
5. Cliquez sur **Translate**, puis vérifiez ou modifiez le texte cible.
6. Cliquez sur **Segment**, inspectez les zones de nettoyage proposées, puis cliquez sur **Clean**.
7. Cliquez sur **Render** et ajustez la police, la taille, l’alignement, l’espacement et la mise en valeur de chaque élément de texte.
8. Utilisez le pinceau de nettoyage et **Apply Cleanup** pour les marques restantes. Undo/Redo ou **Revert All Inpainting on the Current Page** permettent de revenir en arrière si le nettoyage abîme le dessin.

Vous pouvez aussi tracer une zone oubliée et choisir **OCR**, puis **Translate** dans son menu contextuel, sans détecter d’abord le reste de la page. Sa traduction terminée reste intacte si vous lancez ensuite le flux manuel sur toute la page. Pour les fonds colorés ou texturés, essayez le remplissage avec une couleur prélevée, le pinceau de restauration ou le pinceau de clonage avant de répéter l’inpainting sur la même zone. Utilisez Artistic Edit lorsque le lettrage doit s’intégrer au dessin plutôt que rester un calque de texte modifiable ordinaire.

Le détecteur conserve volontairement aussi bien le texte des bulles que le texte libre. Cela évite de perdre des zones OCR valides en mode manuel, mais signifie aussi que **Translate All** peut effacer par inpainting des titres artistiques, des panneaux, des onomatopées ou d’autres lettrages faisant partie du dessin. Une vérification manuelle avant le nettoyage est recommandée lorsque la préservation du dessin est importante.

## Installation

### Prérequis

- Python 3.12
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Un exécutable `llama-server` local et un modèle GGUF compatible si vous souhaitez utiliser la traduction locale gérée par l’application
- WinRAR ou 7-Zip dans le `PATH` pour ouvrir des archives CBR

### Exécution à partir du code source

```bash
git clone https://github.com/biogoly/local-comic-translate.git
cd local-comic-translate
uv init --python 3.12
uv add -r requirements.txt --compile-bytecode
uv run comic.py
```

Sous Windows, `run.bat` est également fourni pour lancer l’application après l’installation initiale des dépendances :

```bat
run.bat
```

### Environnement FLUX.2 Klein facultatif

Le moteur **Local Diffusers** utilise un environnement Python séparé afin que ses versions plus récentes de PyTorch, Diffusers et Transformers ne perturbent pas l’environnement principal de l’application. Le moteur BFL API n’a pas besoin de cet environnement facultatif. Depuis la racine du dépôt, créez un environnement Python 3.12 et installez les dépendances facultatives dont les versions sont fixées :

```bat
py -3.12 -m venv .venv-flux
.venv-flux\Scripts\python.exe -m pip install -r requirements-flux.txt
```

Dans **Artistic Edit**, choisissez **Select FLUX Python…**, puis sélectionnez `.venv-flux\Scripts\python.exe`. Les fichiers des modèles doivent déjà être présents dans le cache Hugging Face, car le processus intégré utilise uniquement les fichiers en cache par défaut. La séparation du profil de ce fork ne crée pas de cache Hugging Face distinct et ne duplique pas les fichiers des modèles sélectionnés par l’utilisateur.

Placez les LoRA `.safetensors` facultatifs dans `models/loras/flux2-klein/4b/` ou `models/loras/flux2-klein/9b/`, selon le modèle concerné. Consultez le [guide du dossier LoRA](../models/loras/flux2-klein/README.md) pour connaître les règles de compatibilité et les métadonnées facultatives. Les poids ne sont pas fournis avec le dépôt. Le processus local actuel utilise BF16 ; le chargement expérimental en INT8/NF4 avec bitsandbytes n’est pas activé.

Pour mettre à jour une copie existante du dépôt :

```bash
git pull
uv add -r requirements.txt --compile-bytecode
```

### Configuration d’un LLM local

`llama.cpp` est un environnement d’exécution externe : il n’est pas installé via `requirements.txt` ni par `uv`. Téléchargez ou compilez [`llama.cpp`](https://github.com/ggml-org/llama.cpp), puis procurez-vous un modèle GGUF de conversation ou de suivi d’instructions adapté à votre matériel.

Le moteur n’est pas lié à un modèle particulier ; Gemma 4 est le principal modèle utilisé pendant le développement et les tests actuels du fork.

Dans l’application :

1. Ouvrez **Settings > Tools** et sélectionnez **Local LLM** comme traducteur.
2. Ouvrez **Settings > LLMs**.
3. Choisissez l’un des modes suivants :
   - **Managed llama.cpp** : sélectionnez `llama-server` (facultatif s’il est déjà dans le `PATH`), le modèle GGUF principal et le projecteur visuel GGUF correspondant si le modèle en a besoin.
   - **External server** : saisissez l’URL de base compatible avec OpenAI et le nom exact du modèle exposé par le serveur. L’URL par défaut, `http://127.0.0.1:11434/v1`, correspond à Ollama.
4. Activez **Provide Image as Input to AI** uniquement si le modèle et le serveur choisis prennent en charge la vision. En mode géré, configurez d’abord le fichier `mmproj` requis.

Les réglages de traduction prudents par défaut sont Temperature `0.20`, Top P `0.90` et Top K `40`. Ce sont de bons points de départ pour la traduction ; ne les modifiez que si la documentation de votre modèle recommande d’autres paramètres d’échantillonnage. Réglez Top K sur `0` pour le désactiver.

### Passer du local aux API et inversement

Pour la traduction de texte :

1. Ouvrez **Settings > Provider APIs**, choisissez le fournisseur, saisissez sa clé, puis sélectionnez ou saisissez l’identifiant du modèle. Activez **Save Keys** uniquement si vous souhaitez conserver la clé d’une session à l’autre.
2. Ouvrez **Settings > Tools > Translator** et choisissez ce fournisseur. Le changement s’applique aux requêtes suivantes sans redémarrage.
3. Sélectionnez de nouveau **Local LLM** pour retrouver la configuration enregistrée de llama.cpp ou du serveur externe. Il n’est pas nécessaire de remplacer le point de terminaison local ni d’effacer les réglages du modèle.

Pour la retouche artistique, saisissez la clé **Black Forest Labs** dans **Provider APIs**, puis, en haut d’**Artistic Edit**, faites passer la liste déroulante de **Local Diffusers** à **Black Forest Labs API**. Choisissez 4B/9B et utilisez **Generate Preview**. Revenir au local réactive les commandes locales. Ce choix est indépendant du traducteur de texte.

Pour tester un autre fournisseur, utilisez une page ou une zone non traduite : les traductions manuelles déjà terminées sont volontairement conservées et les résultats en cache peuvent éviter une nouvelle requête. Les requêtes cloud envoient des données au fournisseur choisi et peuvent être facturées ; un aperçu artistique rejeté reste une génération.

### Accélération GPU

Il existe deux réglages GPU indépendants :

- **Settings > Tools > Use GPU** contrôle les traitements ONNX/PyTorch compatibles, tels que la détection, l’OCR et l’inpainting. Sous Windows, les dépendances installent ONNX Runtime GPU ainsi que les bibliothèques d’exécution CUDA 13/cuDNN 9 correspondantes. Un pilote NVIDIA compatible reste nécessaire, mais il n’est pas nécessaire d’installer séparément CUDA Toolkit ou TensorRT.
- **Settings > LLMs > GPU layers** définit le nombre de couches que `llama.cpp` géré par l’application tente de déporter sur le GPU. Les serveurs externes gèrent eux-mêmes leurs réglages GPU.

Le repli sur le CPU reste possible. Les gains de vitesse sur GPU varient selon le modèle, la taille des pages et l’étape qui limite les performances ; le temps de traitement complet d’une page ne diminue pas forcément proportionnellement à l’utilisation brute du GPU.

## Conseils d’utilisation

- `Ctrl` + molette de la souris : zoom
- `Ctrl` + `=` : zoom avant
- `Ctrl` + `-` : zoom arrière
- `Ctrl` + `0` : adapter la page à la fenêtre
- Flèches gauche/droite : passer d’une page à l’autre
- Les gestes habituels du pavé tactile fonctionnent dans la visionneuse
- Vérifiez que la police choisie prend en charge la langue cible
- Les options de tri des pages sont **Name: A to Z**, **Name: Z to A**, **Modified: Oldest First** et **Modified: Newest First**. Le tri des noms est naturel : `page2` précède donc `page10`.
- Si un fichier CBR provoque l’erreur `RarCannotExec("Cannot find working tool")`, ajoutez le dossier d’installation de WinRAR ou de 7-Zip au `PATH`.

## Fonctionnement

### Détection des bulles et segmentation du texte

L’application utilise le modèle [bubble-and-text-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) du projet d’origine, un RT-DETR-v2 entraîné sur des mangas, des webtoons et des bandes dessinées occidentales. Les résultats de détection restent modifiables avant l’OCR, la segmentation ou le nettoyage.

<img src="https://i.imgur.com/TlzVH3j.jpg" width="49%" alt="Texte détecté dans une bande dessinée"> <img src="https://i.imgur.com/h18XrYT.jpg" width="49%" alt="Texte segmenté dans une bande dessinée">

### OCR

Les moteurs OCR par défaut sont :

- [manga-ocr](https://github.com/kha-white/manga-ocr) pour le japonais
- [Pororo](https://github.com/yunwoong7/korean_ocr_using_pororo) pour le coréen
- [PPOCRv5](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) pour les autres langues prises en charge

Gemini et Microsoft Azure Vision restent disponibles en option pour l’OCR, en vous connectant directement à ces fournisseurs avec vos propres identifiants.

### Traduction

La traduction utilise soit le mode local compatible avec OpenAI, soit les intégrations directes de fournisseurs avec votre propre clé décrites ci-dessus ; elle ne passe pas par le service payant du projet d’origine. Les requêtes portant sur toute la page regroupent les zones reconnues admissibles afin de fournir du contexte. Les zones manuelles terminées sont préservées ; les requêtes sur une zone sélectionnée peuvent utiliser le cache ou une requête plus ciblée. Les modèles multimodaux compatibles peuvent également recevoir l’image de la page en option.

### Inpainting et retouche artistique

Les moteurs de nettoyage local restent LaMa, MI-GAN et AOT. L’outil distinct FLUX.2 Klein Artistic Edit prend en charge les modifications génératives : lettrage artistique traduit, reconstruction ou redimensionnement d’une bulle, et réparation localisée du dessin. Son fonctionnement repose sur une validation de l’aperçu avant application, et non sur un remplacement automatique du nettoyage.

<img src="https://i.imgur.com/cVVGVXp.jpg" width="49%" alt="Bande dessinée avant inpainting"> <img src="https://i.imgur.com/bLkPyqG.jpg" width="49%" alt="Bande dessinée après inpainting">

### Rendu du texte

Le texte traduit ou saisi manuellement est réparti sur plusieurs lignes dans des zones modifiables. La police, la taille, l’alignement, l’espacement, la couleur, le contour, le sens d’écriture et la mise en valeur au niveau des caractères peuvent être ajustés avant l’exportation.

## Tests

Lancez la suite de tests de non-régression depuis la racine du dépôt :

```bash
uv run python -m unittest discover -s tests -v
```

Vérifiez séparément la couverture des traductions de l’interface et les catalogues compilés :

```bash
uv run python resources/translations/check_translations.py
```

Les vérifications de publication couvrent l’OCR et la conservation des zones manuelles, le choix des moteurs locaux ou API, les contrats et les aperçus de retouche artistique, la conservation des retouches dans les projets, l’annulation du nettoyage et du clonage, les thèmes, la persistance des langues et la séparation du profil du fork. La plupart des opérations faisant appel aux modèles ou au réseau sont simulées dans les tests de non-régression ; elles ne téléchargent pas de poids et ne consomment pas de crédits API.

## Limites actuelles

- Ce fork est encore jeune et disponible uniquement sous forme de code source ; les essais sur un large éventail de langues, de modèles, de mises en page et de systèmes d’exploitation sont toujours en cours.
- Le mode automatique ne distingue pas toujours les dialogues du lettrage qui appartient au dessin. Utilisez le mode Manual lorsque la préservation du dessin est importante.
- La détection, l’OCR, la traduction, la segmentation et l’inpainting peuvent tous nécessiter des corrections humaines sur les pages difficiles.
- L’envoi d’images nécessite un véritable modèle multimodal, un serveur compatible et, le cas échéant, le fichier de projecteur correspondant.
- La qualité et la vitesse de génération locale dépendent fortement du modèle choisi, de la taille du contexte et de la RAM/VRAM disponible.
- Artistic Edit nécessite actuellement le mode page unique ; il ne s’agit ni d’une étape automatique appliquée à tout un livre ni d’un éditeur de webtoon multipage. Vérifiez l’orthographe du lettrage généré et l’absence de modifications indésirables du dessin avant de l’appliquer.

## Exemples de bandes dessinées du projet d’origine

Ces exemples montrent le fonctionnement initial de Comic Translate avec GPT hébergé. Certains titres disposent également de traductions officielles en anglais.

- [The Wretched of the High Seas](https://www.drakoo.fr/bd/drakoo/les_damnes_du_grand_large/les_damnes_du_grand_large_-_histoire_complete/9782382330128)
- [Journey to the West](https://ac.qq.com/Comic/comicInfo/id/541812)
- [The Wormworld Saga](https://wormworldsaga.com/index.php)
- [Frieren: Beyond Journey's End](https://renta.papy.co.jp/renta/sc/frm/item/220775/title/742932/)
- [Days of Sand](https://9ekunst.nl/2021/05/20/nieuw-album-van-aimee-de-jongh-is-benauwd-als-een-zandstorm/)
- [Player (OH Hyeon-Jun)](https://comic.naver.com/webtoon/list?titleId=745876&page=1&sort=ASC&tab=fri)
- [Carbon & Silicon](https://www.amazon.com/Carbone-Silicium-French-Mathieu-Bablet-ebook/dp/B0C1LTGZ85/)

## Remerciements

Ce travail est un fork de [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate). L’application d’origine, ses modèles, son interface et ses intégrations constituent le socle des modifications présentées ici.

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [lama-cleaner](https://github.com/Sanster/lama-cleaner)
- [dreMaz/AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [Pororo Korean OCR](https://github.com/yunwoong7/korean_ocr_using_pororo)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [dayu_widgets](https://github.com/phenom-films/dayu_widgets)

Distribué sous la [licence Apache 2.0](../LICENSE).
