"""
Construction d'un dataset propre français-dioula.
Sources :
  1. Connaissances linguistiques structurées (grammaire, vocabulaire, phrases)
  2. Dataset HuggingFace OBY632/merged-bambara-dioula-dataset (si accessible)
  3. Fusion et déduplication
"""

import json
import re
import unicodedata

# ─────────────────────────────────────────────
#  1. DONNÉES ISSUES DE MES CONNAISSANCES
# ─────────────────────────────────────────────

KNOWLEDGE_PAIRS = [
    # ── Salutations & formules de politesse ──
    ("bonjour", "i ni sogoma"),
    ("bonsoir", "i ni wula"),
    ("bonne nuit", "i ni su"),
    ("salut", "i ni ce"),
    ("bienvenue", "i bisimila"),
    ("au revoir", "k'an bɛn"),
    ("à bientôt", "k'an bɛn joona"),
    ("à demain", "k'an bɛn sini"),
    ("bonne journée", "tile ɲuman kɛ"),
    ("merci", "i ni ce"),
    ("merci beaucoup", "i ni ce kosɛbɛ"),
    ("s'il vous plaît", "i ni ce"),
    ("je vous en prie", "ayi, ko si tɛ"),
    ("pardon", "hakɛ"),
    ("excusez-moi", "hakɛ to ne ma"),
    ("félicitations", "i ka kɛnɛ"),
    ("bravo", "i y'a kɛ"),
    ("bon courage", "hɛrɛ don"),

    # ── Réponses courantes ──
    ("oui", "ɔwɔ"),
    ("non", "ayi"),
    ("d'accord", "o filɛ"),
    ("très bien", "ka ɲi kosɛbɛ"),
    ("c'est bien", "o ka ɲi"),
    ("c'est bon", "o ka ɲi"),
    ("c'est vrai", "o si tiɲɛ"),
    ("c'est faux", "o ma tiɲɛ"),
    ("peut-être", "o bɛ se ka kɛ"),
    ("je ne sais pas", "ne tɛ a dɔn"),
    ("je comprends", "ne y'a faamu"),
    ("je ne comprends pas", "ne ma a faamu"),
    ("répétez s'il vous plaît", "fo tugu i ni ce"),
    ("parlez lentement", "kuma joona joona"),
    ("c'est tout", "o dɔrɔn"),
    ("vraiment", "tiɲɛ na"),

    # ── Questions de base ──
    ("comment vas-tu ?", "i ka kɛnɛ wa ?"),
    ("comment vous appelez-vous ?", "i tɔgɔ di ?"),
    ("quel est ton nom ?", "i tɔgɔ ye mun ye ?"),
    ("d'où viens-tu ?", "i bɔra min ?"),
    ("où vas-tu ?", "i bɛ taa min ?"),
    ("où habites-tu ?", "i sigilen don min ?"),
    ("quel âge as-tu ?", "i si ye san joli ye ?"),
    ("que fais-tu ?", "i bɛ mun kɛ ?"),
    ("as-tu faim ?", "kɔngɔ b'i la wa ?"),
    ("as-tu soif ?", "minnɔgɔ b'i la wa ?"),
    ("est-ce que tu vas bien ?", "i kɛnɛna wa ?"),
    ("combien ça coûte ?", "a jɔli ye dɔrɔmɛ joli ye ?"),
    ("quelle heure est-il ?", "lɛrɛ joli ye sisan ?"),
    ("où est le marché ?", "maruso bɛ min ?"),
    ("où est l'hôpital ?", "dɔkɔtɔrɔso bɛ min ?"),
    ("où sont les toilettes ?", "banjiginso bɛ min ?"),

    # ── Présentations ──
    ("je m'appelle Amadou", "ne tɔgɔ ye Amadou ye"),
    ("je suis français", "ne ye faransi mɔgɔ ye"),
    ("je suis ivoirien", "ne ye Kodiwari mɔgɔ ye"),
    ("je suis malien", "ne ye Mali mɔgɔ ye"),
    ("je viens de France", "ne bɔra Faransi"),
    ("je viens de Côte d'Ivoire", "ne bɔra Kodiwari"),
    ("j'habite à Abidjan", "ne sigilen bɛ Abijani"),
    ("j'habite à Bamako", "ne sigilen bɛ Bamakɔ"),
    ("j'ai vingt ans", "ne si ye mugan ye"),
    ("je suis étudiant", "ne ye kalanso denmisɛn ye"),
    ("je suis médecin", "ne ye dɔkɔtɔrɔ ye"),
    ("je suis enseignant", "ne ye karanso kɛlɛla ye"),
    ("je suis commerçant", "ne ye jula ye"),
    ("je suis marié", "ne furaw kɛra"),
    ("je suis célibataire", "ne ma furu"),
    ("j'ai deux enfants", "ne denfɛn ye fila ye"),

    # ── Famille ──
    ("père", "fa"),
    ("mère", "ba"),
    ("fils", "denkɛ"),
    ("fille", "denmusɔ"),
    ("enfant", "denmisɛn"),
    ("frère", "kɔrɔ"),  # frère aîné
    ("sœur", "kɔrɔmusɔ"),
    ("grand-père", "mamankɔrɔba"),
    ("grand-mère", "mamankɔrɔbamuso"),
    ("oncle", "fa kɔrɔ"),
    ("tante", "fa muso"),
    ("cousin", "fadenya denmisɛn"),
    ("mari", "cɛ"),
    ("femme", "muso"),
    ("beau-père", "wolobaliw"),
    ("famille", "somɔgɔw"),
    ("mon père est malade", "ne fa bana"),
    ("ma mère est à la maison", "ne ba bɛ so"),
    ("mon frère est parti", "ne kɔrɔ taara"),
    ("ma sœur est grande", "ne kɔrɔmusɔ ka bɔ"),

    # ── Corps humain ──
    ("tête", "kun"),
    ("cheveux", "duguma"),
    ("visage", "ɲɛ"),
    ("œil / yeux", "ɲɛ"),
    ("oreille", "tulo"),
    ("nez", "bɔɔ"),
    ("bouche", "daa"),
    ("dent", "ɲɛgɛn"),
    ("langue", "kan"),
    ("cou", "kɛlɛ"),
    ("épaule", "ɲɔgɔn"),
    ("bras", "bolo"),
    ("main", "bolo"),
    ("doigt", "bolo la fin"),
    ("dos", "kɔ"),
    ("ventre", "bɔɲɔ"),
    ("jambe", "sen"),
    ("pied", "senkɔnɔ"),
    ("cœur", "sɔn"),
    ("sang", "joli"),
    ("j'ai mal à la tête", "ne kun bana"),
    ("j'ai mal au ventre", "ne bɔɲɔ bana"),
    ("j'ai de la fièvre", "ne fɛn ka gɛlɛn"),

    # ── Santé ──
    ("je suis malade", "ne banara"),
    ("je vais bien", "ne kɛnɛna"),
    ("j'ai la fièvre", "fiɛvɛ b'i ma"),
    ("j'ai mal", "ne bana"),
    ("appelle le médecin", "dɔkɔtɔrɔ wele"),
    ("je veux aller à l'hôpital", "ne b'a fɛ ka taa dɔkɔtɔrɔso"),
    ("donnez-moi des médicaments", "fura di ne ma"),
    ("j'ai besoin d'aide", "dɛmɛ ka di ne ma"),
    ("il est blessé", "a dɛgɛra"),
    ("elle est enceinte", "muso in dɛnnɔgɔna"),

    # ── Nourriture & boissons ──
    ("eau", "ji"),
    ("lait", "nɔnɔ"),
    ("thé", "tii"),
    ("café", "kafe"),
    ("riz", "malo"),
    ("mil", "tɔ"),
    ("maïs", "kaba"),
    ("viande", "sogo"),
    ("poisson", "jegew"),
    ("poulet", "sow"),
    ("œuf", "sow ji"),
    ("pain", "burɔti"),
    ("sucre", "sufuruni"),
    ("sel", "kɔgɔ"),
    ("huile", "turubu"),
    ("légumes", "jiridenw"),
    ("fruit", "jiridenw"),
    ("banane", "namasa"),
    ("orange", "lɛmuru"),
    ("mangue", "mango"),
    ("arachide", "tiga"),
    ("igname", "kɔgɔ jiri"),
    ("sauce", "naji"),
    ("repas", "dumuni"),
    ("je veux manger", "ne b'a fɛ ka dumu"),
    ("je veux boire de l'eau", "ne b'a fɛ ka ji min"),
    ("j'ai faim", "kɔngɔ b'i la"),
    ("j'ai soif", "minnɔgɔ b'i la"),
    ("c'est délicieux", "a ka di"),
    ("c'est épicé", "a ka jɛ"),
    ("donnez-moi du riz", "malo di ne ma"),
    ("je ne mange pas de viande", "ne tɛ sogo dumu"),

    # ── Animaux ──
    ("chien", "wulu"),
    ("chat", "jakuma"),
    ("cheval", "sɔ"),
    ("âne", "fali"),
    ("vache", "nɔnɔfɛn"),
    ("mouton", "saga"),
    ("chèvre", "bɛ"),
    ("poulet", "sow"),
    ("canard", "kɔnkɔn"),
    ("oiseau", "kɔnɔ"),
    ("serpent", "saan"),
    ("lion", "wara"),
    ("éléphant", "sama"),
    ("singe", "misi"),
    ("crocodile", "bama"),
    ("poisson", "jege"),
    ("fourmi", "sumasaw"),
    ("mouche", "finminɛn"),

    # ── Couleurs ──
    ("rouge", "jɛ"),
    ("blanc", "jɛ"),
    ("noir", "fɛn"),
    ("bleu", "bluu"),
    ("vert", "gwɛn"),
    ("jaune", "numulen"),
    ("orange", "oranje"),
    ("marron", "maro"),
    ("gris", "grize"),
    ("la robe est rouge", "fulaw in ye jɛ ye"),
    ("le ciel est bleu", "sankolo ye bluu ye"),

    # ── Nombres ──
    ("zéro", "zɛrɔ"),
    ("un", "kelen"),
    ("deux", "fila"),
    ("trois", "saba"),
    ("quatre", "naani"),
    ("cinq", "duuru"),
    ("six", "wɔɔrɔ"),
    ("sept", "wolonwula"),
    ("huit", "segin"),
    ("neuf", "kɔnɔntɔn"),
    ("dix", "tan"),
    ("onze", "tan ni kelen"),
    ("douze", "tan ni fila"),
    ("quinze", "tan ni duuru"),
    ("vingt", "mugan"),
    ("trente", "bi saba"),
    ("quarante", "bi naani"),
    ("cinquante", "bi duuru"),
    ("cent", "kɛmɛ"),
    ("mille", "waa kelen"),
    ("j'ai trois enfants", "ne denfɛnw ye saba ye"),
    ("il coûte cinq mille francs", "a jɔli ye waa duuru ye"),

    # ── Temps & calendrier ──
    ("aujourd'hui", "bi"),
    ("hier", "kunu"),
    ("demain", "sini"),
    ("matin", "sogoma"),
    ("midi", "tile sigi"),
    ("après-midi", "wula fɛ"),
    ("soir", "wula"),
    ("nuit", "su"),
    ("semaine", "dɔgɔkun"),
    ("mois", "kalo"),
    ("année", "san"),
    ("lundi", "tɛnɛn"),
    ("mardi", "tarata"),
    ("mercredi", "araba"),
    ("jeudi", "alamisa"),
    ("vendredi", "juma"),
    ("samedi", "sibiri"),
    ("dimanche", "kari"),
    ("janvier", "zanwiye"),
    ("février", "feburiye"),
    ("mars", "marisi"),
    ("avril", "awirili"),
    ("mai", "mɛ"),
    ("juin", "zuwan"),
    ("juillet", "zuiye"),
    ("août", "ut"),
    ("septembre", "sɛtanburu"),
    ("octobre", "ɔkutɔburu"),
    ("novembre", "nowanmburu"),
    ("décembre", "desanburu"),
    ("il est huit heures", "lɛrɛ segin ye sisan"),
    ("il est midi", "tile sigi ye sisan"),
    ("je viendrai demain matin", "ne na na sini sogoma"),
    ("la semaine prochaine", "dɔgɔkun nata"),
    ("l'année dernière", "san tɛmɛnen"),

    # ── Lieu & direction ──
    ("ici", "yan"),
    ("là", "yen"),
    ("là-bas", "yen kɔrɔ"),
    ("droite", "nɛmɛn"),
    ("gauche", "giri"),
    ("tout droit", "yɔrɔ kelen"),
    ("devant", "ɲɛ fɛ"),
    ("derrière", "kɔ fɛ"),
    ("en haut", "kan"),
    ("en bas", "bɔ"),
    ("dedans", "kɔnɔ"),
    ("dehors", "bila"),
    ("près", "kɔsɔbɛ"),
    ("loin", "jan"),
    ("à gauche", "giri fɛ"),
    ("à droite", "nɛmɛn fɛ"),
    ("tournez à gauche", "giri fɛ jɛ"),
    ("tournez à droite", "nɛmɛn fɛ jɛ"),
    ("continuez tout droit", "taa yɔrɔ kelen"),
    ("c'est loin ?", "a jan wa ?"),
    ("ce n'est pas loin", "a man jan"),
    ("le marché est là-bas", "maruso bɛ yen"),

    # ── Maison & objets ──
    ("maison", "so"),
    ("chambre", "nɔfɛ"),
    ("cuisine", "tɔgɔso"),
    ("porte", "daa"),
    ("fenêtre", "finɛtiri"),
    ("lit", "sufu"),
    ("chaise", "sigilan"),
    ("table", "mesa"),
    ("eau", "ji"),
    ("feu", "tasuma"),
    ("lumière", "yeelen"),
    ("téléphone", "telefɔni"),
    ("argent", "wari"),
    ("clé", "lakile"),
    ("sac", "jakuma jakisa"),
    ("vêtement", "fani"),
    ("chaussure", "dɔɔni"),
    ("chapeau", "bonɛ"),
    ("livre", "gafe"),
    ("stylo", "kalamu"),
    ("ouverture la porte", "daa tɛmɛ"),
    ("ferme la porte", "daa datugu"),
    ("allume la lumière", "yeelen jalaki"),
    ("éteins la lumière", "yeelen datugu"),

    # ── Marché & commerce ──
    ("acheter", "san"),
    ("vendre", "feere"),
    ("payer", "sara"),
    ("prix", "jɔli"),
    ("cher", "gɛlɛya"),
    ("pas cher", "nɔgɔya"),
    ("donner", "di"),
    ("recevoir", "sɔrɔ"),
    ("changer", "yɛlɛma"),
    ("monnaie", "wari fitinin"),
    ("combien vaut ce tissu ?", "fani in ye dɔrɔmɛ joli ye ?"),
    ("c'est trop cher", "a gɛlɛya kosɛbɛ"),
    ("faites-moi un bon prix", "a nɔgɔya ne ye"),
    ("je veux acheter du pain", "ne b'a fɛ ka burɔti san"),
    ("avez-vous de la monnaie ?", "wari fitinin b'i bolo wa ?"),
    ("je n'ai pas d'argent", "wari tɛ ne bolo"),
    ("prenez l'argent", "wari minɛ"),

    # ── Transports ──
    ("voiture", "mɔbili"),
    ("moto", "mɔtɔ"),
    ("vélo", "nɛgɛso"),
    ("bus", "busi"),
    ("taxi", "takisi"),
    ("train", "gari"),
    ("avion", "misiriw"),
    ("bateau", "kulun"),
    ("route", "sira"),
    ("gare", "gariso"),
    ("aéroport", "misiriso"),
    ("je veux un taxi", "ne b'a fɛ takisi"),
    ("conduisez-moi à l'hôtel", "ne tɛmɛn otelisofa"),
    ("arrêtez ici", "jan yan"),
    ("combien pour aller au marché ?", "dɔrɔmɛ joli bɛ se ka taa marusola ?"),
    ("le bus part à quelle heure ?", "busi bɛ taa lɛrɛ joli ?"),

    # ── Travail & activités ──
    ("travailler", "baara kɛ"),
    ("étudier", "kalan"),
    ("lire", "kalan"),
    ("écrire", "sɛbɛn"),
    ("manger", "dumu"),
    ("boire", "min"),
    ("dormir", "suu"),
    ("se lever", "wuli"),
    ("marcher", "taa"),
    ("courir", "boli"),
    ("s'asseoir", "sigi"),
    ("parler", "kuma"),
    ("écouter", "lamɛn"),
    ("regarder", "filɛ"),
    ("chercher", "ɲini"),
    ("trouver", "sɔrɔ"),
    ("aider", "dɛmɛ"),
    ("donner", "di"),
    ("prendre", "minɛ"),
    ("partir", "taa"),
    ("venir", "na"),
    ("rester", "to"),
    ("revenir", "sɛgɛn"),
    ("ouvrir", "tɛmɛn"),
    ("fermer", "datugu"),
    ("je travaille tous les jours", "ne bɛ baara kɛ tile o tile"),
    ("il étudie à l'école", "a bɛ kalan kalanso la"),
    ("nous allons au marché", "an bɛ taa maruso"),
    ("elle cuisine le repas", "a bɛ dumuni tɔgɔ"),

    # ── Nature & environnement ──
    ("soleil", "tile"),
    ("lune", "kalo"),
    ("étoile", "wolo"),
    ("pluie", "sanji"),
    ("vent", "foo"),
    ("nuage", "sankolo"),
    ("arbre", "jiri"),
    ("fleur", "jirifuraw"),
    ("herbe", "jaba"),
    ("rivière", "ba"),
    ("fleuve", "ba bɛlɛ"),
    ("mer", "baji"),
    ("montagne", "kulu"),
    ("terre", "duguma"),
    ("ciel", "sankolo"),
    ("il pleut", "sanji bɛ sagi"),
    ("il fait chaud", "teliman ka bɔ"),
    ("il fait froid", "sɛli ka bɔ"),
    ("le soleil se lève", "tile bɛ wuli"),
    ("il fait nuit", "su kɔra"),

    # ── Religion & culture ──
    ("Dieu", "Ala"),
    ("mosquée", "misiri"),
    ("église", "kirisosi"),
    ("prière", "dali"),
    ("jeûne", "tɔɔrɔ"),
    ("fête", "fɛstu"),
    ("mariage", "furu"),
    ("funérailles", "sɔgɔbali"),
    ("griot", "jeli"),
    ("chef", "dugutigi"),
    ("village", "dugu"),
    ("quartier", "dugukolo"),
    ("Allah vous bénisse", "Ala k'i dɛmɛ"),
    ("que Dieu vous aide", "Ala k'i dɛmɛ"),
    ("inch'Allah", "Ala sɔn na"),
    ("bon ramadan", "ramadan ɲuman"),

    # ── Phrases utiles complètes ──
    ("je ne parle pas dioula", "ne tɛ dioula kuma"),
    ("parlez-vous français ?", "i bɛ faransi kuma wa ?"),
    ("je suis perdu", "ne gɛlɛyara sira la"),
    ("aidez-moi s'il vous plaît", "ne dɛmɛ i ni ce"),
    ("appelez la police", "polisi wele"),
    ("au feu !", "tasuma !"),
    ("attention !", "filɛ !"),
    ("arrêtez !", "jɔ !"),
    ("allez !", "taa !"),
    ("vite !", "joona !"),
    ("doucement !", "hɛrɛ hɛrɛ !"),
    ("c'est interdit", "a ma dafara"),
    ("je veux rentrer chez moi", "ne b'a fɛ ka sɛgɛn ne so"),
    ("ma valise a été volée", "ne jalikisa bɔra ne bolo"),
    ("j'ai perdu mon téléphone", "ne telefɔni tɛ ne bolo tun"),
    ("où est l'ambassade ?", "anbisadiso bɛ min ?"),
    ("je veux appeler ma famille", "ne b'a fɛ ka ne somɔgɔw wele"),
    ("l'eau est propre", "ji in ka ɲi"),
    ("l'eau est sale", "ji in kɔrɔra"),
    ("il fait beau aujourd'hui", "bi ka ɲi"),
    ("la route est longue", "sira in jan"),
    ("le village est grand", "dugu in ka bɔ"),
    ("les enfants jouent", "denmisɛnw bɛ tulon kɛ"),
    ("les femmes cuisinent", "musow bɛ dumuni tɔgɔ"),
    ("les hommes travaillent aux champs", "cɛw bɛ baara kɛ foroba la"),
    ("il fait ses prières", "a bɛ a ka dali kɛ"),
    ("nous sommes frères", "an ye fadenyaw ye"),
    ("la paix soit avec vous", "hɛrɛ ka kɛ aw fɛ"),
    ("que la paix règne", "hɛrɛ ka to"),
    ("je t'aime", "ne b'i fɛ"),
    ("tu me manques", "ne b'i kɔsɛbɛ"),
    ("sois prudent", "i yɛrɛ kɔlɔsi"),
    ("prends soin de toi", "i yɛrɛ kɔlɔsi"),
    ("dors bien", "su ɲuman kɛ"),
    ("mange bien", "dumu ɲuman"),
    ("travaille bien", "baara kɛ ɲuman"),
    ("étudie bien", "kalan ɲuman"),
    ("reviens vite", "sɛgɛn joona"),
    ("ne t'inquiète pas", "i jija kosɛbɛ o la"),
    ("tout ira bien", "a bɛna ɲi"),
    ("ne pleure pas", "i kana ɲɔgɔn"),
    ("le bébé pleure", "den ɲɔgɔnna"),
    ("le bébé sourit", "den diyara"),
    ("je suis content de te voir", "ne diyara i lajɛlen"),
    ("longtemps qu'on ne s'est pas vu", "a wulila an ma ɲɔgɔn ye"),

    # ── École & apprentissage ──
    ("école", "kalanso"),
    ("classe", "kalankɔnɔ"),
    ("professeur", "karanmɔgɔ"),
    ("élève", "kalandenw"),
    ("livre", "gafe"),
    ("cahier", "gafe fitini"),
    ("crayon", "kalamu"),
    ("tableau", "jɔlɔni"),
    ("leçon", "kalan lakoro"),
    ("examen", "sɛgɛsɛgɛli"),
    ("je vais à l'école", "ne bɛ taa kalanso"),
    ("j'apprends le dioula", "ne bɛ dioula kalan"),
    ("le professeur explique la leçon", "karanmɔgɔ bɛ kalan lakoro ladili"),
    ("les élèves écoutent", "kalandenw bɛ lamɛn"),
    ("j'ai réussi mon examen", "ne sɛgɛsɛgɛlini kɛra ɲi"),

    # ── Émotions & sentiments ──
    ("je suis heureux", "ne diyara"),
    ("je suis triste", "ne bɛ dimi"),
    ("je suis fatigué", "ne gɛlɛyara"),
    ("je suis en colère", "fɛɛrɛ b'i la"),
    ("j'ai peur", "ne bɛ siran"),
    ("j'ai honte", "ne maloya"),
    ("je suis surpris", "ne yɛlɛmana"),
    ("je suis désolé", "hakɛ to ne ma"),
    ("je suis jaloux", "ne bɛ lɔgɔnya"),
    ("il est content", "a diyara"),
    ("elle pleure", "a bɛ ɲɔgɔn"),
    ("ils rient", "aw bɛ yɛlɛ"),

    # ── Agriculture ──
    ("champ", "foroba"),
    ("semence", "jiridenw"),
    ("récolte", "suman jɛgɛli"),
    ("pluie", "sanji"),
    ("houe", "daba"),
    ("grenier", "suman so"),
    ("semer", "jiridenw sigi"),
    ("cultiver", "foroba baara kɛ"),
    ("récolter", "suman jɛgɛ"),
    ("la récolte est bonne cette année", "suman kɛra ɲi san in"),
    ("il n'a pas plu depuis longtemps", "sanji ma sagi a wulila"),
    ("nous allons aux champs", "an bɛ taa foroba"),
    ("le forgeron fabrique des outils", "numu bɛ nɛgɛ kɛ"),

    # ── Proverbes & expressions ──
    ("l'union fait la force", "an ka ɲɔgɔn dɛmɛ"),
    ("la parole est d'argent le silence est d'or", "kumakan ka ɲi kalan la"),
    ("qui cherche trouve", "min bɛ ɲini a bɛ sɔrɔ"),
    ("peu à peu l'oiseau fait son nid", "dɔɔnin dɔɔnin ye muru di"),
    ("mieux vaut tard que jamais", "ka jɔ ka gɛlɛn ni ka bɔ tɛ"),
]

def normalize(text):
    return text.strip()

def build_dataset():
    pairs = []
    seen = set()

    for fr, dyu in KNOWLEDGE_PAIRS:
        fr_n = normalize(fr)
        dyu_n = normalize(dyu)
        key = (fr_n.lower(), dyu_n.lower())
        if key not in seen and fr_n and dyu_n:
            seen.add(key)
            pairs.append({
                "source": fr_n,
                "target": dyu_n,
                "source_lang": "fr",
                "target_lang": "dioula"
            })

    return pairs, seen

if __name__ == "__main__":
    print("Construction du dataset depuis les connaissances linguistiques...")
    pairs, seen = build_dataset()
    print(f"Paires depuis connaissances : {len(pairs)}")

    # ── Tentative de chargement HuggingFace ──
    try:
        from datasets import load_dataset
        print("\nChargement OBY632/merged-bambara-dioula-dataset...")
        ds = load_dataset('OBY632/merged-bambara-dioula-dataset')

        hf_added = 0
        for split in ds:
            for row in ds[split]:
                fr = (row.get('fr') or '').strip()
                dyu = (row.get('transcription') or '').strip()
                if not fr or not dyu:
                    continue
                key = (fr.lower(), dyu.lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "source": fr,
                        "target": dyu,
                        "source_lang": "fr",
                        "target_lang": "dioula"
                    })
                    hf_added += 1

        print(f"Paires ajoutées depuis HuggingFace : {hf_added}")
    except Exception as e:
        print(f"HuggingFace non disponible : {e}")

    print(f"\nTotal final : {len(pairs)} paires uniques")

    output_path = "dataset_clean.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Fichier sauvegardé : {output_path}")

    # Stats
    src_lens = [len(p['source'].split()) for p in pairs]
    tgt_lens = [len(p['target'].split()) for p in pairs]
    print(f"\nStats source — min:{min(src_lens)} max:{max(src_lens)} moy:{sum(src_lens)/len(src_lens):.1f}")
    print(f"Stats cible  — min:{min(tgt_lens)} max:{max(tgt_lens)} moy:{sum(tgt_lens)/len(tgt_lens):.1f}")
