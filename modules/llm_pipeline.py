import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def call_ollama(prompt, num_predict=700):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.8,
                "num_predict": num_predict
            }
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"]


def get_profile_instructions(profil_public):
    profil = (profil_public or "").lower().strip()

    if profil == "enfants":
        return """
ADAPTATION OBLIGATOIRE AU PROFIL : ENFANTS
- Utilise des mots très simples.
- Fais des phrases courtes.
- Évite les noms scientifiques sauf si demandés.
- Ton positif, ludique et facile à comprendre.
- Question au public amusante ou basée sur l'observation.
- Réponse spectateur très simple, 1 à 2 phrases.
"""

    if profil == "scolaires":
        return """
ADAPTATION OBLIGATOIRE AU PROFIL : SCOLAIRES
- Utilise un vocabulaire pédagogique.
- Explique simplement les mots difficiles.
- Fais le lien avec habitat, alimentation, comportement et rôle écologique.
- Donne des phrases claires, niveau collège/lycée.
- Question au public qui fait réfléchir.
"""

    if profil == "experts":
        return """
ADAPTATION OBLIGATOIRE AU PROFIL : EXPERTS
- Utilise un vocabulaire scientifique précis.
- Mentionne le nom scientifique quand il est disponible.
- Précise les notions d'écologie, comportement, conservation ou adaptation.
- Évite les formulations trop simplifiées.
- Question au public plus technique.
"""

    return """
ADAPTATION OBLIGATOIRE AU PROFIL : ADULTES
- Utilise un vocabulaire clair et informatif.
- Reste concis.
- Mentionne les informations importantes sans jargon excessif.
- Question au public ouverte et naturelle.
"""


def format_detections(detections):
    lines = []
    for i, det in enumerate(detections, start=1):
        animal = det.get("animal") or det.get("label") or "unknown"
        confidence = float(det.get("confidence", 0))
        lines.append(f"{i}. {animal} : {confidence:.2f}")
    return "\n".join(lines)


def get_animal_names(detections):
    return [
        det.get("animal") or det.get("label")
        for det in detections
        if det.get("animal") or det.get("label")
    ]


def generate_suggestions(
    detections,
    transcription,
    profil_public,
    rag_context,
    scenario="",
    spectator_question=""
):
    detections_fiables = [
        d for d in detections
        if float(d.get("confidence", 0)) >= 0.50
    ]

    if not detections_fiables:
        detections_fiables = detections

    profile_instructions = get_profile_instructions(profil_public)
    detections_text = format_detections(detections_fiables)
    animal_names = get_animal_names(detections_fiables)
    animal_count = len(animal_names)
    plusieurs_animaux = animal_count > 1

    objets_detectes_section = f"""## Objets détectés
{detections_text}
"""

    audio_disponible = (
        transcription
        and transcription.strip()
        and transcription.strip().lower() != "aucun audio fourni."
    )

    reformulation_section = ""
    if audio_disponible:
        reformulation_section = f"""
## Reformulation du médiateur
Reformule uniquement cette phrase :
"{transcription}"

Contraintes :
- Conserve exactement le sens.
- Si l'audio ne cite pas d'animal précis, ne rajoute aucun animal.
- N'ajoute aucune information issue du RAG.
- Ne mets pas de guillemets.
- Adapte le niveau de langage au profil du public.
- Une seule phrase naturelle.
"""

    comparaison_section = ""
    if plusieurs_animaux:
        comparaison_section = """
## Comparaison
Compare brièvement TOUS les animaux détectés.

Contraintes :
- Mentionne chaque animal au moins une fois.
- Compare uniquement habitat, alimentation, comportement ou rôle écologique.
- Utilise uniquement les fiches RAG correspondantes.
- Maximum 2 phrases.
- Si il y'a 4 animaux détectés, fais une comparaison générale entre eux.
- Adapte la formulation au profil du public.
"""

    prompt_suggestions = f"""
Tu es l'assistant IA de TeleVirtuality.
Tu aides un médiateur humain pendant un live sur la faune.

Animaux détectés EXACTEMENT :
{detections_text}

Liste exacte des animaux :
{", ".join(animal_names)}

Nombre exact d'animaux :
{animal_count}

Profil du public :
{profil_public}

{profile_instructions}

Scénario de la visite :
{scenario}

CONTEXTE RAG STRUCTURÉ PAR ANIMAL :
{rag_context}

RÈGLES ABSOLUES :
- Ne génère jamais la section "Objets détectés".
- Cette section est déjà produite par Python.
- Utilise exactement les animaux listés.
- Ne supprime aucun animal.
- N'ajoute aucun animal.
- Ne remplace jamais un animal par un autre.
- Ne traduis pas les noms des animaux.
- Ne recopie jamais les consignes du prompt.

RÈGLES RAG :
- Chaque bloc RAG correspond uniquement à l'animal indiqué par "ANIMAL DÉTECTÉ".
- Pour Deer, utilise uniquement le bloc RAG Deer.
- Pour Zebra, utilise uniquement le bloc RAG Zebra.
- Pour Mule, utilise uniquement le bloc RAG Mule.
- Pour Cattle, utilise uniquement le bloc RAG Cattle.
- N'utilise jamais une information provenant du bloc RAG d'un autre animal.
- Si une information existe dans le bloc RAG de l'animal, utilise cette information sans ajouter une information supplémentaire.
- Ne mélange jamais les caractéristiques de deux animaux différents.

RÈGLES DE COHÉRENCE :
- Le scénario décrit le contexte de visite, pas l'habitat naturel.
- Ne déduis jamais le lieu réel à partir de l'habitat naturel.
- Le résumé doit décrire uniquement ce qui est observé dans le live.
- Si un animal représente un groupe général, n'invente pas de nom scientifique précis.
- Pour Deer, écris "Famille : Cervidae" si aucune espèce précise n'est fournie.
- Zebra = zèbre, jamais zébu.
- Zébu = bovin domestique à bosse.
- Cattle = bétail/bovin domestique.
- Mule = hybride cheval × âne.
- Le conseil au médiateur ne doit jamais être vide.

FORMAT À PRODUIRE :

## Résumé de la scène
Une seule phrase courte.
Mentionne tous les animaux détectés.
Ne donne pas d'habitat ou d'alimentation ici.
Ne déduis jamais le lieu réel depuis le RAG.
Le style doit suivre l'adaptation au profil.

{reformulation_section}

## Informations utiles par animal
Crée exactement {animal_count} sous-sections.
Chaque sous-section commence par le nom exact de l'animal.

Pour chaque animal :
- Nom scientifique ou famille : ...
- Habitat naturel : ...
- Alimentation : ...
- Rôle écologique : ...

Adapte le vocabulaire au profil du public.
Utilise uniquement les rubriques RAG correspondantes à chaque animal.
Donne les informations utiles pour tous les animaux détectés.
Si il y'a plusieurs animaux détectés, fais une sous-section pour chacun d'eux.

{comparaison_section}

## Anecdote
Une seule anecdote courte et fiable.
Elle doit concerner uniquement un animal détecté.
Utilise d'abord l'anecdote présente dans le bloc RAG correspondant.
Adapte le style au profil.

## Statut de conservation
Donne le statut pour chaque animal détecté.
Si le statut dépend de l'espèce précise, écris : variable selon l'espèce.
utilise uniquement la rubrique "Statut de conservation" du RAG.
Adapte la formulation au profil.

## Question à poser au public
Une seule question ouverte, liée aux animaux détectés.
Utilise les questions possibles du RAG si elles existent.
La question doit être clairement adaptée au profil du public.

## Conseil au médiateur
Une phrase obligatoire commençant par "Tu peux...".
Elle doit être directement utilisable pendant le live.
Elle doit être adaptée au profil du public.
Utilise une information présente dans le RAG.
"""

    suggestions = call_ollama(prompt_suggestions, num_predict=700)

    if spectator_question and spectator_question.strip():
        prompt_question = f"""
Tu es l'assistant IA de TeleVirtuality.
Ta seule mission est de répondre à la question d'un spectateur.

Question du spectateur :
{spectator_question}

Animaux détectés EXACTEMENT :
{detections_text}

Liste exacte des animaux :
{", ".join(animal_names)}

Profil du public :
{profil_public}

{profile_instructions}

Scénario :
{scenario}

CONTEXTE RAG STRUCTURÉ PAR ANIMAL :
{rag_context}

RÈGLES :
- Réponds directement à la question.
- Ne recopie pas la question.
- Ne reformule pas la question.
- Réponds uniquement pour les animaux détectés.
- Si plusieurs animaux sont détectés, réponds pour chacun d'eux.
- Utilise le bloc RAG correspondant à chaque animal.
- Ne mélange jamais les fiches RAG.
- Si le RAG ne contient pas la réponse, utilise des connaissances générales fiables.
- Le scénario n'est pas l'habitat naturel.
- Si la question concerne l'habitat, donne l'habitat naturel réel.
- Réponse courte : 2 phrases maximum.
- Le vocabulaire doit suivre strictement le profil du public.
"""

        answer_question = call_ollama(prompt_question, num_predict=280)
        suggestions += "\n\n## Réponse à la question du spectateur\n"
        suggestions += answer_question

    return objets_detectes_section + "\n" + suggestions