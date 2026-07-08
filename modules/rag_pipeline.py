import os

DATA_DIR = "data/animals"


def clean_animal_name(animal):
    return animal.lower().strip().replace(" ", "_")


def retrieve_context_for_animal(animal):
    if not animal:
        return ""

    animal_clean = clean_animal_name(animal)
    fiche_path = os.path.join(DATA_DIR, f"{animal_clean}.txt")

    if os.path.exists(fiche_path):
        with open(fiche_path, "r", encoding="utf-8") as f:
            fiche = f.read().strip()

        return f"""
==============================
ANIMAL DÉTECTÉ : {animal}
FICHIER RAG : {animal_clean}.txt
==============================

{fiche}

==============================
FIN FICHE : {animal}
==============================
"""

    return f"""
==============================
ANIMAL DÉTECTÉ : {animal}
FICHIER RAG : ABSENT
==============================

Aucune fiche RAG trouvée pour cet animal.

==============================
FIN FICHE : {animal}
==============================
"""


def retrieve_context(detections):
    if not detections:
        return ""

    contexts = []
    animals_seen = set()

    for det in detections:
        animal = det.get("animal") or det.get("label")

        if not animal:
            continue

        animal_clean = clean_animal_name(animal)

        if animal_clean in animals_seen:
            continue

        animals_seen.add(animal_clean)
        contexts.append(retrieve_context_for_animal(animal))

    return "\n\n".join(contexts)