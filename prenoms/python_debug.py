"""
Script de diagnostic : trace le chemin exact entre deux prénoms dans un groupe.
Reconstitue le graphe de fusions depuis les paires sauvegardées + N1,
puis cherche le chemin BFS entre deux prénoms donnés.
"""
import json, pickle, sys
from collections import defaultdict, deque

GROUPED   = "prenoms/data/3_prenoms_grouped.json"
GROUPES   = "prenoms/data/3_groupes_prenoms.json"
CACHE_PKL = "prenoms/data/3_paires_prenoms.pkl"
INPUT     = "prenoms/data/2_prenoms_clean.json"

# Les deux prénoms à tracer
PRENOM_A = "maria"
PRENOM_B = "arava"
# Groupe suspect
GROUPE_ID = 1521
# Combien de membres du groupe afficher
N_MEMBRES = 30

print("=== Chargement ===")
with open(INPUT,   "r", encoding="utf-8") as f: data_in  = json.load(f)
with open(GROUPED, "r", encoding="utf-8") as f: data_grp = json.load(f)

prenoms    = [d["prenom"] for d in data_in]
p_to_idx   = {p: i for i, p in enumerate(prenoms)}
textes     = {d["prenom"]: (d.get("histoire","") or "") + " " +
                            (d.get("etymologie","") or "") + " " +
                            (d.get("provenance","") or "")
              for d in data_in}

# Membres du groupe 1521
membres = [d["prenom"] for d in data_grp if d["id_groupe_total"] == GROUPE_ID]
print(f"\nGroupe {GROUPE_ID} : {len(membres)} membres")
print("Premiers membres :", membres[:N_MEMBRES])

# Textes des deux prénoms cibles
for p in [PRENOM_A, PRENOM_B]:
    d = next((x for x in data_in if x["prenom"] == p), None)
    if d:
        print(f"\n--- {p} ---")
        print(f"  len prenom   : {len(p)}")
        print(f"  histoire     : {(d.get('histoire') or '')[:120]}")
        print(f"  etymologie   : {(d.get('etymologie') or '')[:120]}")
        print(f"  provenance   : {(d.get('provenance') or '')[:80]}")
        print(f"  prenoms_lies : {d.get('prenoms_lies', [])}")
        print(f"  id_groupe    : {d.get('id_groupe')}")

# N1 : reconstituer les liens explicites
print("\n=== Liens N1 (prenoms_lies) dans le groupe ===")
from collections import Counter
lies_freq = Counter()
for d in data_in:
    for p in d.get("prenoms_lies", []):
        lies_freq[p] += 1

edges_n1 = set()
for d in data_in:
    i = p_to_idx.get(d["prenom"])
    for p_lie in d.get("prenoms_lies", []):
        if lies_freq[p_lie] > 8:
            continue
        j = p_to_idx.get(p_lie)
        if j is not None and i != j:
            edges_n1.add((min(i,j), max(i,j)))

# N2 : paires FAISS sauvegardées
print("=== Chargement cache paires N2 ===")
try:
    with open(CACHE_PKL, "rb") as f:
        paires_n2 = pickle.load(f)
    print(f"Paires N2 chargées : {len(paires_n2)}")
except FileNotFoundError:
    paires_n2 = set()
    print("Cache N2 introuvable")

# Graphe complet
graphe = defaultdict(set)
for i, j in edges_n1:
    graphe[i].add(('N1', j))
    graphe[j].add(('N1', i))
for i, j in paires_n2:
    graphe[i].add(('N2', j))
    graphe[j].add(('N2', i))

# BFS entre PRENOM_A et PRENOM_B
print(f"\n=== Chemin BFS {PRENOM_A} -> {PRENOM_B} ===")
idx_a = p_to_idx.get(PRENOM_A)
idx_b = p_to_idx.get(PRENOM_B)

if idx_a is None or idx_b is None:
    print("Prénom introuvable dans le corpus")
    sys.exit(1)

visited = {idx_a: None}  # idx -> (niveau, parent_idx)
queue   = deque([(idx_a, [])])
found   = None

while queue:
    cur, path = queue.popleft()
    if cur == idx_b:
        found = path + [(cur, None, None)]
        break
    for niveau, voisin in graphe[cur]:
        if voisin not in visited:
            visited[voisin] = cur
            queue.append((voisin, path + [(cur, niveau, voisin)]))

if found is None:
    print("Aucun chemin direct trouvé dans le graphe sauvegardé.")
    print("Le lien vient peut-être de l'injection des groupes Phase 1 (id_groupe).")
    # Vérifier si même id_groupe Phase 1
    d_a = next((x for x in data_in if x["prenom"] == PRENOM_A), {})
    d_b = next((x for x in data_in if x["prenom"] == PRENOM_B), {})
    print(f"id_groupe Phase1 {PRENOM_A} : {d_a.get('id_groupe')}")
    print(f"id_groupe Phase1 {PRENOM_B} : {d_b.get('id_groupe')}")
else:
    print(f"Chemin en {len(found)-1} sauts :")
    for step in found[:-1]:
        cur_idx, niveau, next_idx = step
        p_cur  = prenoms[cur_idx]
        p_next = prenoms[next_idx]
        t_cur  = textes.get(p_cur, "").strip()[:80]
        t_next = textes.get(p_next, "").strip()[:80]
        print(f"  [{niveau}] {p_cur!r:20s} (texte: {t_cur!r})")
        print(f"        -> {p_next!r:20s} (texte: {t_next!r})")

# Afficher les voisins directs d'arava dans le graphe
print(f"\n=== Voisins directs de {PRENOM_B} dans le graphe ===")
for niveau, voisin in sorted(graphe[idx_b], key=lambda x: prenoms[x[1]]):
    p_v   = prenoms[voisin]
    t_v   = textes.get(p_v, "").strip()[:60]
    print(f"  [{niveau}] {p_v:20s} | texte: {t_v!r}")