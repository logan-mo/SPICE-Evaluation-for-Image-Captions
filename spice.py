import json
import requests
import networkx as nx
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

LOGGING = False
CORENLP_URL = "http://localhost:9000"
STOPWORDS = {"a", "an", "the", "some", "any"}
PUNCTUATION_TAGS = {".", ",", "''", "``", ":", "-LRB-", "-RRB-"}

lemmatizer = WordNetLemmatizer()

properties = {
    "annotators": "tokenize,pos,lemma,depparse",
    "outputFormat": "json",
    "split": "false",
}

ALLOWED_DEPS = {
    "nsubj",
    "dobj",
    "amod",
    "prep",
    "nmod",
    "root",
    "advmod",
    "compound",
    "acl",
    "xcomp",
    "auxpass",
    "cop",
    "number",
    "nmod:poss",
    "nmod:of",
    "nmod:in",
    "nmod:during",
    "obl:in",
    "mark",
}


def are_synonyms(word1, word2):
    """Check if two words are synonyms using WordNet"""
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    # Get all lemma names for each word
    synonyms1 = {lemma.name() for syn in synsets1 for lemma in syn.lemmas()}
    synonyms2 = {lemma.name() for syn in synsets2 for lemma in syn.lemmas()}

    # Check if they share any synonyms
    return bool(synonyms1 & synonyms2)


def match_with_synonyms(word1, word2):
    """Match words directly or through synonyms"""
    if word1 == word2:
        return True
    return are_synonyms(word1, word2)


def _get_dependencies(sentence):
    """Send a sentence to CoreNLP and return dependency graph."""
    response = requests.post(
        CORENLP_URL,
        params={"properties": json.dumps(properties)},
        data=sentence.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )
    if response.status_code != 200:
        raise Exception(f"CoreNLP Error: {response.text}")

    parsed_data = response.json()
    tokens = parsed_data["sentences"][0]["tokens"]
    dependencies = parsed_data["sentences"][0]["basicDependencies"]

    word_pos = {token["word"]: token["pos"] for token in tokens}

    return dependencies, word_pos


def get_dependencies(sentence):
    """Send a sentence to CoreNLP and return dependency graph for all sentences."""
    response = requests.post(
        CORENLP_URL,
        params={"properties": json.dumps(properties)},
        data=sentence.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )

    if response.status_code != 200:
        raise Exception(f"CoreNLP Error: {response.text}")

    parsed_data = response.json()
    scene_graph = []
    word_pos = {}

    deps = []

    for sentence_data in parsed_data["sentences"]:
        dependencies = sentence_data["enhancedDependencies"]
        triples = []

        for token in sentence_data["tokens"]:
            word_pos[token["word"].lower()] = token["pos"]

        for dep in dependencies:
            if dep["dep"] in ALLOWED_DEPS:

                triples.append(
                    {
                        "governorGloss": dep["governorGloss"].lower(),
                        "dep": dep["dep"].lower(),
                        "dependentGloss": dep["dependentGloss"].lower(),
                    }
                )
            deps.append(dep["dep"])

        scene_graph.extend(triples)

    return scene_graph, word_pos


def lemmatize_word(word, pos):
    """Lemmatize words based on POS tags."""
    if pos.startswith("V"):
        return lemmatizer.lemmatize(word, "v")
    elif pos.startswith("N"):
        return lemmatizer.lemmatize(word, "n")
    return word


def build_scene_graph(dependencies, word_pos):
    """Create a scene graph from dependency relations."""
    G = nx.DiGraph()

    for dep in dependencies:
        raw_head = dep["governorGloss"]
        raw_tail = dep["dependentGloss"]
        relation = dep["dep"]

        # TODO: Filter out words less than or equal to 2 characters
        if (
            raw_tail.lower() in STOPWORDS
            or word_pos.get(raw_tail) in PUNCTUATION_TAGS
            or len(raw_tail) <= 2
        ):
            continue
        if (
            raw_head.lower() in STOPWORDS
            or word_pos.get(raw_head) in PUNCTUATION_TAGS
            or len(raw_head) <= 2
        ):
            continue

        head = lemmatize_word(raw_head, word_pos.get(raw_head, ""))
        tail = lemmatize_word(raw_tail, word_pos.get(raw_tail, ""))

        G.add_edge(head, tail, label=relation)

    return G


def plot_dependency_graph(dependencies, word_pos, additional_tags=""):
    """Plot dependency graph with entities (blue) and actions (red), ignoring stopwords/punctuation."""

    G = nx.DiGraph()

    entity_deps = {"nsubj", "dobj", "pobj", "nmod", "compound", "amod", "prep"}
    action_deps = {
        "root",
        "acl",
        "xcomp",
        "auxpass",
        "advmod",
    }

    entities, actions = set(), set()

    for dep in dependencies:
        raw_head = dep["governorGloss"]
        raw_tail = dep["dependentGloss"]
        relation = dep["dep"]

        # Ignore punctuation and stopwords
        if (
            (
                raw_tail.lower() in STOPWORDS
                or word_pos.get(raw_tail) in PUNCTUATION_TAGS
                or raw_head.lower() in STOPWORDS
                or word_pos.get(raw_head) in PUNCTUATION_TAGS
            )
            or len(raw_tail) <= 2
            or len(raw_head) <= 2
        ):
            continue

        # Get POS tags and lemmatize words
        head = lemmatize_word(raw_head, word_pos.get(raw_head, ""))
        tail = lemmatize_word(raw_tail, word_pos.get(raw_tail, ""))

        if relation in entity_deps:
            entities.add(tail)
        if relation in action_deps or head == "ROOT":
            actions.add(tail)

        G.add_edge(head, tail, label=relation)

    # Define node colors
    node_colors = []
    for node in G.nodes():
        if node in entities:
            node_colors.append("lightblue")  # Entities (nouns)
        elif node in actions:
            node_colors.append("lightcoral")  # Actions (verbs)
        else:
            node_colors.append("lightgray")  # Other words

    # Draw the graph
    # pos = nx.spring_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, "label")

    plt.figure(figsize=(10, 7))  # Make the figure larger

    # nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=2000, font_size=10)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2500,
        font_size=12,
        font_weight="bold",
        edgecolors="black",
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels,
        font_size=10,
        font_color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.title(
        f"Entities (Blue) and Actions (Red) in Sentence (Filtered + Lemmatized), {additional_tags}"
    )
    plt.show()


def get_spice_score(
    caption1, caption2, corenlp_url="http://localhost:9000", plot=False
):
    # Define parameters for CoreNLP
    def get_scene_graph(caption):
        response = requests.post(
            CORENLP_URL,
            params={"properties": json.dumps(properties)},
            data=caption.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )

        if response.status_code != 200:
            raise Exception("Error processing caption with CoreNLP")

        parsed_data = response.json()
        scene_graph = []

        for sentence in parsed_data["sentences"]:
            dependencies = sentence["enhancedDependencies"]
            triples = []
            for dep in dependencies:
                if dep["dep"] in ALLOWED_DEPS:
                    triples.append(
                        (
                            dep["governorGloss"].lower(),
                            dep["dep"].lower(),
                            dep["dependentGloss"].lower(),
                        )
                    )
            scene_graph.append(triples)

        return scene_graph

    sg1 = get_scene_graph(caption1)
    sg2 = get_scene_graph(caption2)

    if LOGGING:
        print(sg1)
        print(sg2)

    # Calculate SPICE precision, recall, and F1 score
    sg1_set = set(tuple(triple) for sentence in sg1 for triple in sentence)
    sg2_set = set(tuple(triple) for sentence in sg2 for triple in sentence)

    # matching_triples = len(sg1_set & sg2_set)  # Intersection of sets
    matching_triples = sum(
        1
        for triple1 in sg1_set
        for triple2 in sg2_set
        if match_with_synonyms(triple1[0], triple2[0])  # Match governor word
        and triple1[1] == triple2[1]  # Exact relation match
        and match_with_synonyms(triple1[2], triple2[2])  # Match dependent word
    )
    precision = matching_triples / max(len(sg1_set), 1)
    recall = matching_triples / max(len(sg2_set), 1)

    dependencies1, word_pos1 = get_dependencies(caption1)
    dependencies2, word_pos2 = get_dependencies(caption2)
    if plot:
        plot_dependency_graph(dependencies1, word_pos1, additional_tags="Caption 1")
        plot_dependency_graph(dependencies2, word_pos2, additional_tags="Caption 2")

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "metrics": {"precision": precision, "recall": recall, "spice_score": f1_score},
        "caption_1": {
            "dependencies": dependencies1,
            "word_pos": word_pos1,
        },  # Dependency graph and POS tags
        "caption_2": {
            "dependencies": dependencies2,
            "word_pos": word_pos2,
        },
    }


if __name__ == "__main__":
    # Example Usage
    caption1 = "Dean Bombac, center, of Slovenia, plays against Miguel Sanchez-Migallon of Spain during a men's handball match at the 2024 Summer Olympics, Saturday, July 27, 2024, in Paris, France. (AP Photo/Aaron Favila)"
    caption2 = "Wide shot of a men's handball match during the 2024 Summer Olympics in Paris. Dean Bombac of Slovenia, wearing a white and blue uniform, is at the center, actively engaging with Miguel Sanchez-Migallon of Spain, in a red and black uniform. Bombac holds the ball in his right hand, while Sanchez-Migallon attempts to block him with his left arm. Another player in a white uniform is visible in the background. The audience is partially visible in the blurred background. The floor is a shade of green, and the atmosphere is dynamic and competitive."

    spice_result = get_spice_score(caption1, caption2, plot=True)
    print(spice_result)
