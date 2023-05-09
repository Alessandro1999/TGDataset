from typing import *
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import db_utilities as dbu

# libraries for the embeddings
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# libraries for the clustering
from sklearn.cluster import KMeans

# libraries for the topics
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_sentence_embeddings(sentences: List[str], transformer_model: str = 'all-MiniLM-L12-v2', save_to: Path = None) -> np.array:
    """
    Computes sentence embeddings for a list of sentences using the SentenceTransformer library.
    :param sentences: list of sentences to compute embeddings for
    :param transformer_model: name of the transformer model to use
    :param save_to: path to save the embeddings to (None if not saving)
    :return: tensor of embeddings
    """
    model = SentenceTransformer(transformer_model).to(device)
    embeddings = model.encode(sentences, convert_to_tensor=True).to('cpu')
    if save_to:
        torch.save(embeddings, save_to)
    return embeddings.numpy()


def get_sentence_embeddings(path: Path, language='en') -> np.array:
    """
    Loads sentence embeddings from a file (if they exists, otherwise, it computes them).
    :param path: path to the embeddings file
    :param language: language of the dataset
    :return: tensor of embeddings
    """
    if path.exists():
        return torch.load(path).numpy()
    else:
        dataset = get_dataset_from_lang(language)
        df = build_dataset_dataframe(dataset)
        return compute_sentence_embeddings(df['message'], save_to=path)


def cluster_embeddings(embeddings: np.array, df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    """
    Clusters embeddings using KMeans.
    :param embeddings: embeddings to cluster
    :param df: dataframe of the dataset
    :param n_clusters: number of clusters to use
    :param save_to: path to save the embeddings to (None if not saving)
    :return: array of cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    df['cluster'] = kmeans.labels_
    return df


def get_dataset_from_lang(language: str) -> List[Dict]:
    """
    Gets the dataset for a given language from the database.
    :param language: language to get the dataset for
    :return: dictionary of the dataset
    """
    df: pd.DataFrame = pd.read_csv(
        "labeled_data\channel_to_language_mapping.csv", sep="\t")
    channel_ids: List[int] = df[df['language'] == language]['ch_id'].tolist()
    return dbu.get_channels_by_ids(channel_ids)


def build_dataset_dataframe(dataset: List[Dict], save_to: Path = None) -> pd.DataFrame:
    """
    Builds a dataframe (ch_id, message) from a dataset.
    :param dataset: dataset to build the dataframe from
    :return: dataframe of the dataset
    """
    channels, username, messages = [], []
    for channel in dataset:
        for message in channel['text_messages']:
            channels.append(channel['_id'])
            username.append(channel['username'])
            messages.append(message['message'])
    df = pd.DataFrame(
        {'ch_id': channels, 'username': username, 'message': messages})
    if save_to:
        df.to_csv(save_to, sep=',', index=False)
    return df


def get_dataset_dataframe(path: Path, language: str = 'en') -> pd.DataFrame:
    """
    Loads a dataset dataframe from a file (if it exists, otherwise, it builds it).
    :param path: path to the dataframe file
    :return: dataframe of the dataset
    """
    if path.exists():
        return pd.read_csv(path)
    else:
        dataset = get_dataset_from_lang(language)
        return build_dataset_dataframe(dataset, save_to=path)


def topics_and_score(df: pd.DataFrame, top_n: int = 25) -> Tuple[List[List[str]], float]:
    '''
    Given the dataframe with the 'cluster' column and the 'message' one,
    this function computes the top 25 words for each cluster(topic) according to per-cluster TF-IDF
    and it also returns the coherence score of the topics with the corpus
    '''
    num_clusters: int = max(set(df.cluster)) + 1
    # for each one of the clusters, group all the messages toghether in one single string
    clusters_corpus: List[str] = []
    for c in range(num_clusters):
        cluster_corpus: str = " ".join(list(df[df.cluster == c].message))
        clusters_corpus.append(cluster_corpus)

    # vectorize the corpus
    vectorizer = TfidfVectorizer()
    X: np.ndarray = vectorizer.fit_transform(clusters_corpus)

    X: torch.Tensor = torch.from_numpy(X.toarray())
    # take for each row (so for each topic/cluster) the top_n values according to TF-IDF
    values, idx = torch.topk(X, k=top_n, axis=-1)

    words = vectorizer.get_feature_names_out()
    topics: List[List[str]] = []
    for c in range(len(idx)):  # for each cluster
        topic = []
        for word in idx[c]:  # for each word of the top_n of the cluster
            topic.append(words[word])  # add the word to the topic
        topics.append(topic)  # add the topic to the list of topics

    # compute the BoW of the messages
    dictionary = corpora.Dictionary()
    # tokenize the messages
    messages = [m.split(" ") for m in list(df.message)]
    # convert them into Bag of Words format
    BoW_corpus = [dictionary.doc2bow(message, allow_update=True)
                  for message in tqdm(messages)]

    # compute topic coherence
    cm = CoherenceModel(topics=topics, corpus=BoW_corpus,
                        dictionary=dictionary, coherence='u_mass')

    seg_top = cm.segment_topics()
    return topics, cm.get_coherence(), cm.get_coherence_per_topic(seg_top)


topics: int = 15
language: str = 'en'

df: pd.DataFrame = get_dataset_dataframe(
    Path(f'{language}_dataset.csv'), language=language)
embeddings: np.array = get_sentence_embeddings(
    f'{language}_embeddings.pt', language=language)
df = cluster_embeddings(embeddings, df, n_clusters=topics)


topics, coherence, coherence_per_topic = topics_and_score(df)
print(f'Coherence: {coherence}')
for i, topic in enumerate(topics):
    print(f'Topic: {topic}, coherence: {coherence_per_topic[i]}')
