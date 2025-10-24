import csv
import pickle


def create_user_cluster_dict():
    """Создает словарь user_id -> cluster из CSV файла"""
    user_cluster_dict = {}
    res = {}

    with open("./backend/rec_systems/results/k_means.pkl", "rb") as f:
        dict_means = pickle.load(f)
        res["clusters"] = dict_means

    print(dict_means)

    with open("./backend/rec_systems/k_means/dataset/user_clusters.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            user_id = int(row["user_id"])
            cluster = int(row["cluster"])
            user_cluster_dict[user_id] = cluster

    res["users"] = user_cluster_dict

    with open("./backend/rec_systems/results/k_means2.pkl", "wb") as f:
        pickle.dump(res, f)


create_user_cluster_dict()