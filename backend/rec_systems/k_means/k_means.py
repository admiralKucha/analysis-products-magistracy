import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

DATASET_PATH = "./backend/k_means/dataset/"


class ClusteringUsers:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._user_features = None
        self.k = 10  # по методу локтя
        self._k_means = None

        self.orders = pd.read_csv(self.dataset_path + "orders.csv")

        order_products_prior = pd.read_csv(self.dataset_path + "order_products__prior.csv")
        order_products_train = pd.read_csv(self.dataset_path + "order_products__train.csv")
        prior_train_concat = pd.concat([order_products_prior, order_products_train])
        self.full_data = pd.merge(self.orders, prior_train_concat, on="order_id")

        self.retrain_k_means()

    def _prepare_user_features(self):
        full_data_with_orders = self.full_data.copy()

        user_features = (
            full_data_with_orders.groupby("user_id")
            .agg(
                total_orders=("order_number", "max"),  # Общее количество заказов пользователя
                total_products=("product_id", "count"),  # Общее количество купленных продуктов
                total_unique_products=(
                    "product_id",
                    "nunique",
                ),  # Количество уникальных продуктов, которые покупал пользователь
                avg_products_per_order=(
                    "product_id",
                    lambda x: x.count() / x.nunique(),
                ),  # Среднее количество продуктов в заказе
            )
            .reset_index()
        )

        orders = self.orders.copy()
        user_order_features = (
            orders.groupby("user_id")
            .agg(
                avg_days_since_prior=("days_since_prior_order", "mean"),  # Средний перерыв между заказами
                avg_order_dow=("order_dow", "mean"),  # Средний день недели для заказа
            )
            .reset_index()
        )

        user_features = pd.merge(user_features, user_order_features, on="user_id")
        user_features = user_features.fillna(0)

        features_for_clustering = user_features.drop("user_id", axis=1)

        # Масштабирование признаков
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_for_clustering)

        return scaled_features, user_features

    def retrain_k_means(self):
        scaled_features, user_features = self._prepare_user_features()

        self._k_means = KMeans(n_clusters=self.k, random_state=42, n_init="auto")
        self._k_means.fit(scaled_features)

        user_features["cluster"] = self._k_means.labels_
        self._user_features = user_features.copy()

    def plot_silhouette_score(self, min_k=2, max_k=15):
        scaled_features, _ = self._prepare_user_features()

        sample_size = min(20000, scaled_features.shape[0])

        indices = np.random.choice(scaled_features.shape[0], sample_size, replace=False)
        features_sample = scaled_features[indices, :]

        silhouette_scores = []
        k_range = range(min_k, max_k)  # Силуэт нельзя посчитать для k=1

        print(f"Вычисление метрик для разного k на выборке из {sample_size} пользователей...")
        for k in k_range:
            # Обучаем модель на ВСЕХ данных
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)

            # А силуэт считаем только на ВЫБОРКЕ!
            # Важно: для labels_ тоже нужна выборка по тем же индексам
            labels_sample = kmeans.labels_[indices]
            score = silhouette_score(features_sample, labels_sample)
            silhouette_scores.append(score)

            print(f"k={k}, Silhouette Score: {score:.4f}")

        # Строим график
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, marker="o")
        plt.title("Коэффициент силуэта для разных K (на выборке)")
        plt.xlabel("Количество кластеров (k)")
        plt.ylabel("Silhouette Score")
        plt.show()

    def plot_optimal_k_elbow(self, max_k=15):
        scaled_features = self._prepare_user_features()[0]

        inertia_values = []
        k_range = range(1, max_k + 1)

        print("Вычисление инерции для разного k...")
        for k in k_range:
            # Создаем и обучаем модель KMeans
            print(k, "шаг")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(scaled_features)

            # Сохраняем значение инерции (WCSS)
            inertia_values.append(kmeans.inertia_)

        print(inertia_values)

        # Строим график
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia_values, marker="o", linestyle="--")

        plt.title("Метод локтя для определения оптимального k", fontsize=16)
        plt.xlabel("Количество кластеров (k)", fontsize=12)
        plt.ylabel("Инерция (WCSS)", fontsize=12)
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()

    def get_recommendation(self, user_cluster: int, num_recs: int = 20):
        data_with_clusters = pd.merge(self.full_data, self._user_features[["user_id", "cluster"]], on="user_id")
        cluster_bestsellers = (
            data_with_clusters.groupby(["cluster", "product_id"]).size().reset_index(name="purchase_count")
        )

        # Сортируем, чтобы найти самые популярные товары в каждом кластере
        cluster_bestsellers = cluster_bestsellers.sort_values(["cluster", "purchase_count"], ascending=[True, False])

        # print(f"\n--- Топ-{num_recs} товаров для Кластера {user_cluster} ---")
        top_products = cluster_bestsellers[cluster_bestsellers["cluster"] == user_cluster].head(num_recs)
        # print(top_products)
        return top_products

    def get_user_info(self, user_id):
        """
        :param user_id
        :return:
            total_orders: Общее число заказов
            total_products:  Общее количество купленных продуктов
            total_unique_products: Количество уникальных продуктов
            avg_products_per_order: Среднее количество продуктов в заказе
            avg_days_since_prior: Средний перерыв между заказами
            avg_order_dow: Средний день недели для заказа
            cluster: Номер кластера пользователя
        """
        if self._user_features is None:
            return None

        return self._user_features[self._user_features["user_id"] == user_id]


def get_products_basket(first=-1):
    orders = pd.read_csv(DATASET_PATH + "order_products__train.csv")
    products = pd.read_csv(DATASET_PATH + "products.csv")
    orders.columns = orders.columns.str.strip()
    products.columns = products.columns.str.strip()
    merged = pd.merge(orders, products, on="product_id", how="left")
    baskets = merged.groupby("order_id")["product_name"].apply(list).tolist()
    return baskets[0:first]


if __name__ == "__main__":
    clustering_users = ClusteringUsers(DATASET_PATH)
    print(clustering_users.get_recommendation(5))
    user_info = clustering_users.get_user_info(user_id=25)
    print(user_info.keys())
    # clustering_users.plot_optimal_k_elbow(100)
    # clustering_users.plot_silhouette_score(max_k=100)
