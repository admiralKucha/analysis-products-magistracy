import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.patheffects as path_effects

DATASET_PATH = './dataset/'

class ClusteringUsers:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._user_features = None
        self.k = 10  # по методу локтя
        self._k_means = None

        self.orders = pd.read_csv(self.dataset_path + 'orders.csv')

        order_products_prior = pd.read_csv(self.dataset_path + 'order_products__prior.csv')
        order_products_train = pd.read_csv(self.dataset_path + 'order_products__train.csv')
        prior_train_concat = pd.concat([order_products_prior, order_products_train])
        self.full_data = pd.merge(self.orders, prior_train_concat, on='order_id')

        self.retrain_k_means()


    def _prepare_user_features(self):
        full_data_with_orders = self.full_data.copy()

        user_features = full_data_with_orders.groupby('user_id').agg(
            total_orders=('order_number', 'max'),                       # Общее количество заказов пользователя
            total_products=('product_id', 'count'),                     # Общее количество купленных продуктов
            total_unique_products=('product_id', 'nunique'),            # Количество уникальных продуктов, которые покупал пользователь
            avg_products_per_order=('product_id',
                                    lambda x: x.count() / x.nunique())  # Среднее количество продуктов в заказе
        ).reset_index()

        orders = self.orders.copy()
        user_order_features = orders.groupby('user_id').agg(
            avg_days_since_prior=('days_since_prior_order', 'mean'),    # Средний перерыв между заказами
            avg_order_dow=('order_dow', 'mean')                         # Средний день недели для заказа
        ).reset_index()

        user_features = pd.merge(user_features, user_order_features, on='user_id')
        user_features = user_features.fillna(0)

        features_for_clustering = user_features.drop('user_id', axis=1)

        # Масштабирование признаков
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_for_clustering)

        return scaled_features, user_features


    def retrain_k_means(self):
        scaled_features, user_features = self._prepare_user_features()

        self._k_means = KMeans(n_clusters=self.k, random_state=42, n_init='auto')
        self._k_means.fit(scaled_features)

        user_features['cluster'] = self._k_means.labels_
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
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.title('Коэффициент силуэта для разных K (на выборке)')
        plt.xlabel('Количество кластеров (k)')
        plt.ylabel('Silhouette Score')
        plt.show()

    def plot_optimal_k_elbow(self, max_k=15):
        scaled_features = self._prepare_user_features()[0]

        inertia_values = []
        k_range = range(1, max_k + 1)

        print("Вычисление инерции для разного k...")
        for k in k_range:
            # Создаем и обучаем модель KMeans
            print(k, 'шаг')
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(scaled_features)

            # Сохраняем значение инерции (WCSS)
            inertia_values.append(kmeans.inertia_)

        print(inertia_values)

        # Строим график
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia_values, marker='o', linestyle='--')

        plt.title('Метод локтя для определения оптимального k', fontsize=16)
        plt.xlabel('Количество кластеров (k)', fontsize=12)
        plt.ylabel('Инерция (WCSS)', fontsize=12)
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()

    def get_recommendation(self, user_cluster: int, num_recs: int = 5):
        data_with_clusters = pd.merge(self.full_data, self._user_features[['user_id', 'cluster']], on='user_id')
        cluster_bestsellers = data_with_clusters.groupby(['cluster', 'product_id']).size().reset_index(
            name='purchase_count')

        # Сортируем, чтобы найти самые популярные товары в каждом кластере
        cluster_bestsellers = cluster_bestsellers.sort_values(['cluster', 'purchase_count'], ascending=[True, False])

        # print(f"\n--- Топ-{num_recs} товаров для Кластера {user_cluster} ---")
        top_products = cluster_bestsellers[cluster_bestsellers['cluster'] == user_cluster].head(num_recs)
        # print(top_products)
        return top_products

    def get_user_info(self, user_id):
        '''
        :param user_id
        :return:
            total_orders: Общее число заказов
            total_products:  Общее количество купленных продуктов
            total_unique_products: Количество уникальных продуктов
            avg_products_per_order: Среднее количество продуктов в заказе
            avg_days_since_prior: Средний перерыв между заказами
            avg_order_dow: Средний день недели для заказа
            cluster: Номер кластера пользователя
        '''
        if self._user_features is None:
            return None

        return self._user_features[self._user_features['user_id'] == user_id]

    def plot_users_data_2d(self):
        df = self._user_features.copy()
        features = df.drop(['user_id', 'cluster'], axis=1)
        cluster_labels = df['cluster']

        # 2. Масштабирование признаков
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # 3. Применение PCA для уменьшения размерности до 2 компонент
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)

        # Создание DataFrame с результатами PCA
        pca_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
        pca_df['cluster'] = cluster_labels

        # 4. Поиск представителя для каждого кластера
        representative_indices = []
        unique_clusters = sorted(df['cluster'].unique())

        for cluster_id in unique_clusters:
            # Находим оригинальные индексы всех точек, принадлежащих кластеру
            original_indices = df[df['cluster'] == cluster_id].index

            # Выбираем масштабированные данные для этих точек
            cluster_points_scaled = scaled_features[original_indices]

            # Вычисляем центроид (среднее значение признаков) для кластера
            centroid = cluster_points_scaled.mean(axis=0)

            # Находим точку, ближайшую к центроиду
            distances = euclidean_distances(cluster_points_scaled, centroid.reshape(1, -1))
            closest_point_local_index = np.argmin(distances)

            # Находим оригинальный индекс этой точки в исходном DataFrame
            representative_original_index = original_indices[closest_point_local_index]

            representative_indices.append(representative_original_index)

        # Получаем PCA-координаты и исходные данные для представителей
        representative_pca_coords = pca_df.loc[representative_indices]
        representative_data = df.loc[representative_indices].round(2)  # Округляем для наглядности

        # 5. Визуализация результатов (график + таблица)

        # Создаем макет: 2 ряда, 1 колонка. График будет занимать верхнюю часть, а таблица - нижнюю
        fig, axes = plt.subplots(2, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1]})
        ax_plot = axes[0]
        ax_table = axes[1]

        # --- Отрисовка графика (верхняя часть) ---
        # Отображение всех точек
        sns.scatterplot(
            ax=ax_plot,
            x='principal component 1',
            y='principal component 2',
            hue='cluster',
            palette=sns.color_palette("viridis", n_colors=len(unique_clusters)),
            data=pca_df,
            legend='full',
            alpha=0.7
        )

        # Выделение представителей кластеров на графике
        ax_plot.scatter(
            representative_pca_coords['principal component 1'],
            representative_pca_coords['principal component 2'],
            s=150,  # Увеличиваем размер
            edgecolor='red',
            facecolors='white',
            linewidth=2.5,
            label='Примеры'
        )

        # Добавление меток для представителей
        for idx, row in representative_pca_coords.iterrows():
            txt = ax_plot.text(
                row['principal component 1'] + 0.1,
                row['principal component 2'] + 0.1,
                f"C {int(row['cluster'])}",  # Короткая метка "C" + номер кластера
                fontsize=14,
                fontweight='bold',
                color='white',
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])


        ax_plot.set_title('2D проекция данных', fontsize=14)
        #ax_plot.set_xlabel('Первая главная компонента', fontsize=12)
        #ax_plot.set_ylabel('Вторая главная компонента', fontsize=12)
        ax_plot.legend(title='Кластеры')
        ax_plot.grid(True)

        # --- Отрисовка таблицы (нижняя часть) ---
        ax_table.axis('off')  # Отключаем оси для области таблицы
        #ax_table.set_title('Примеры пользователей', fontsize=14, pad=20)
        ax_table.axis('tight')

        # Создаем таблицу
        table = ax_table.table(
            cellText=representative_data.values,
            colLabels=representative_data.columns,
            loc='center',
            cellLoc='center'
        )

        # Автоматически настраиваем ширину столбцов по содержимому
        table.auto_set_column_width(col=list(range(len(representative_data.columns))))

        # Настраиваем стиль таблицы
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.4)  # Немного увеличим масштаб для лучшего вида

        # Делаем заголовок жирным
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', color='white')
                cell.set_facecolor('#40466e')
            # Устанавливаем форматирование для ячеек с данными
            if i > 0:
                cell.set_text_props(ha='center')
                # Пример форматирования числовых значений
                try:
                    cell.get_text().set_text(f'{float(cell.get_text().get_text()):.2f}')
                except ValueError:
                    pass

        fig.tight_layout(pad=3.0) # Обеспечиваем, чтобы все элементы поместились
        plt.show()


    def save_k_means_weights(self, save_path: str='./dataset/weights.csv'):
        if self._user_features is None:
            return

        df_copy = self._user_features.copy()
        df_copy.to_csv(save_path, index=False)


def get_products_basket(first=-1):
    orders = pd.read_csv(DATASET_PATH + 'order_products__train.csv')
    products = pd.read_csv(DATASET_PATH + 'products.csv')
    orders.columns = orders.columns.str.strip()
    products.columns = products.columns.str.strip()
    merged = pd.merge(orders, products, on="product_id", how="left")
    baskets = merged.groupby("order_id")["product_name"].apply(list).tolist()
    return baskets[0:first]


if __name__ == '__main__':
    clustering_users = ClusteringUsers(DATASET_PATH)
    clustering_users.get_recommendation(5)
    user_info = clustering_users.get_user_info(user_id=25)
    print(clustering_users._user_features)
    print(user_info.keys())
    clustering_users.save_k_means_weights()
    clustering_users.plot_users_data_2d()


    #clustering_users.plot_optimal_k_elbow(100)
    #clustering_users.plot_silhouette_score(max_k=100)

