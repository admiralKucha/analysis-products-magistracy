import os

import dill
import pandas as pd

import apriori

PATH = "./backend/rec_systems/"
DATASET_PATH = "k_means/dataset/"
CACHE_FILE = "results/apriori_cache.pkl"


class AprioriCalculator:
    def __init__(self, use_cache: bool = True, start_path: str = PATH):
        self.dataset_path = start_path + DATASET_PATH
        self.cache_file = start_path + CACHE_FILE
        self.result = []

        if use_cache and os.path.exists(self.cache_file):
            self._load_from_cache()
        else:
            self._calculate_apriori()
            if use_cache:
                self._save_to_cache()

    def _calculate_apriori(self):
        # Собираем данные. Нам важны категории продуктов, а не сами продукты
        order_products_prior = pd.read_csv(self.dataset_path + "order_products__prior.csv")
        order_products_train = pd.read_csv(self.dataset_path + "order_products__train.csv")
        prior_train_concat = pd.concat([order_products_prior, order_products_train])

        categories = pd.read_csv(self.dataset_path + "aisles.csv")
        products = pd.read_csv(self.dataset_path + "products.csv")

        order_products = pd.merge(prior_train_concat, products, on="product_id")
        order_products_category = pd.merge(order_products, categories, on="aisle_id")

        # Собираем транзакции. Список множеств категорий товаров в заказе
        basket = order_products_category.groupby(["order_id"])["product_id"].apply(set).reset_index()
        transactions = basket["product_id"].tolist()

        # Прогоняем алгоритмом Априори
        tmp_result = list(apriori.apriori(transactions, min_support=0.0005, min_confidence=0.01))
        self.result = []
        for r in tmp_result:
            # Берём только полезные правила
            if len(r.items) > 1:
                self.result.append(r)

    def _save_to_cache(self):
        with open(self.cache_file, "wb") as f:
            dill.dump(self.result, f)

    def _load_from_cache(self):
        with open(self.cache_file, "rb") as f:
            self.result = dill.load(f)

    def predict(self, basket, min_confidence=0.1):
        predict_set = set()

        for r in self.result:
            # Смотрим пересекающиеся правила
            if (basket & r.items) != set():
                for statistic in r.ordered_statistics:
                    # Если у нас есть более достоверная рекомендация, то эту пропускаем
                    if statistic.confidence < min_confidence:
                        continue

                    # Данная статистика должна быть подмножеством нашей корзины
                    if statistic.items_base.issubset(basket):
                        add_items = statistic.items_add - basket
                        # Если ничего нового нет, то пропускаем
                        if len(add_items) == 0:
                            continue

                        predict_set = predict_set.union(add_items)

        return predict_set

    def test_on_all_aisles(self):
        # Получение всех категорий
        categories = pd.read_csv(self.dataset_path + "aisles.csv")
        for category in list(categories["aisle_id"]):
            # Собираем корзину из одной этой категории. В дальнейшем можно пособирать корзины из нескольких категорий
            basket = {category}
            predict_set = self.predict(basket)

            # Если система смогла что-то порекомендовать, то пишем
            if predict_set != set():
                print(f"Корзина: {basket};\t\tПредложение: {predict_set};")

        for category1 in list(categories["aisle_id"]):
            # Собираем корзину из одной двух категорий
            for category2 in list(categories["aisle_id"]):
                basket = {category1, category2}
                predict_set = self.predict(basket)

                # Если система смогла что-то порекомендовать, то пишем
                if predict_set != set():
                    print(f"Корзина: {basket};\t\tПредложение: {predict_set};")


a = AprioriCalculator(use_cache=True)
b = set()
for el in a.result:
    t = el.items
    b.update(t)

print(b)
print(len(b))
