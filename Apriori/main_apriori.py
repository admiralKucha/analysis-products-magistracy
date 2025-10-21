import pandas as pd
import apriori

DATASET_PATH = './dataset/'


class AprioriCalculator:
    def __init__(self, dataset_path: str):
        """
        Инициализация, подсчёт по алгоритму Априори
        :param dataset_path:
        """
        self.dataset_path = dataset_path
        self.result = []

        # Собираем данные. Нам важны категории продуктов, а не сами продукты
        order_products_prior = pd.read_csv(self.dataset_path + 'order_products__prior.csv')
        order_products_train = pd.read_csv(self.dataset_path + 'order_products__train.csv')
        prior_train_concat = pd.concat([order_products_prior, order_products_train])

        categories = pd.read_csv(self.dataset_path + 'aisles.csv')
        products = pd.read_csv(self.dataset_path + 'products.csv')

        order_products = pd.merge(prior_train_concat, products, on='product_id')
        order_products_category = pd.merge(order_products, categories, on='aisle_id')

        # Собираем транзакции. Список множеств категорий товаров в заказе
        basket = order_products_category.groupby(['order_id'])['aisle'].apply(set).reset_index()
        transactions = basket['aisle'].tolist()

        # Прогоняем алгоритмом Априори
        tmp_result = list(apriori.apriori(transactions))
        self.result = []
        for r in tmp_result:
            # Берём только полезные правила
            if len(r.items) > 1:
                self.result.append(r)

    def predict(self, basket):
        """
        Предсказание рекомендации по добавлению в корзину
        :param basket: Текущая корзина - множество категорий в заказе
        :return: множество рекомендованных категорий товаров и его достоверность
        """
        max_confidence = 0
        predict_set = set()

        for r in self.result:
            # Смотрим пересекающиеся правила
            if (basket & r.items) != set():
                for statistic in r.ordered_statistics:
                    # Если у нас есть более достоверная рекомендация, то эту пропускаем
                    if statistic.confidence < max_confidence:
                        continue

                    # Данная статистика должна быть подмножеством нашей корзины
                    if statistic.items_base.issubset(basket):
                        add_items = statistic.items_add - basket
                        # Если ничего нового нет, то пропускаем
                        if len(add_items) == 0:
                            continue

                        predict_set = add_items
                        max_confidence = statistic.confidence

        return predict_set, max_confidence

    def test_on_all_aisles(self):
        """
        Тестирование рекомендации на всех категориях товаров
        """

        # Получение всех категорий
        categories = pd.read_csv(self.dataset_path + 'aisles.csv')
        for category in list(categories['aisle']):
            # Собираем корзину из одной этой категории. В дальнейшем можно пособирать корзины из нескольких категорий
            basket = {category}
            predict_set, confidence = self.predict(basket)

            # Если система смогла что-то порекомендовать, то пишем
            if predict_set != set():
                print(f'Корзина: {basket};\t\tПредложение: {predict_set};\t\tДостоверность: {confidence}')


if __name__ == '__main__':
    calculator = AprioriCalculator(DATASET_PATH)
    calculator.test_on_all_aisles()
