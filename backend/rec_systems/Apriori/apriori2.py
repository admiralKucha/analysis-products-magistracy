import pickle

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

PATH = "./backend/rec_systems/"
DATASET_PATH = "k_means/dataset/"
dataset_path = PATH + DATASET_PATH
flag = True

if not flag:
    # Загрузка данных
    # order_products_prior = pd.read_csv(dataset_path + "order_products__prior.csv")
    order_products_train = pd.read_csv(dataset_path + "order_products__train.csv")

    # prior_train_concat = pd.concat([order_products_prior, order_products_train])
    prior_train_concat = order_products_train

    products = pd.read_csv(dataset_path + "products.csv")

    # Объединение данных
    order_products = pd.merge(prior_train_concat, products, on="product_id")

    # Собираем транзакции. Список множеств товаров в заказе
    basket = order_products.groupby(["order_id"])["product_id"].apply(list).reset_index()
    transactions = basket["product_id"].tolist()

    # Преобразуем транзакции в формат one-hot encoding
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    print("Размерность данных:", df.shape)

    frequent_itemsets = apriori(df, min_support=0.0001, use_colnames=True, max_len=1, low_memory=True)

    if len(frequent_itemsets) == 0:
        print("Не найдено частых наборов. Попробуйте уменьшить min_support.")
        exit()

    print(f"\nНайдено {len(frequent_itemsets)} частых наборов:")
    print(frequent_itemsets.head(10))

    # Генерируем правила ассоциации
    if len(frequent_itemsets) <= 1:
        print("Недостаточно частых наборов для генерации правил")
        exit()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)

    if len(rules) == 0:
        print("Не найдено правил ассоциации. Попробуйте уменьшить min_threshold.")
        exit()

    # ФИЛЬТРУЕМ: оставляем только правила с lift > 1
    rules = rules[rules["lift"] > 1.0]
    with open("./backend/rec_systems/results/apriori2.pkl", "wb") as f:
        pickle.dump(rules, f)

else:
    with open("./backend/rec_systems/results/apriori2.pkl", "rb") as f:
        rules = pickle.load(f)

if len(rules) == 0:
    print("Нет правил с lift > 1. Попробуйте уменьшить min_threshold.")
    exit()

print(f"\nНайдено {len(rules)} правил с lift > 1:")

dict_all_rules = {}
for _, rule in rules.iterrows():
    antecedents = rule["antecedents"]
    consequents = rule["consequents"]
    confidence = rule["confidence"]

    for mini_rule in antecedents:
        el = dict_all_rules.get(mini_rule, {})
        for value in consequents:
            old_confidence = el.get(value, 0)
            if confidence > old_confidence:
                el[value] = confidence
            else:
                el[value] = old_confidence

        dict_all_rules[mini_rule] = el


for key, value in dict_all_rules.items():
    print(key, value)

with open("./backend/rec_systems/results/apriori2_dict.pkl", "wb") as f:
    pickle.dump(dict_all_rules, f)
