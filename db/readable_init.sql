DROP TABLE IF EXISTS categories CASCADE;
CREATE TABLE categories(
	id Serial Primary key, -- из aisles.csv
	category_name text NOT NULL UNIQUE -- Название категории 
	);

DROP TABLE IF EXISTS departments CASCADE;
CREATE TABLE departments(
	id Serial Primary key, -- из departments.csv
	department_name text NOT NULL UNIQUE -- Название отдела 
	);

DROP TABLE IF EXISTS products CASCADE;
CREATE TABLE products(
	id Serial Primary key, -- из products.csv
    product_name text NOT NULL UNIQUE, -- название товара
    category_id integer NOT NULL, --Категория товара
    department_id integer NOT NULL, -- Отдел, в котором лежит товар
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
    FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE CASCADE
	);

DROP TABLE IF EXISTS orders CASCADE;
CREATE TABLE orders(
	id Serial Primary key, -- из order.csv
    user_id integer NOT NULL, --id пользователя
    eval_set text NOT NULL, -- набор данных
    order_number integer NOT NULL, --номер заказа у пользователя
    order_dow integer NOT NULL, --день недели, в который был сделан заказ
    order_hour_of_day integer NOT NULL, --час, в который был сделан заказ
    days_since_prior_order integer --дней с последнего заказа(не больше 30)
	);

DROP TABLE IF EXISTS orders_products CASCADE;
CREATE TABLE orders_products(
	order_id Serial Primary key, -- из order_products_*.csv
    product_id integer NOT NULL, --id из таблицы продуктов
    add_to_cart_order integer NOT NULL, --позиция выбора в корзину
	reordered boolean NOT NULL, -- выбирался ли он раньше пользователем
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
	);

