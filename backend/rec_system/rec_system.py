import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# 1. ЗАГРУЗКА ДАННЫХ ИЗ CSV ФАЙЛОВ
print("Загрузка данных из CSV файлов...")

def load_csv_data():
    base_path = r"C:\uni\analysis-products-magistracy\backend\init_db\csv"
    
    # Загружаем основной файл с заказами
    try:
        orders_df = pd.read_csv(f'{base_path}/orders.csv')
        print(f"Загружено orders: {len(orders_df)} строк")
    except FileNotFoundError:
        print("Файл orders.csv не найден!")
        return None, None, None

    # Загружаем PRIOR данные (исторические заказы)
    try:
        order_products_prior_df = pd.read_csv(f'{base_path}/order_products__prior.csv')
        print(f"Загружено order_products__prior: {len(order_products_prior_df)} строк")
    except FileNotFoundError:
        print("Файл order_products__prior.csv не найден!")
        order_products_prior_df = pd.DataFrame()

    # Загружаем TRAIN данные (обучающие заказы)
    try:
        order_products_train_df = pd.read_csv(f'{base_path}/order_products__train.csv')
        print(f"Загружено order_products__train: {len(order_products_train_df)} строк")
    except FileNotFoundError:
        print("Файл order_products__train.csv не найден!")
        order_products_train_df = pd.DataFrame()

    # Объединяем все продукты в один DataFrame
    all_products_df = pd.concat([order_products_prior_df, order_products_train_df], ignore_index=True)
    
    # Добавляем информацию о eval_set из orders_df
    all_products_df = all_products_df.merge(orders_df, on='order_id', how='left')
    
    return orders_df, all_products_df, order_products_train_df

# Загружаем данные
orders_df, all_products_df, order_products_train_df = load_csv_data()

# Показываем информацию о данных
print("\nИнформация о данных:")
print("Orders columns:", orders_df.columns.tolist())
print("Orders shape:", orders_df.shape)
print("Orders eval_set counts:")
print(orders_df['eval_set'].value_counts())

print("\nAll products shape:", all_products_df.shape)
print("All products eval_set counts:")
print(all_products_df['eval_set'].value_counts())

# 2. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ
print("\nПредварительная обработка данных...")

# Проверяем и чиним данные
print("Проверка пропущенных значений:")
print("Orders NaN:", orders_df.isnull().sum().sum())
print("Products NaN:", all_products_df.isnull().sum().sum())

# Заполняем пропуски в days_since_prior_order (для первых заказов)
orders_df['days_since_prior_order'] = orders_df['days_since_prior_order'].fillna(0)

# Разделяем данные согласно eval_set
print("\nРазделение данных по eval_set:")
train_orders = orders_df[orders_df['eval_set'].isin(['prior', 'train'])]
test_orders = orders_df[orders_df['eval_set'] == 'test']

print(f"Train orders (prior + train): {len(train_orders)}")
print(f"Test orders: {len(test_orders)}")

# Берем соответствующие продукты
train_order_ids = train_orders['order_id'].tolist()
test_order_ids = test_orders['order_id'].tolist()

train_products = all_products_df[all_products_df['order_id'].isin(train_order_ids)]
test_products = all_products_df[all_products_df['order_id'].isin(test_order_ids)]

print(f"Train products: {len(train_products)}")
print(f"Test products: {len(test_products)}")

# 3. ПОДГОТОВКА ФИЧЕЙ И ПОСЛЕДОВАТЕЛЬНОСТЕЙ
print("\nПодготовка фичей и последовательностей...")

# Создаем словарь для кодирования product_id
all_products = pd.concat([train_products['product_id'], test_products['product_id']])
product_encoder = LabelEncoder()
product_encoder.fit(all_products)
NUM_PRODUCTS = len(product_encoder.classes_)
print(f"Всего уникальных товаров: {NUM_PRODUCTS}")

# Функция для создания последовательностей с временными фичами
def create_sequences_with_features(users_data, products_data, sequence_length=10):
    sequences = []
    targets = []
    user_features = []
    
    user_count = 0
    for user_id, user_orders in users_data.groupby('user_id'):
        user_count += 1
        if user_count % 1000 == 0:
            print(f"Обработано пользователей: {user_count}")
            
        # Сортируем заказы по order_number
        user_orders = user_orders.sort_values('order_number')
        
        orders_list = []
        for _, order in user_orders.iterrows():
            order_id = order['id']
            order_products = products_data[products_data['order_id'] == order_id]
            
            # Временные фичи для заказа
            time_features = [
                order['order_dow'] / 6.0,  # Нормализуем день недели [0,1]
                order['order_hour_of_day'] / 23.0,  # Нормализуем час [0,1]
                (order['days_since_prior_order'] or 0) / 30.0  # Нормализуем дни [0,1]
            ]
            
            # Товары в заказе
            product_ids = order_products['product_id'].values
            
            orders_list.append({
                'products': product_ids,
                'time_features': time_features,
                'order_id': order_id
            })
        
        # Создаем скользящее окно (исключая последний заказ для test set)
        max_i = len(orders_list) - 1 if user_orders['eval_set'].iloc[-1] != 'test' else len(orders_list)
        
        for i in range(max_i - 1):
            start_idx = max(0, i - sequence_length + 1)
            sequence = orders_list[start_idx:i + 1]
            target = orders_list[i + 1]['products']
            
            # Паддим последовательность если нужно
            if len(sequence) < sequence_length:
                padding = [{
                    'products': np.array([]),
                    'time_features': [0.0, 0.0, 0.0]
                }] * (sequence_length - len(sequence))
                sequence = padding + sequence
            
            sequences.append(sequence)
            targets.append(target)
            user_features.append([user_id] + orders_list[i + 1]['time_features'])
    
    return sequences, targets, user_features

# Создаем последовательности для обучения и теста
print("Создание обучающих последовательностей...")
train_sequences, train_targets, train_user_features = create_sequences_with_features(train_orders, train_products)

print("Создание тестовых последовательностей...")
test_sequences, test_targets, test_user_features = create_sequences_with_features(test_orders, test_products)

print(f"Train sequences: {len(train_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# 4. DATASET И DATALOADER (остается без изменений)
class OrderSequenceDataset(Dataset):
    def __init__(self, sequences, targets, user_features, product_encoder, max_products_per_order=50):
        self.sequences = sequences
        self.targets = targets
        self.user_features = user_features
        self.product_encoder = product_encoder
        self.max_products = max_products_per_order
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        user_feat = self.user_features[idx]
        
        # Кодируем последовательность
        seq_products = torch.zeros(len(sequence), self.max_products, dtype=torch.long)
        seq_time_features = torch.zeros(len(sequence), 3, dtype=torch.float)
        seq_mask = torch.zeros(len(sequence), self.max_products, dtype=torch.bool)
        
        for i, order in enumerate(sequence):
            # Временные фичи
            seq_time_features[i] = torch.tensor(order['time_features'])
            
            # Товары
            products = order['products']
            if len(products) > 0:
                encoded_products = self.product_encoder.transform(products[:self.max_products])
                seq_products[i, :len(encoded_products)] = torch.tensor(encoded_products)
                seq_mask[i, :len(encoded_products)] = True
        
        # Цель - multi-hot encoding
        target_tensor = torch.zeros(NUM_PRODUCTS, dtype=torch.float)
        if len(target) > 0:
            encoded_target = self.product_encoder.transform(target)
            target_tensor[encoded_target] = 1.0
        
        # User features (исключаем user_id)
        user_tensor = torch.tensor(user_feat[1:], dtype=torch.float)
        
        return seq_products, seq_time_features, seq_mask, user_tensor, target_tensor

# 5. МОДЕЛЬ RNN (остается без изменений)
class EnhancedBasketRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, dropout=0.3):
        super(EnhancedBasketRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.product_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + 3, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim + 3)
        self.fc1 = nn.Linear(hidden_dim + 3, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
                
    def forward(self, products, time_features, mask, user_features):
        batch_size, seq_len, max_products = products.size()
        
        # Обработка продуктов
        products_flat = products.view(-1)
        embedded_flat = self.product_embedding(products_flat)
        embedded = embedded_flat.view(batch_size, seq_len, max_products, -1)
        
        mask_expanded = mask.unsqueeze(-1)
        embedded_masked = embedded * mask_expanded
        sum_embeddings = torch.sum(embedded_masked, dim=2)
        product_count = torch.sum(mask, dim=2, keepdim=True).clamp(min=1)
        order_embeddings = sum_embeddings / product_count
        
        # Объединяем с временными фичами
        lstm_input = torch.cat([order_embeddings, time_features], dim=2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        last_output = lstm_out[:, -1, :]
        
        # Добавляем user features
        combined = torch.cat([last_output, user_features], dim=1)
        
        # Классификация
        x = self.batch_norm(combined)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 6. ОБУЧЕНИЕ МОДЕЛИ
def train_model():
    # Параметры
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32  # Уменьшил для стабильности
    LEARNING_RATE = 0.001
    EPOCHS = 5       # Уменьшил для быстрого тестирования
    SEQ_LENGTH = 5   # Уменьшил длину последовательности
    
    # Создаем датасеты
    train_dataset = OrderSequenceDataset(
        train_sequences, train_targets, train_user_features, product_encoder
    )
    test_dataset = OrderSequenceDataset(
        test_sequences, test_targets, test_user_features, product_encoder
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Модель и оптимизатор
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedBasketRNN(
        NUM_PRODUCTS, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Начало обучения...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, (products, time_feat, mask, user_feat, target) in enumerate(train_loader):
            products, time_feat, mask, user_feat, target = [
                x.to(device) for x in [products, time_feat, mask, user_feat, target]
            ]
            
            optimizer.zero_grad()
            output = model(products, time_feat, mask, user_feat)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Валидация
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for products, time_feat, mask, user_feat, target in test_loader:
                products, time_feat, mask, user_feat, target = [
                    x.to(device) for x in [products, time_feat, mask, user_feat, target]
                ]
                
                output = model(products, time_feat, mask, user_feat)
                loss = criterion(output, target)
                epoch_test_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Test Loss:  {avg_test_loss:.4f}')
    
    return model, device

# ЗАПУСК ОБУЧЕНИЯ
if __name__ == "__main__":
    if orders_df is not None and len(all_products_df) > 0:
        model, device = train_model()
        
        # Пример предсказания
        if len(test_sequences) > 0:
            test_dataset = OrderSequenceDataset(
                test_sequences, test_targets, test_user_features, product_encoder
            )
            
            sample = test_dataset[0]
            products, time_feat, mask, user_feat, target = sample
            
            model.eval()
            with torch.no_grad():
                products, time_feat, mask, user_feat = [
                    x.unsqueeze(0).to(device) for x in [products, time_feat, mask, user_feat]
                ]
                
                output = model(products, time_feat, mask, user_feat)
                probabilities = torch.sigmoid(output).cpu().numpy()[0]
                
                # Топ-10 предсказаний
                top_n = 10
                top_indices = np.argsort(probabilities)[-top_n:][::-1]
                top_products = product_encoder.inverse_transform(top_indices)
                top_probs = probabilities[top_indices]
                
                print("\nПример предсказания для тестового пользователя:")
                for product_id, prob in zip(top_products, top_probs):
                    print(f"Product {product_id}: {prob:.3f}")
    else:
        print("Не удалось загрузить данные!")