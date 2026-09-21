# car-price-prediction

Прогноз цены подержанных автомобилей по их характеристикам: очистка данных, SQL-анализ и сравнение четырёх регрессионных моделей (Python, pandas, scikit-learn).

## Данные

- `data/car_data.csv` — исходные данные: 8 128 объявлений, 12 столбцов.
- `data/cars_clean.csv` — после очистки: 6 717 записей. Удалено 1 202 дубликата и 209 строк с пропусками, столбец `max_power` приведён к числовому типу.
- Валюта цены в данных не указана.

## Что сделано

1. Очистка и первичный анализ данных (`notebooks/analysis.ipynb`).
2. Признаки: 6 числовых (`year`, `km_driven`, `mileage(km/ltr/kg)`, `engine`, `max_power`, `seats`) и 4 категориальных (`fuel`, `seller_type`, `transmission`, `owner`), закодированные через `OneHotEncoder`.
3. Разбиение на обучающую и тестовую выборки 80/20 (`random_state=42`, 1 344 объекта в тесте).
4. Обучение и сравнение четырёх моделей: Linear Regression, Random Forest, Gradient Boosting и Random Forest с изменёнными гиперпараметрами.
5. SQL-запросы (SQLite): топ-10 самых дорогих автомобилей, средняя цена по типу топлива, коробке передач и типу продавца.
6. Сохранение модели и энкодера через `joblib`.

## Результаты

Метрики на тестовой выборке:

| Модель | MAE | MAPE | R² |
|---|---:|---:|---:|
| Linear Regression | 168 306 | 56.7% | 0.632 |
| **Random Forest (100 деревьев)** | **77 301** | **18.8%** | **0.920** |
| Gradient Boosting (200 деревьев, learning_rate 0.05) | 89 559 | 23.0% | 0.903 |
| Random Forest (300 деревьев, min_samples_leaf=2, max_features=sqrt) | 83 740 | 20.7% | 0.904 |

Лучший результат у базового Random Forest, он и сохранён в `data/car_price_model.pkl`. Наиболее важные признаки: `max_power` (59%), `year` (25%) и `km_driven` (5%).

## Структура репозитория

```
notebooks/analysis.ipynb        # весь анализ и обучение
data/car_data.csv               # исходные данные
data/cars_clean.csv             # очищенные данные
data/car_price_model.pkl        # обученный Random Forest
data/car_price_encoder.pkl      # обученный OneHotEncoder
```

## Как запустить

```
pip install pandas numpy scikit-learn seaborn matplotlib joblib jupyter
jupyter notebook notebooks/analysis.ipynb
```

Модель и энкодер сохранены с scikit-learn 1.8.0. Если у вас другая версия, выполните ноутбук целиком: он пересоздаст оба файла.

## Пример использования
