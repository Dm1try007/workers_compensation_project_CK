import streamlit as st

import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
# Прогнозирование стоимости страховых выплат
---
## Введение
- Анализ страховых случаев компенсации работникам
- Цель: предсказать UltimateIncurredClaimCost
- Источник: OpenML, датасет Workers Compensation (ID 42876)
---
## Бизнес-задача
- Планирование резервов и тарифов
- Начальная оценка часто отличается от итоговой стоимости
- Регрессионная модель помогает оценивать риск и масштаб выплат
---
## Этапы работы
1. Загрузка и обзор данных
2. Предобработка (даты, категории, масштабирование)
3. Обучение нескольких моделей регрессии
4. Оценка метриками MAE, RMSE, R2
5. Анализ важности признаков
6. Streamlit-приложение с двумя страницами
---
## Предобработка
- Из дат извлечены: месяц, день недели, задержка сообщения
- Категориальные признаки переведены в числовые
- Числовые признаки масштабированы
---
## Модели
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost (если установлен)
---
## Метрики
- MAE: средняя абсолютная ошибка
- RMSE: корень из MSE
- R2: доля объясненной вариации
---
## Приложение Streamlit
- Основная страница: загрузка, обучение, метрики, графики, важность признаков, предсказание
- Страница презентации: слайды Reveal
---
## Итоги
- Получена модель для прогноза стоимости выплат
- Выбрана лучшая модель по RMSE
- Определены наиболее значимые признаки
"""

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night"], index=0)
        height = st.number_input("Высота слайдов", value=520, step=10)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom"], index=0)
        plugins = st.multiselect("Плагины", ["highlight", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

presentation_page()
