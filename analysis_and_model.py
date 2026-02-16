import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.datasets._openml import OpenMLError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

# ---- Константы ----
DATA_ID = 42876
TARGET_COL = "UltimateIncurredClaimCost"

DT_CANDIDATES = ["DateTimeOfAccident", "DateReported"]  # в идеале эти два
LOCAL_CSV_PATH = os.path.join("data", "workers_compensation.csv")

# Категориальные, которые ожидаются по методичке (но реально могут отсутствовать)
CATEGORICAL_BASE = ["Gender", "MaritalStatus", "PartTimeFullTime", "ClaimDescription"]

# Числовые, которые масштабируются (берутся по пересечению с реальными колонками)
NUMERICAL_BASE = [
    "Age", "DependentChildren", "DependentsOther",
    "WeeklyPay", "HoursWorkedPerWeek", "DaysWorkedPerWeek",
    "InitialCaseEstimate",
    "AccidentMonth", "AccidentDayOfWeek", "ReportingDelay"
]

# Явные кандидаты на ID
POSSIBLE_ID_COLS = ["ClaimNumber", "ClaimNumberId", "ID", "Id", "CaseId", "CaseID"]

XGB_AVAILABLE = True

# ---- Загрузка данных ----
def _load_local_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Локальный CSV не найден: {path}")
    return pd.read_csv(path)


def load_workers_compensation() -> pd.DataFrame:
    """
    1) Пробуется OpenML fetch_openml(data_id=42876)
    2) Если OpenML недоступен/не находит датасет -> чтение data/workers_compensation.csv
    """
    try:
        data = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto")
        return data.frame.copy()
    except OpenMLError:
        return _load_local_csv(LOCAL_CSV_PATH)
    except Exception:
        return _load_local_csv(LOCAL_CSV_PATH)


# ---- Предобработка ----
def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _drop_id_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # 1) Отбросить явно известные ID
    present = [c for c in POSSIBLE_ID_COLS if c in data.columns]
    if present:
        data = data.drop(columns=present)

    # 2) Отбросить "почти уникальные" строковые столбцы (ID)
    for c in list(data.columns):
        if c == TARGET_COL:
            continue
        if data[c].dtype == "object":
            nunique = data[c].nunique(dropna=True)
            if len(data) > 0 and (nunique / len(data) > 0.95):
                data = data.drop(columns=[c])

    return data


def _add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Если есть оба datetime-столбца: строятся AccidentMonth/AccidentDayOfWeek/ReportingDelay и исходные datetime удаляются.
    Если нет: пропускается.
    """
    out = data.copy()

    if all(col in out.columns for col in DT_CANDIDATES):
        dt_acc = DT_CANDIDATES[0]
        dt_rep = DT_CANDIDATES[1]

        out[dt_acc] = _safe_to_datetime(out[dt_acc])
        out[dt_rep] = _safe_to_datetime(out[dt_rep])

        out["AccidentMonth"] = out[dt_acc].dt.month
        out["AccidentDayOfWeek"] = out[dt_acc].dt.dayofweek
        out["ReportingDelay"] = (out[dt_rep] - out[dt_acc]).dt.days

        out = out.drop(columns=[dt_acc, dt_rep], errors="ignore")

        for c in ["AccidentMonth", "AccidentDayOfWeek", "ReportingDelay"]:
            if c in out.columns:
                if out[c].isnull().any():
                    out[c] = out[c].fillna(out[c].median())

    return out


def preprocess_fit_transform(df_raw: pd.DataFrame):
    """
    Возвращает:
      X_scaled, y,
      encoders, scaler,
      used_categorical_cols,
      feature_columns (порядок колонок X)
    """
    if TARGET_COL not in df_raw.columns:
        raise ValueError(f"Не найдена целевая переменная: {TARGET_COL}")

    data = df_raw.copy()

    data = _drop_id_like_columns(data)

    data = _add_datetime_features(data)

    # y
    y = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.drop(columns=[TARGET_COL])

    valid_mask = ~y.isna()
    data = data.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    categorical_present = [c for c in CATEGORICAL_BASE if c in data.columns]
    object_cols = [c for c in data.columns if data[c].dtype == "object"]
    for c in object_cols:
        if c not in categorical_present:
            categorical_present.append(c)

    encoders = {}
    for col in categorical_present:
        le = LabelEncoder()
        data[col] = data[col].astype(str).fillna("Unknown")
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    leftovers = [c for c in data.columns if data[c].dtype == "object"]
    if leftovers:
        data = data.drop(columns=leftovers)

    # X
    X = data.copy()

    scaler = StandardScaler()
    X_scaled = X.copy()

    numeric_present = [c for c in NUMERICAL_BASE if c in X_scaled.columns]
    if numeric_present:
        X_scaled[numeric_present] = scaler.fit_transform(X_scaled[numeric_present])
    else:
        scaler = StandardScaler()

    feature_columns = list(X_scaled.columns)

    return X_scaled, y, encoders, scaler, categorical_present, feature_columns


def preprocess_transform_for_inference(input_df: pd.DataFrame, encoders: dict, scaler: StandardScaler,
                                      used_categorical_cols: list, feature_columns: list) -> pd.DataFrame:
    """
    Преобразования для одного/нескольких новых случаев.
    - datetime -> фичи (если колонки есть)
    - категориальные -> по обученным encoders, неизвестные -> fallback
    - масштабирование числовых (по NUMERICAL_BASE)
    - подгонка колонок под feature_columns (добавление отсутствующих, порядок)
    """
    data = input_df.copy()

    data = _add_datetime_features(data)

    for col in used_categorical_cols:
        if col not in data.columns:
            data[col] = "Unknown"

        le = encoders[col]
        vals = data[col].astype(str).fillna("Unknown").tolist()

        known = set(le.classes_.tolist())
        fallback = le.classes_[0] if len(le.classes_) > 0 else "Unknown"
        vals_safe = [v if v in known else fallback for v in vals]
        data[col] = le.transform(vals_safe)

    leftovers = [c for c in data.columns if data[c].dtype == "object"]
    if leftovers:
        data = data.drop(columns=leftovers)

    X = data.copy()
    numeric_present = [c for c in NUMERICAL_BASE if c in X.columns]
    if numeric_present:
        X[numeric_present] = scaler.transform(X[numeric_present])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = 0

    X = X[feature_columns]

    return X


# ---- Метрики/графики ----
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2), "y_pred": y_pred}


def plot_pred_vs_true(y_true, y_pred, title: str):
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.25)
    mn = float(min(y_true.min(), np.min(y_pred)))
    mx = float(max(y_true.max(), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    plt.xlabel("Реальные значения")
    plt.ylabel("Предсказанные значения")
    plt.title(title)
    plt.tight_layout()
    return fig

# ---- Streamlit page ----
def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")

    with st.sidebar:
        st.header("Настройки")
        test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("random_state", min_value=0, value=42, step=1)

        st.divider()
        st.subheader("Модели")
        use_linear = st.checkbox("Linear Regression", value=True)
        use_ridge = st.checkbox("Ridge Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_xgb = st.checkbox("XGBoost", value=True, disabled=not XGB_AVAILABLE)

        st.divider()
        st.subheader("Random Forest параметры")
        rf_n_estimators = st.number_input("n_estimators", min_value=50, max_value=500, value=200, step=50)

        st.subheader("XGBoost параметры")
        xgb_n_estimators = st.number_input("xgb n_estimators", min_value=50, max_value=500, value=200, step=50)
        xgb_lr = st.slider("xgb learning_rate", 0.01, 0.3, 0.1, 0.01)

        st.divider()

    colA, colB = st.columns([1.2, 0.8], gap="large")

    with colA:
        st.subheader("Загрузка данных")
        if st.button("Загрузить данные", type="primary"):
            with st.spinner("Загрузка данных..."):
                df = load_workers_compensation()
                st.session_state["df_raw"] = df
            st.success(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")

        if "df_raw" in st.session_state:
            df_raw = st.session_state["df_raw"]
            st.write(df_raw.head(10))

            st.subheader("Проверка:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Пропуски (всего)", int(df_raw.isnull().sum().sum()))
            with c2:
                st.metric("Строк", int(df_raw.shape[0]))
            with c3:
                st.metric("Столбцов", int(df_raw.shape[1]))

    with colB:
        st.subheader("Ориентиры по признакам")
        st.markdown(
            f"""
- Целевая переменная: `{TARGET_COL}`
- Datetime-колонки: `{DT_CANDIDATES[0]}`, `{DT_CANDIDATES[1]}`
- Категориальные: {", ".join([f"`{c}`" for c in CATEGORICAL_BASE])}
- Числовые: {", ".join([f"`{c}`" for c in NUMERICAL_BASE])}
"""
        )

    st.divider()
    st.subheader("Предобработка и обучение")

    if "df_raw" not in st.session_state:
        st.info("Для продолжения требуется загрузить данные.")
        return

    df_raw = st.session_state["df_raw"]

    if st.button("Выполнить предобработку и обучить модели"):
        with st.spinner("Предобработка..."):
            X_all, y_all, encoders, scaler, used_cat_cols, feature_cols = preprocess_fit_transform(df_raw)

        st.session_state["X_all"] = X_all
        st.session_state["y_all"] = y_all
        st.session_state["encoders"] = encoders
        st.session_state["scaler"] = scaler
        st.session_state["used_cat_cols"] = used_cat_cols
        st.session_state["feature_cols"] = feature_cols

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=float(test_size), random_state=int(random_state)
        )
        st.session_state["split"] = (X_train, X_test, y_train, y_test)

        # models
        models = {}
        if use_linear:
            models["Linear Regression"] = LinearRegression()
        if use_ridge:
            models["Ridge Regression"] = Ridge(alpha=1.0, random_state=int(random_state))
        if use_rf:
            models["Random Forest"] = RandomForestRegressor(
                n_estimators=int(rf_n_estimators),
                random_state=int(random_state),
                n_jobs=-1
            )
        if use_xgb and XGB_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=int(xgb_n_estimators),
                learning_rate=float(xgb_lr),
                random_state=int(random_state),
                n_jobs=-1
            )

        results = {}
        trained = {}

        with st.spinner("Обучение моделей..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                results[name] = {k: metrics[k] for k in ["MAE", "MSE", "RMSE", "R2"]}
                trained[name] = model

        st.session_state["results"] = results
        st.session_state["trained_models"] = trained

        best_name = min(results.keys(), key=lambda k: results[k]["RMSE"]) if results else None
        st.session_state["best_model_name"] = best_name

        st.success("Готово: предобработка выполнена, модели обучены.")

    # результаты
    if "results" in st.session_state and "split" in st.session_state:
        results = st.session_state["results"]
        X_train, X_test, y_train, y_test = st.session_state["split"]

        st.subheader("Сравнение моделей")
        res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(res_df, use_container_width=True)

        best_name = st.session_state.get("best_model_name")
        if best_name:
            st.info(f"Лучшая модель по RMSE: {best_name}")

        st.divider()
        st.subheader("Графики качества")
        trained = st.session_state["trained_models"]

        to_plot = list(trained.keys())[:2]
        cols = st.columns(len(to_plot)) if to_plot else []
        for i, name in enumerate(to_plot):
            model = trained[name]
            y_pred = model.predict(X_test)
            fig = plot_pred_vs_true(y_test, y_pred, f"{name}: предсказания vs реальные")
            cols[i].pyplot(fig, clear_figure=True)

        st.divider()
        st.subheader("Важность признаков (Random Forest)")
        if "Random Forest" in trained:
            rf = trained["Random Forest"]
            importances = rf.feature_importances_
            fi = pd.DataFrame({"feature": X_train.columns, "importance": importances}).sort_values(
                "importance", ascending=False
            )
            st.dataframe(fi.head(20), use_container_width=True)

            top_n = 10
            fig2 = plt.figure(figsize=(8, 5))
            plt.barh(fi["feature"].head(top_n)[::-1], fi["importance"].head(top_n)[::-1])
            plt.xlabel("Важность")
            plt.title("Топ-10 наиболее важных признаков")
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=True)
        else:
            st.caption("Random Forest не обучался, поэтому важность признаков не отображается.")

        st.divider()
        st.subheader("Предсказание стоимости для нового случая")

        best_name = st.session_state.get("best_model_name")
        trained = st.session_state.get("trained_models", {})
        encoders = st.session_state.get("encoders")
        scaler = st.session_state.get("scaler")
        used_cat_cols = st.session_state.get("used_cat_cols", [])
        feature_cols = st.session_state.get("feature_cols", [])

        if not best_name:
            st.warning("Нет обученной модели для предсказания.")
            return

        model = trained[best_name]

        with st.form("prediction_form"):
            st.write("Ввод параметров случая:")

            date_accident = st.date_input("Дата несчастного случая")
            date_reported = st.date_input("Дата сообщения о случае")

            age = st.number_input("Возраст работника", min_value=0, value=35, step=1)
            weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=500, step=10)
            initial_est = st.number_input("Начальная оценка стоимости случая ($)", min_value=0, value=5000, step=100)

            gender = st.selectbox("Пол работника", ["M", "F"])
            marital = st.text_input("Семейное положение", value="Single")
            ptft = st.selectbox("Тип занятости", ["Part Time", "Full Time"])
            claim_desc = st.text_input("Описание заявки", value="Other")

            dep_children = st.number_input("Дети на иждивении", min_value=0, value=0, step=1)
            dep_other = st.number_input("Другие иждивенцы", min_value=0, value=0, step=1)
            hours_per_week = st.number_input("Часов работы в неделю", min_value=0, value=40, step=1)
            days_per_week = st.number_input("Дней работы в неделю", min_value=0, value=5, step=1)

            submitted = st.form_submit_button("Предсказать")

        if submitted:
            if encoders is None or scaler is None or not feature_cols:
                st.error("Нет данных для предсказания: требуется обучить модели.")
                return

            row = {
                "DateTimeOfAccident": pd.to_datetime(str(date_accident)),
                "DateReported": pd.to_datetime(str(date_reported)),
                "Age": int(age),
                "WeeklyPay": int(weekly_pay),
                "InitialCaseEstimate": float(initial_est),

                "Gender": str(gender),
                "MaritalStatus": str(marital),
                "PartTimeFullTime": str(ptft),
                "ClaimDescription": str(claim_desc),

                "DependentChildren": int(dep_children),
                "DependentsOther": int(dep_other),
                "HoursWorkedPerWeek": int(hours_per_week),
                "DaysWorkedPerWeek": int(days_per_week),
            }

            inf_df = pd.DataFrame([row])

            X_inf = preprocess_transform_for_inference(
                inf_df,
                encoders=encoders,
                scaler=scaler,
                used_categorical_cols=used_cat_cols,
                feature_columns=feature_cols
            )

            pred = float(model.predict(X_inf)[0])
            st.success(f"Предсказанная итоговая стоимость возмещения: ${pred:,.2f}")


analysis_and_model_page()
