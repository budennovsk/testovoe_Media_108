import streamlit as st
import pandas as pd
import catboost
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast


class CatBoostModel:
    def __init__(self, model_path, features):
        """
        Инициализация модели.
        :param model_path: Путь к сохраненной модели.
        :param features: Список признаков для предсказания.
        """
        self.model = catboost.CatBoostClassifier()
        self.model.load_model(model_path)
        self.features = features

    def load_data(self, uploaded_file):
        """
        Загрузка данных из файла.
        :param uploaded_file: Файл с данными.
        :return: DataFrame с данными.
        """
        df = pd.read_csv(uploaded_file, sep='\t')
        return df

    def generate_new_features(self, df):
        """
        Генерация новых признаков, таких как синус и косинус для времени.
        :param df: DataFrame с данными.
        :return: DataFrame с новыми признаками.
        """
        # Генерация новых признаков для времени
        df['ym:s:dateTimeUTC'] = pd.to_datetime(df['ym:s:dateTimeUTC'])
        df['hour'] = df['ym:s:dateTimeUTC'].dt.hour
        df['minute'] = df['ym:s:dateTimeUTC'].dt.minute

        # Преобразуем час и минуту в синус и косинус
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Перебираем все колонки и находим те, которые содержат массивы
        for column in df.columns:
            if column == "ym:s:goalsID":
                continue
            if df[column].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).all():
                new_column_name = f"{column}_length"
                df[new_column_name] = df[column].apply(self.get_array_length)

        return df

    def get_array_length(self, arr_str):
        """
        Функция для извлечения длины массива.
        :param arr_str: Строка, представляющая массив.
        :return: Длина массива.
        """
        try:
            arr = ast.literal_eval(arr_str)
            return len(arr)
        except:
            return 0

    def preprocess_data(self, df):
        """
        Препроцессинг данных перед подачей на модель.
        :param df: DataFrame с данными.
        :return: Преобразованные данные.
        """
        # Преобразуем целевой столбец в бинарный формат
        df['target'] = df['ym:s:goalsID'].apply(lambda x: 1 if '333755569' in str(x) else 0)

        # Разделяем данные на признаки и цель
        X = df[self.features]
        y = df['target']

        # Обработка категориальных признаков
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        X[cat_features] = X[cat_features].fillna("missing")

        # Применяем CatBoostEncoder для категориальных признаков
        encoder = ce.CatBoostEncoder(cols=cat_features)
        X_encoded = encoder.fit_transform(X, y)

        # Заполняем пропущенные значения для числовых данных медианой
        imputer_num = SimpleImputer(strategy='median')
        X_encoded[X_encoded.select_dtypes(include=['number']).columns] = imputer_num.fit_transform(
            X_encoded.select_dtypes(include=['number']))

        return X_encoded, y

    def predict(self, X):
        """
        Предсказание с использованием модели.
        :param X: Признаки для предсказания.
        :return: Предсказания модели.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Получение вероятностей для предсказания.
        :param X: Признаки для предсказания.
        :return: Вероятности для каждого класса.
        """
        return self.model.predict_proba(X)[:, 1]  # Получаем вероятность для положительного класса

    def compute_metrics(self, y_true, y_pred, y_prob):
        """
        Вычисление метрик модели.
        :param y_true: Истинные значения.
        :param y_pred: Предсказанные значения.
        :param y_prob: Вероятности предсказанных значений.
        :return: Словарь с метриками.
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)



        # F-beta score (например, beta = 0.5)
        beta_fscore = fbeta_score(y_true, y_pred, beta=0.5)

        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_prob)

        # PR AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_vals, precision_vals)

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,

            "F-beta Score (Beta=0.5)": beta_fscore,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc
        }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Построение матрицы ошибок.
        :param y_true: Истинные значения.
        :param y_pred: Предсказанные значения.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Предсказание')
        plt.ylabel('Истинные значения')
        st.pyplot(fig)


class StreamlitApp:
    def __init__(self, model):
        """
        Инициализация приложения.
        :param model: Экземпляр модели.
        """
        self.model = model

    def run(self):
        """
        Запуск приложения Streamlit.
        """
        st.title("Прогнозирование с использованием CatBoost")
        st.write("Загрузите данные для предсказания:")

        # Загружаем файл
        uploaded_file = st.file_uploader("Выберите файл", type=["tsv"])

        if uploaded_file:
            # Загрузка и предобработка данных
            df = self.model.load_data(uploaded_file)
            df = self.model.generate_new_features(df)
            X_encoded, y = self.model.preprocess_data(df)

            # Прогнозирование и вычисление метрик
            predictions = self.model.predict(X_encoded)
            predicted_probabilities = self.model.predict_proba(X_encoded)

            # Вывод Confusion Matrix
            self.model.plot_confusion_matrix(y, predictions)

            # Вывод метрик
            metrics = self.model.compute_metrics(y, predictions, predicted_probabilities)
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.4f}")


# Инициализация и запуск приложения
model_path = 'catboost_model_1.cbm'
features = [
    'ym:s:automaticAdvEngine', 'ym:s:automaticTrafficSource', 'ym:s:bounce',
    'ym:s:counterUserIDHash', 'ym:s:cross_device_firstAdvEngine', 'ym:s:cross_device_firstTrafficSource',
    'ym:s:cross_device_last_significantAdvEngine', 'ym:s:cross_device_last_significantTrafficSource',
    'ym:s:cross_device_last_yandex_direct_clickAdvEngine', 'ym:s:cross_device_last_yandex_direct_clickTrafficSource',
    'ym:s:cross_device_lastAdvEngine', 'ym:s:cross_device_lastTrafficSource', 'ym:s:endURL',
    'ym:s:firstAdvEngine', 'ym:s:firstTrafficSource', 'ym:s:goalsCurrency', 'ym:s:goalsDateTime',
    'ym:s:goalsOrder', 'ym:s:goalsPrice', 'ym:s:goalsSerialNumber', 'ym:s:ipAddress', 'ym:s:isNewUser',
    'ym:s:last_yandex_direct_clickAdvEngine', 'ym:s:last_yandex_direct_clickTrafficSource',
    'ym:s:lastAdvEngine', 'ym:s:lastsignAdvEngine', 'ym:s:lastsignTrafficSource', 'ym:s:lastTrafficSource',
    'ym:s:pageViews', 'ym:s:regionCity', 'ym:s:regionCityID', 'ym:s:regionCountry', 'ym:s:regionCountryID',
    'ym:s:startURL', 'ym:s:visitDuration', 'ym:s:visitID', 'ym:s:watchIDs', 'ym:s:cross_device_lastReferalSource',
    'ym:s:lastReferalSource', 'ym:s:automaticDirectClickOrderName', 'ym:s:cross_device_firstDirectClickOrderName',
    'ym:s:cross_device_last_significantDirectClickOrderName',
    'ym:s:cross_device_last_yandex_direct_clickDirectClickOrderName',
    'ym:s:cross_device_lastDirectBannerGroup', 'ym:s:cross_device_lastDirectClickBanner',
    'ym:s:cross_device_lastDirectClickOrderName', 'ym:s:firstDirectClickOrderName',
    'ym:s:last_yandex_direct_clickDirectClickOrderName', 'ym:s:lastDirectBannerGroup', 'ym:s:lastDirectClickBanner',
    'ym:s:lastDirectClickOrderName', 'ym:s:lastsignDirectClickOrderName', 'ym:s:cross_device_lastDirectClickBannerName',
    'ym:s:lastDirectClickBannerName', 'ym:s:automatichasGCLID', 'ym:s:cross_device_firsthasGCLID',
    'ym:s:cross_device_last_significanthasGCLID', 'ym:s:cross_device_last_yandex_direct_clickhasGCLID',
    'ym:s:cross_device_lasthasGCLID', 'ym:s:firsthasGCLID', 'ym:s:last_yandex_direct_clickhasGCLID',
    'ym:s:lasthasGCLID', 'ym:s:lastsignhasGCLID', 'ym:s:browser', 'ym:s:browserEngine', 'ym:s:browserEngineVersion1',
    'ym:s:browserEngineVersion2', 'ym:s:browserEngineVersion3', 'ym:s:browserEngineVersion4', 'ym:s:browserLanguage',
    'ym:s:browserMajorVersion', 'ym:s:browserMinorVersion', 'ym:s:clientTimeZone', 'ym:s:cookieEnabled',
    'ym:s:deviceCategory', 'ym:s:javascriptEnabled', 'ym:s:operatingSystem', 'ym:s:operatingSystemRoot', 'ym:s:referer',
    'ym:s:screenColors', 'ym:s:screenFormat', 'ym:s:physicalScreenHeight', 'ym:s:physicalScreenWidth',
    'ym:s:screenHeight',
    'ym:s:screenOrientation', 'ym:s:screenWidth', 'ym:s:windowClientHeight', 'ym:s:windowClientWidth',
    'ym:s:offlineCallFirstTimeCaller',
    'ym:s:offlineCallHoldDuration', 'ym:s:offlineCallMissed', 'ym:s:offlineCallTag', 'ym:s:offlineCallTalkDuration',
    'ym:s:offlineCallURL', 'hour', 'minute', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'ym:s:goalsCurrency_length',
    'ym:s:goalsDateTime_length', 'ym:s:goalsOrder_length', 'ym:s:goalsPrice_length', 'ym:s:goalsSerialNumber_length',
    'ym:s:watchIDs_length', 'ym:s:offlineCallFirstTimeCaller_length', 'ym:s:offlineCallHoldDuration_length',
    'ym:s:offlineCallMissed_length', 'ym:s:offlineCallTag_length', 'ym:s:offlineCallTalkDuration_length',
    'ym:s:offlineCallURL_length'
]

model = CatBoostModel(model_path, features)
app = StreamlitApp(model)
app.run()
