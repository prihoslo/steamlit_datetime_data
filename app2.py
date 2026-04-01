# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import datetime

# Настройка страницы
st.set_page_config(page_title="Анализ временных рядов", layout="wide")

# Заголовок
st.title("📈 Анализ и прогнозирование временных рядов")
st.markdown("---")

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки")

# Загрузка данных
uploaded_file = st.sidebar.file_uploader(
    "Загрузите файл с данными (Excel или CSV)" \
    "ВАЖНО!!!" \
    "файл для загрузки должен быть до агрегации дней в недели",
    type=['xlsx', 'xls', 'csv']
)

# Если файл не загружен, используем пример
if uploaded_file is None:
    st.info("📁 Загрузите файл с данными или используйте пример ниже")
    use_example = st.checkbox("Использовать пример с данными Superstore")
    
    if use_example:
        # Загрузка примера данных
        @st.cache_data
        def load_example_data():
            # Создаем пример данных
            dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='W')
            np.random.seed(42)
            trend = np.linspace(0, 10, len(dates))
            seasonal = 5 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            noise = np.random.normal(0, 2, len(dates))
            sales = 100 + trend + seasonal + noise
            sales = np.maximum(sales, 0)
            
            df = pd.DataFrame({
                'Order Date': dates,
                'Sales': sales,
                'Category': 'Furniture'
            })
            return df
        
        df = load_example_data()
        st.success("✅ Загружены примерные данные Superstore")
    else:
        st.stop()
else:
    # Загрузка пользовательского файла
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    
    df = load_data(uploaded_file)
    st.success(f"✅ Загружен файл: {uploaded_file.name}")

# Показываем первые строки данных
st.subheader("📊 Исходные данные")
st.write(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
st.dataframe(df.head(10))

# Выбор столбцов
st.sidebar.subheader("Выбор столбцов")
date_col = st.sidebar.selectbox(
    "Выберите столбец с датой",
    options=df.columns,
    index=0 if 'Order Date' not in df.columns else df.columns.get_loc('Order Date')
)

value_col = st.sidebar.selectbox(
    "Выберите столбец со значениями",
    options=df.columns,
    index=df.columns.get_loc('Sales') if 'Sales' in df.columns else 1
)

category_col = st.sidebar.selectbox(
    "Выберите столбец с категорией (опционально)",
    options=['Нет'] + list(df.columns),
    index=0
)

# Фильтрация по категории
if category_col != 'Нет':
    categories = ['Все'] + sorted(df[category_col].unique().tolist())
    selected_category = st.sidebar.selectbox("Выберите категорию", categories)
    
    if selected_category != 'Все':
        data = df[df[category_col] == selected_category].copy()
        st.info(f"📌 Выбрана категория: {selected_category}")
    else:
        data = df.copy()
else:
    data = df.copy()

# Подготовка временного ряда
data[date_col] = pd.to_datetime(data[date_col])
data = data.sort_values(date_col)
data = data.set_index(date_col)

# Агрегация по периодам
st.sidebar.subheader("Агрегация данных")
freq_options = {
    'День': 'D',
    'Неделя': 'W',
    'Месяц': 'M',
    'Квартал': 'Q',
    'Полугодие': '6M',
    'Год': 'Y'
}
selected_freq = st.sidebar.selectbox("Период агрегации", list(freq_options.keys()))
agg_method = st.sidebar.radio("Метод агрегации", ['Сумма', 'Среднее'])

# Агрегируем данные
if agg_method == 'Сумма':
    time_data = data[[value_col]].resample(freq_options[selected_freq]).sum()
else:
    time_data = data[[value_col]].resample(freq_options[selected_freq]).mean()

time_data.columns = ['value']
time_data = time_data.dropna()

st.subheader(f"📈 Временной ряд ({selected_freq}, {agg_method.lower()})")
st.write(f"Количество точек: {len(time_data)}")

# Визуализация временного ряда
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_data.index,
    y=time_data['value'],
    mode='lines+markers',
    name='Исходные данные',
    line=dict(color='blue', width=2),
    marker=dict(size=4)
))
fig.update_layout(
    title='Временной ряд',
    xaxis_title='Дата',
    yaxis_title='Значение',
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Разделение на train/test
st.sidebar.subheader("Разделение данных")
test_size = st.sidebar.slider("Размер тестовой выборки (%)", 10, 40, 20)
split_date = time_data.index[int(len(time_data) * (1 - test_size/100))]
data_train = time_data[time_data.index < split_date]
data_test = time_data[time_data.index >= split_date]

st.write(f"📅 Дата разделения: {split_date.date()}")
st.write(f"Обучающая выборка: {len(data_train)} точек")
st.write(f"Тестовая выборка: {len(data_test)} точек")

# Выбор модели
st.sidebar.subheader("Выбор модели")
model_choice = st.sidebar.selectbox(
    "Модель для прогнозирования",
    ["Скользящее среднее (MA)", "ARIMA", "Prophet"]
)

# Настройка параметров в зависимости от выбранной модели
if model_choice == "Скользящее среднее (MA)":
    st.sidebar.subheader("Параметры MA")
    n_window = st.sidebar.slider("Размер окна", 1, 52, 4)
    use_weighted = st.sidebar.checkbox("Использовать взвешенное скользящее среднее")
    
    if use_weighted:
        st.sidebar.write("Веса для последних 5 точек (сумма=1)")
        weights = []
        for i in range(5):
            w = st.sidebar.slider(f"Вес t-{i+1}", 0.0, 1.0, 0.2, key=f"w{i}")
            weights.append(w)
        weights = np.array(weights) / np.sum(weights)
    
elif model_choice == "ARIMA":
    st.sidebar.subheader("Параметры ARIMA")
    
    # Автоматическое определение сезонности
    auto_seasonality = st.sidebar.checkbox("Автоопределение сезонности", value=True)
    
    if not auto_seasonality:
        p = st.sidebar.number_input("p (AR порядок)", 0, 10, 12)
        d = st.sidebar.number_input("d (интеграция)", 0, 2, 1)
        q = st.sidebar.number_input("q (MA порядок)", 0, 10, 12)
        
        use_seasonal = st.sidebar.checkbox("Использовать сезонную компоненту", value=True)
        if use_seasonal:
            P = st.sidebar.number_input("P (сезонный AR)", 0, 5, 1)
            D = st.sidebar.number_input("D (сезонная интеграция)", 0, 2, 1)
            Q = st.sidebar.number_input("Q (сезонный MA)", 0, 5, 1)
            seasonal_period = st.sidebar.number_input("Сезонный период", 1, 53, 52)
        
    # Визуализация ACF/PACF
    if st.sidebar.checkbox("Показать ACF/PACF"):
        st.subheader("Автокорреляционные функции")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(data_train['value'], ax=ax1, lags=min(60, len(data_train)//2))
        plot_pacf(data_train['value'], ax=ax2, lags=min(60, len(data_train)//2))
        st.pyplot(fig)
        
elif model_choice == "Prophet":
    st.sidebar.subheader("Параметры Prophet")
    
    changepoint_prior_scale = st.sidebar.slider(
        "changepoint_prior_scale", 
        0.001, 0.5, 0.05, 
        format="%.3f"
    )
    seasonality_prior_scale = st.sidebar.slider(
        "seasonality_prior_scale", 
        0.1, 20.0, 10.0,
        format="%.1f"
    )
    seasonality_mode = st.sidebar.selectbox(
        "seasonality_mode",
        ["additive", "multiplicative"]
    )
    
    use_holidays = st.sidebar.checkbox("Учитывать праздники", value=False)
    use_monthly = st.sidebar.checkbox("Добавить месячную сезонность", value=True)
    
    forecast_periods = st.sidebar.number_input(
        "Период прогноза (шагов вперед)", 
        1, 100, 52
    )

# Прогнозирование
st.markdown("---")
st.subheader("🎯 Прогнозирование")

if st.button("🚀 Запустить прогнозирование", type="primary"):
    with st.spinner("Выполняется прогнозирование..."):
        
        if model_choice == "Скользящее среднее (MA)":
            if use_weighted:
                # Взвешенное скользящее среднее
                def weighted_moving_average(x, n, weights):
                    weights = np.array(weights)
                    result = x.rolling(n).apply(
                        lambda x: np.dot(x, weights) / weights.sum(), 
                        raw=True
                    ).shift(1)
                    return result
                
                moving_avg = weighted_moving_average(
                    data_train['value'], 
                    n_window, 
                    weights
                )
                pred_train = moving_avg
                pred_test = moving_avg
                model_name = f"Взвешенное MA (окно={n_window})"
            else:
                # Простое скользящее среднее
                moving_avg = data_train['value'].rolling(
                    window=n_window, 
                    closed='left'
                ).mean()
                pred_train = moving_avg
                pred_test = moving_avg
                model_name = f"Скользящее среднее (окно={n_window})"
            
            # Прогноз на тестовую выборку (используем последнее значение)
            pred_test = pd.Series(
                [data_train['value'].iloc[-n_window:].mean()] * len(data_test),
                index=data_test.index
            )
            
            # Метрики
            train_mae = mean_absolute_error(
                data_train['value'][n_window:], 
                pred_train[n_window:]
            )
            test_mae = mean_absolute_error(data_test['value'], pred_test)
            train_r2 = r2_score(
                data_train['value'][n_window:], 
                pred_train[n_window:]
            )
            test_r2 = r2_score(data_test['value'], pred_test)
            
            # Визуализация
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_data.index, y=time_data['value'],
                mode='lines+markers', name='Исходные данные',
                line=dict(color='blue', width=2), marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=pred_train.index, y=pred_train,
                mode='lines', name='Прогноз (обучение)',
                line=dict(color='green', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=pred_test.index, y=pred_test,
                mode='lines', name='Прогноз (тест)',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(
                title=f'{model_name}<br>MAE тест: {test_mae:.2f}, R² тест: {test_r2:.3f}',
                xaxis_title='Дата', yaxis_title='Значение',
                height=500, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "ARIMA":
            # Подготовка данных для ARIMA
            if auto_seasonality:
                # Автоопределение параметров
                from statsmodels.tsa.stattools import adfuller
                
                # Проверка стационарности
                result = adfuller(data_train['value'])
                d = 0 if result[1] < 0.05 else 1
                
                # Определение сезонности
                seasonal_period = None
                for period in [52, 26, 13, 12, 4]:
                    if len(data_train) > period * 2:
                        seasonal_period = period
                        break
                
                P, D, Q = 1, d, 1
                p, q = 12, 12
                
                st.info(f"Автоопределенные параметры: p=12, d={d}, q=12, сезонный период={seasonal_period}")
            else:
                p, d, q = p, d, q
                if use_seasonal:
                    seasonal_period = seasonal_period
                    P, D, Q = P, D, Q
            
            try:
                if use_seasonal:
                    arim = ARIMA(
                        data_train['value'],
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    ).fit()
                else:
                    arim = ARIMA(
                        data_train['value'],
                        order=(p, d, q),
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    ).fit()
                
                # Предсказания
                pred_train = arim.fittedvalues
                forecast = arim.forecast(steps=len(data_test))
                pred_test = pd.Series(forecast, index=data_test.index)
                
                # Метрики
                train_mae = mean_absolute_error(data_train['value'], pred_train)
                test_mae = mean_absolute_error(data_test['value'], pred_test)
                train_r2 = r2_score(data_train['value'], pred_train)
                test_r2 = r2_score(data_test['value'], pred_test)
                
                # Визуализация
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_data.index, y=time_data['value'],
                    mode='lines+markers', name='Исходные данные',
                    line=dict(color='blue', width=2), marker=dict(size=4)
                ))
                fig.add_trace(go.Scatter(
                    x=pred_train.index, y=pred_train,
                    mode='lines', name='Прогноз (обучение)',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=pred_test.index, y=pred_test,
                    mode='lines', name='Прогноз (тест)',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title=f'ARIMA ({p},{d},{q})<br>MAE тест: {test_mae:.2f}, R² тест: {test_r2:.3f}',
                    xaxis_title='Дата', yaxis_title='Значение',
                    height=500, hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Показываем диагностику
                with st.expander("📊 Диагностика модели ARIMA"):
                    st.write(arim.summary().as_text())
                
            except Exception as e:
                st.error(f"Ошибка при обучении ARIMA: {str(e)}")
                st.info("Попробуйте изменить параметры модели")
                
        elif model_choice == "Prophet":
            # Подготовка данных для Prophet
            train_prophet = data_train.reset_index()
            train_prophet.columns = ['ds', 'y']
            
            # Создание модели
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            # Добавление месячной сезонности
            if use_monthly:
                model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )
            
            # Добавление праздников
            if use_holidays:
                russian_holidays = pd.DataFrame({
                    'holiday': 'new_year',
                    'ds': pd.date_range(start='2015-01-01', end='2025-01-01', freq='YS'),
                    'lower_window': -5,
                    'upper_window': 8
                })
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode,
                    holidays=russian_holidays
                )
            
            # Обучение
            model.fit(train_prophet)
            
            # Создание будущих дат
            future = model.make_future_dataframe(
                periods=len(data_test) + forecast_periods,
                freq=selected_freq.lower()[0] if selected_freq != 'Неделя' else 'W'
            )
            
            # Прогноз
            forecast = model.predict(future)
            
            # Разделение на train/test/future
            forecast_train = forecast[forecast['ds'] < split_date]
            forecast_test = forecast[
                (forecast['ds'] >= split_date) & 
                (forecast['ds'] < data_test.index[-1] + pd.Timedelta(days=7))
            ]
            forecast_future = forecast[forecast['ds'] > data_test.index[-1]]
            
            # Метрики
            train_mae = mean_absolute_error(
                train_prophet['y'][:-len(data_test)],
                forecast_train['yhat'][:len(train_prophet)-len(data_test)]
            )
            test_mae = mean_absolute_error(data_test['value'], forecast_test['yhat'])
            train_r2 = r2_score(
                train_prophet['y'][:-len(data_test)],
                forecast_train['yhat'][:len(train_prophet)-len(data_test)]
            )
            test_r2 = r2_score(data_test['value'], forecast_test['yhat'])
            
            # Визуализация с Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_data.index, y=time_data['value'],
                mode='lines+markers', name='Исходные данные',
                line=dict(color='blue', width=2), marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_train['ds'], y=forecast_train['yhat'],
                mode='lines', name='Прогноз (обучение)',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_test['ds'], y=forecast_test['yhat'],
                mode='lines', name='Прогноз (тест)',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'], y=forecast_future['yhat'],
                mode='lines', name='Прогноз (будущее)',
                line=dict(color='purple', width=2, dash='dash')
            ))
            
            # Добавление доверительных интервалов
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_upper'],
                mode='lines', name='Верхняя граница',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat_lower'],
                mode='lines', name='Нижняя граница',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
            
            fig.update_layout(
                title=f'Prophet<br>MAE тест: {test_mae:.2f}, R² тест: {test_r2:.3f}',
                xaxis_title='Дата', yaxis_title='Значение',
                height=500, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Компоненты прогноза
            with st.expander("📈 Компоненты прогноза"):
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
        
        # Показываем метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE (обучение)", f"{train_mae:.2f}")
        with col2:
            st.metric("MAE (тест)", f"{test_mae:.2f}")
        with col3:
            st.metric("R² (обучение)", f"{train_r2:.3f}")
        with col4:
            st.metric("R² (тест)", f"{test_r2:.3f}")
        
        # Интерпретация результатов
        st.markdown("---")
        st.subheader("📝 Интерпретация результатов")
        
        if test_mae < data_test['value'].mean() * 0.1:
            st.success(f"✅ Отличный результат! MAE составляет {test_mae/data_test['value'].mean()*100:.1f}% от среднего значения")
        elif test_mae < data_test['value'].mean() * 0.2:
            st.info(f"👍 Хороший результат. MAE составляет {test_mae/data_test['value'].mean()*100:.1f}% от среднего значения")
        else:
            st.warning(f"⚠️ Результат можно улучшить. MAE составляет {test_mae/data_test['value'].mean()*100:.1f}% от среднего значения")
        
        if test_r2 > 0.8:
            st.success("✅ Модель хорошо объясняет вариацию данных (R² > 0.8)")
        elif test_r2 > 0.5:
            st.info("👍 Модель удовлетворительно объясняет вариацию данных (R² > 0.5)")
        else:
            st.warning("⚠️ Модель плохо объясняет вариацию данных. Попробуйте изменить параметры или выбрать другую модель")

else:
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Запустить прогнозирование'")

# Дополнительная информация в боковой панели
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Информация")
st.sidebar.info(
    """
    **Модели:**
    - **Скользящее среднее**: простой метод прогнозирования
    - **ARIMA**: классическая модель для временных рядов
    - **Prophet**: модель от Facebook с поддержкой сезонности
    
    **Метрики:**
    - **MAE**: средняя абсолютная ошибка
    - **R²**: коэффициент детерминации
    """
)