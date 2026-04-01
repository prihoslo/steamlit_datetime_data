import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import io
import chardet

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="Анализ временных рядов", layout="wide")
st.title("📈 Анализ и прогнозирование временных рядов")
st.markdown("---")

# Функция для определения кодировки CSV файла
def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

# Функция для загрузки данных
def load_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            # Определяем кодировку для CSV
            encoding = detect_encoding(uploaded_file)
            df = pd.read_csv(uploaded_file, encoding=encoding)
        elif file_extension in ['xlsx', 'xls']:
            # Для Excel файлов
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"Неподдерживаемый формат файла: {file_extension}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {str(e)}")
        st.info("Попробуйте сохранить файл в формате CSV (UTF-8) и загрузить заново")
        return None

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите файл с данными",
        type=['xlsx', 'xls', 'csv'],
        help="Поддерживаются форматы: XLSX, XLS, CSV"
    )
    
    st.markdown("---")
    
    # Инициализация переменных
    df_preview = None
    date_column = None
    sales_column = None
    category_column = 'Без фильтрации'
    selected_category = None
    
    # Настройки модели (будут доступны только после загрузки файла)
    if uploaded_file is not None:
        # Загрузка данных для предварительного просмотра
        try:
            df_preview = load_data(uploaded_file)
            
            if df_preview is not None and not df_preview.empty:
                st.success(f"✅ Файл загружен: {len(df_preview)} строк")
                
                st.write("**Предварительный просмотр данных:**")
                st.dataframe(df_preview.head(5))
                
                # Выбор колонок
                date_column = st.selectbox(
                    "Колонка с датами",
                    options=df_preview.columns,
                    help="Выберите колонку, содержащую даты"
                )
                
                # Попытка автоматического определения числовых колонок
                numeric_columns = df_preview.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) > 0:
                    default_sales = numeric_columns[0] if numeric_columns else df_preview.columns[1]
                else:
                    default_sales = df_preview.columns[1] if len(df_preview.columns) > 1 else df_preview.columns[0]
                
                sales_column = st.selectbox(
                    "Колонка с целевой переменной (продажи)",
                    options=df_preview.columns,
                    index=list(df_preview.columns).index(default_sales) if default_sales in df_preview.columns else 0,
                    help="Выберите колонку с числовыми значениями для прогнозирования"
                )
                
                # Проверка, что выбранная колонка числовая
                if sales_column not in numeric_columns:
                    st.warning(f"⚠️ Выбранная колонка '{sales_column}' не является числовой. Это может вызвать ошибки.")
                
                # Выбор категории (опционально)
                object_columns = df_preview.select_dtypes(include=['object']).columns.tolist()
                if len(object_columns) > 0:
                    category_column = st.selectbox(
                        "Колонка с категорией (опционально)",
                        options=['Без фильтрации'] + object_columns
                    )
                    
                    if category_column != 'Без фильтрации':
                        categories = df_preview[category_column].dropna().unique()
                        if len(categories) > 0:
                            selected_category = st.selectbox(
                                "Выберите категорию",
                                options=categories
                            )
            else:
                st.error("Не удалось загрузить данные из файла")
                
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")
            df_preview = None
    
    st.markdown("---")
    
    # Параметры модели (доступны только если загружены данные)
    if df_preview is not None and date_column and sales_column:
        st.subheader("🎯 Параметры прогнозирования")
        
        # Процент тестовых данных
        test_size_percent = st.slider(
            "Процент тестовых данных",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Процент данных, который будет использован для тестирования модели"
        )
        
        # Выбор модели
        model_choice = st.selectbox(
            "Выберите модель",
            options=["ARIMA", "Prophet"],
            help="ARIMA - классическая статистическая модель, Prophet - модель от Facebook"
        )
        
        # Горизонт прогнозирования
        forecast_weeks = st.number_input(
            "Количество недель для прогноза",
            min_value=1,
            max_value=104,
            value=52,
            step=4,
            help="На сколько недель вперед сделать прогноз"
        )
        
        st.markdown("---")
        
        # Кнопка запуска
        run_button = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)
    else:
        run_button = False
        if df_preview is None:
            st.info("👈 Сначала загрузите файл с данными")
        elif date_column is None or sales_column is None:
            st.info("👈 Выберите колонки с датами и целевой переменной")

# Основная область приложения
if run_button and uploaded_file is not None and date_column and sales_column:
    try:
        # Загрузка данных
        with st.spinner("Загрузка и обработка данных..."):
            df = load_data(uploaded_file)
            
            if df is None or df.empty:
                st.error("Не удалось загрузить данные")
                st.stop()
            
            # Преобразование колонки с датами
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                # Удаляем строки с некорректными датами
                df = df.dropna(subset=[date_column])
            except Exception as e:
                st.error(f"Ошибка при преобразовании дат: {str(e)}")
                st.stop()
            
            # Проверка числовой колонки
            if not pd.api.types.is_numeric_dtype(df[sales_column]):
                st.warning(f"Колонка '{sales_column}' не является числовой. Попытка преобразования...")
                try:
                    df[sales_column] = pd.to_numeric(df[sales_column], errors='coerce')
                    df = df.dropna(subset=[sales_column])
                except:
                    st.error(f"Не удалось преобразовать колонку '{sales_column}' в числовой формат")
                    st.stop()
            
            # Фильтрация по категории
            if category_column != 'Без фильтрации' and selected_category:
                df = df[df[category_column] == selected_category]
                st.info(f"📌 Данные отфильтрованы по категории: **{selected_category}**")
                if len(df) == 0:
                    st.error("Нет данных для выбранной категории")
                    st.stop()
            
            # Создание временного ряда
            df = df.set_index(date_column)
            df = df[[sales_column]].sort_index()
            
            # Проверка на дубликаты дат
            if df.index.duplicated().any():
                st.warning("Обнаружены дубликаты дат. Выполняется агрегация...")
                df = df.groupby(df.index).sum()
            
            # Ресемплинг по неделям
            weekly_data = df[sales_column].resample('W').sum()
            weekly_data = weekly_data.dropna()
            
            if len(weekly_data) < 10:
                st.error(f"Недостаточно данных для анализа. Получено только {len(weekly_data)} недель")
                st.stop()
            
            st.success(f"✅ Данные успешно загружены. Всего недель: {len(weekly_data)}")
            st.info(f"📅 Период данных: с {weekly_data.index[0].strftime('%Y-%m-%d')} по {weekly_data.index[-1].strftime('%Y-%m-%d')}")
        
        # Разделение на train/test
        with st.spinner("Подготовка данных..."):
            train_size = int(len(weekly_data) * (1 - test_size_percent / 100))
            
            # Убеждаемся, что train_size не слишком мал
            train_size = max(train_size, 10)
            
            train_data = weekly_data[:train_size]
            test_data = weekly_data[train_size:]
            
            st.info(f"📊 Обучающая выборка: {len(train_data)} недель")
            st.info(f"📊 Тестовая выборка: {len(test_data)} недель")
        
        # Прогнозирование
        if model_choice == "ARIMA":
            with st.spinner("Обучение модели ARIMA..."):
                # ARIMA модель с параметрами из кода
                try:
                    arima_model = ARIMA(
                        train_data,
                        order=(12, 1, 12),
                        seasonal_order=(1, 1, 1, 52),
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    ).fit()
                    
                    # Прогноз на тестовую выборку + будущий период
                    forecast_steps = len(test_data) + forecast_weeks
                    forecast = arima_model.forecast(steps=forecast_steps)
                    
                    forecast_test = forecast[:len(test_data)]
                    forecast_future = forecast[len(test_data):]
                    
                    # Метрики
                    train_pred = arima_model.fittedvalues
                    train_mae = mean_absolute_error(train_data, train_pred)
                    train_r2 = r2_score(train_data, train_pred)
                    
                    test_mae = mean_absolute_error(test_data, forecast_test)
                    test_r2 = r2_score(test_data, forecast_test)
                    
                    # Создание индексов для будущего периода
                    last_date = weekly_data.index[-1]
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=7),
                        periods=forecast_weeks,
                        freq='W'
                    )
                    
                except Exception as e:
                    st.error(f"Ошибка при обучении ARIMA модели: {str(e)}")
                    st.stop()
                
        else:  # Prophet
            with st.spinner("Обучение модели Prophet..."):
                try:
                    # Подготовка данных для Prophet
                    train_df = pd.DataFrame({
                        'ds': train_data.index,
                        'y': train_data.values
                    })
                    
                    # Инициализация Prophet с параметрами из кода
                    model = Prophet(
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10,
                        holidays_prior_scale=10,
                        seasonality_mode='additive',
                        yearly_seasonality=True,
                        weekly_seasonality=True
                    )
                    
                    # Добавление месячной сезонности
                    model.add_seasonality(
                        name='monthly',
                        period=30.5,
                        fourier_order=5
                    )
                    
                    model.fit(train_df)
                    
                    # Прогноз на тестовую + будущий период
                    total_forecast_steps = len(test_data) + forecast_weeks
                    future = model.make_future_dataframe(
                        periods=total_forecast_steps,
                        freq='W'
                    )
                    
                    forecast = model.predict(future)
                    
                    # Разделение прогноза
                    forecast_train = forecast.iloc[:len(train_data)]
                    forecast_test = forecast.iloc[len(train_data):len(train_data)+len(test_data)]
                    forecast_future = forecast.iloc[len(train_data)+len(test_data):]
                    
                    # Метрики
                    train_mae = mean_absolute_error(train_data.values, forecast_train['yhat'].values)
                    train_r2 = r2_score(train_data.values, forecast_train['yhat'].values)
                    
                    test_mae = mean_absolute_error(test_data.values, forecast_test['yhat'].values)
                    test_r2 = r2_score(test_data.values, forecast_test['yhat'].values)
                    
                    future_dates = forecast_future['ds'].values
                    
                except Exception as e:
                    st.error(f"Ошибка при обучении Prophet модели: {str(e)}")
                    st.stop()
        
        # Отображение результатов
        st.markdown("---")
        st.header("📊 Результаты анализа")
        
        # Метрики в три колонки
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "MAE на обучающей выборке",
                f"{train_mae:.2f}",
                help="Средняя абсолютная ошибка на обучающих данных"
            )
        
        with col2:
            delta_percent = ((test_mae - train_mae) / train_mae * 100) if train_mae > 0 else 0
            st.metric(
                "MAE на тестовой выборке",
                f"{test_mae:.2f}",
                delta=f"{delta_percent:.1f}%",
                delta_color="inverse",
                help="Средняя абсолютная ошибка на тестовых данных"
            )
        
        with col3:
            st.metric(
                "R² на тестовой выборке",
                f"{test_r2:.3f}",
                help="Коэффициент детерминации (чем ближе к 1, тем лучше)"
            )
        
        # Визуализация
        st.subheader("📈 График прогнозирования")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Отображение данных
        ax.plot(weekly_data.index, weekly_data.values, 
                label='Фактические данные', color='blue', linewidth=1.5, alpha=0.7, marker='o', markersize=3)
        
        # Вертикальная линия разделения
        split_date = train_data.index[-1] if model_choice == "ARIMA" else train_df['ds'].iloc[-1]
        ax.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, label='Разделение train/test')
        
        # Отображение прогноза
        if model_choice == "ARIMA":
            ax.plot(test_data.index, forecast_test, 
                    label=f'Прогноз на тестовую выборку (MAE={test_mae:.2f})', 
                    color='green', linewidth=2, linestyle='--', marker='s', markersize=3)
            ax.plot(future_dates, forecast_future, 
                    label=f'Прогноз на {forecast_weeks} недель вперед', 
                    color='red', linewidth=2, linestyle=':', marker='^', markersize=3)
        else:
            ax.plot(forecast_test['ds'], forecast_test['yhat'], 
                    label=f'Прогноз на тестовую выборку (MAE={test_mae:.2f})', 
                    color='green', linewidth=2, linestyle='--', marker='s', markersize=3)
            ax.plot(forecast_future['ds'], forecast_future['yhat'], 
                    label=f'Прогноз на {forecast_weeks} недель вперед', 
                    color='red', linewidth=2, linestyle=':', marker='^', markersize=3)
            
            # Доверительный интервал для Prophet
            if model_choice == "Prophet":
                ax.fill_between(forecast_future['ds'], 
                                forecast_future['yhat_lower'], 
                                forecast_future['yhat_upper'], 
                                color='red', alpha=0.2, label='Доверительный интервал (80%)')
        
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Продажи', fontsize=12)
        ax.set_title(f'Прогнозирование временного ряда (Модель: {model_choice})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Интерпретация результатов
        st.subheader("📝 Интерпретация результатов")
        
        if test_r2 > 0.8:
            r2_quality = "отличное"
            r2_color = "green"
        elif test_r2 > 0.6:
            r2_quality = "хорошее"
            r2_color = "lightgreen"
        elif test_r2 > 0.4:
            r2_quality = "удовлетворительное"
            r2_color = "orange"
        else:
            r2_quality = "слабое"
            r2_color = "red"
        
        mean_test = np.mean(test_data) if len(test_data) > 0 else 1
        if test_mae / mean_test < 0.1:
            error_quality = "очень низкая"
            error_color = "green"
        elif test_mae / mean_test < 0.2:
            error_quality = "низкая"
            error_color = "lightgreen"
        elif test_mae / mean_test < 0.3:
            error_quality = "средняя"
            error_color = "orange"
        else:
            error_quality = "высокая"
            error_color = "red"
        
        st.markdown(f"""
        **Качество модели:**
        - Коэффициент детерминации R² = **{test_r2:.3f}** - это <span style='color:{r2_color}'>{r2_quality}</span> качество модели
        - Средняя абсолютная ошибка MAE = **{test_mae:.2f}** - это <span style='color:{error_color}'>{error_quality}</span> ошибка
        - Модель объясняет **{test_r2*100:.1f}%** вариации данных
        
        **Выводы:**
        - Модель **{model_choice}** показывает **{r2_quality}** способность прогнозировать продажи
        - Прогноз на {forecast_weeks} недель вперед имеет среднюю ошибку {test_mae:.2f}
        - Средние продажи за период: **{mean_test:.2f}**
        - Относительная ошибка прогноза: **{(test_mae / mean_test * 100):.1f}%**
        """, unsafe_allow_html=True)
        
        # Таблица с прогнозом
        st.subheader(f"🔮 Прогноз на {forecast_weeks} недель вперед")
        
        if model_choice == "ARIMA":
            forecast_df = pd.DataFrame({
                'Дата': future_dates,
                'Прогнозное значение': forecast_future
            })
        else:
            forecast_df = pd.DataFrame({
                'Дата': future_dates,
                'Прогнозное значение': forecast_future['yhat'].values,
                'Нижняя граница (80%)': forecast_future['yhat_lower'].values,
                'Верхняя граница (80%)': forecast_future['yhat_upper'].values
            })
        
        # Форматирование чисел
        for col in forecast_df.columns:
            if col != 'Дата':
                forecast_df[col] = forecast_df[col].round(2)
        
        st.dataframe(forecast_df, use_container_width=True)
        
        # Кнопка для скачивания прогноза
        csv = forecast_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Скачать прогноз в CSV",
            data=csv,
            file_name=f"forecast_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Произошла ошибка: {str(e)}")
        st.info("Пожалуйста, проверьте формат данных и попробуйте снова.")
        st.exception(e)
        
else:
    if uploaded_file is None:
        st.info("👈 Пожалуйста, загрузите файл с данными в боковой панели")
    elif not run_button:
        st.info("👈 Настройте параметры и нажмите 'Запустить анализ'")
    
    # Пример формата данных
    st.markdown("---")
    st.subheader("📋 Пример формата данных")
    
    # Создаем пример данных
    sample_data = pd.DataFrame({
        'Order Date': pd.date_range('2020-01-01', periods=10, freq='M'),
        'Sales': np.random.randint(1000, 5000, 10),
        'Category': ['Furniture', 'Technology', 'Office Supplies'] * 3 + ['Furniture']
    })
    
    st.write("**Пример структуры данных:**")
    st.dataframe(sample_data)
    
    st.markdown("""
    **Требования к данным:**
    - 📅 Колонка с датами в формате YYYY-MM-DD
    - 💰 Колонка с числовыми значениями (продажи, доход и т.д.)
    - 🏷️ (Опционально) Колонка с категориями для фильтрации
    
    **Рекомендации:**
    - Используйте данные как минимум за 1-2 года для лучших результатов
    - Убедитесь, что в данных нет пропусков в колонке с датами
    - Значения в целевой колонке должны быть положительными числами
    """)
    
    st.info("💡 **Совет:** Если возникают проблемы с загрузкой Excel файлов, попробуйте сохранить данные в формате CSV (UTF-8) и загрузить их.")