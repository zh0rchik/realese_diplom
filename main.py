import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import os
import threading
import warnings
import sys
import json

warnings.filterwarnings('ignore')


class KlystronAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор блоков усилителей мощности клистронов")
        self.root.geometry("1400x700")

        # Пути по умолчанию к файлам параметров
        self.default_regression_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'regression_params.json')
        self.default_classification_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'classification_params.json')
        self.regression_file_path = tk.StringVar(value="regression_params.json")
        self.classification_file_path = tk.StringVar(value="classification_params.json")

        # Загрузка параметров из JSON-файлов
        self.load_model_parameters()

        # Переменные для хранения моделей и данных
        self.regression_models = {'yn': None, 'yc': None, 'yv': None}
        self.classification_models = {'zn': None, 'zc': None, 'zv': None}
        self.scalers_reg = {'yn': None, 'yc': None, 'yv': None}
        self.scalers_class = {'zn': None, 'zc': None, 'zv': None}
        self.reg_data = None
        self.class_data = None
        self.regression_metrics = {}
        self.classification_metrics = {}
        self.code_names = {
            'yn': 'низких частот (мощность)',
            'yc': 'средних частот (мощность)',
            'yv': 'высоких частот (мощность)',
            'zn': 'низких частот (состояние)',
            'zc': 'средних частот (состояние)',
            'zv': 'высоких частот (состояние)'
        }
        self.class_mapping = {
            1: "хорошее",
            2: "удовлетворительное",
            3: "неудовлетворительное"
        }

        # Создаем главное меню
        self.create_main_menu()

        # Экран загрузки
        self.loading_frame = ttk.Frame(self.root)
        self.loading_label = ttk.Label(self.loading_frame, text="Поиск оптимальных гиперпараметров...",
                                       font=("Arial", 14))
        self.loading_label.pack(pady=20)
        self.progress = ttk.Progressbar(self.loading_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)

        # Вкладки
        self.notebook = ttk.Notebook(self.root)

        # Создаем вкладки
        self.create_data_tab()
        self.create_regression_tab()
        self.create_classification_tab()
        self.create_prediction_tab()

        self.notebook.pack(padx=10, pady=10, fill='both', expand=True)

    def load_model_parameters(self):
        """Загрузка параметров моделей из JSON-файлов"""
        valid_reg = True
        valid_class = True
        try:
            # Путь к файлу регрессии
            if self.regression_file_path.get().strip():
                reg_path = self.regression_file_path.get()
            else:
                reg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_params.json')

            with open(reg_path, 'r', encoding='utf-8') as f:
                reg_data = json.load(f)

            # Проверка на наличие нужных ключей и param_grid для регрессии
            required_keys = ['yn', 'yc', 'yv']

            for key in required_keys:
                if key not in reg_data:
                    valid_reg = False
                    break
                if 'param_grid' not in reg_data[key]:
                    valid_reg = False
                    break

            if not valid_reg:
                messagebox.showwarning("Неверный файл регрессии",
                                       "Выбранный JSON-файл не содержит нужных объектов. Будет использован файл по умолчанию.")
                # Загружаем файл по умолчанию
                default_reg_path = os.path.join('regression_params.json')
                self.regression_file_path.set(default_reg_path)
                with open(default_reg_path, 'r', encoding='utf-8') as f:
                    self.regression_params = json.load(f)
            else:
                self.regression_params = reg_data

            # Путь к файлу классификации
            if self.classification_file_path.get().strip():
                class_path = self.classification_file_path.get()
            else:
                class_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification_params.json')

            with open(class_path, 'r', encoding='utf-8') as f:
                class_data = json.load(f)

            # Проверка на наличие нужных ключей и param_grid для классификации
            required_keys = ['zn', 'zc', 'zv']
            for key in required_keys:
                if key not in class_data:
                    valid_class = False
                    break
                if 'param_grid' not in class_data[key]:
                    valid_class = False
                    break

            if not valid_class:
                messagebox.showwarning("Неверный файл классификации",
                                       "Выбранный JSON-файл не содержит нужных объектов. Будет использован файл по умолчанию.")
                # Загружаем файл по умолчанию
                default_class_path = os.path.join('classification_params.json')
                self.classification_file_path.set(default_class_path)
                with open(default_class_path, 'r', encoding='utf-8') as f:
                    self.classification_params = json.load(f)
            else:
                self.classification_params = class_data

            # проверка всех моделей
            required_models = ['yn', 'yc', 'yv', 'zn', 'zc', 'zv']
            reg_models = ['yn', 'yc', 'yv']
            class_models = ['zn', 'zc', 'zv']

            for model in reg_models:
                if model not in self.regression_params:
                    raise ValueError(f"Не найдены параметры для модели регрессии {model}")

            for model in class_models:
                if model not in self.classification_params:
                    raise ValueError(f"Не найдены параметры для модели классификации {model}")

        except FileNotFoundError as e:
            messagebox.showerror("Ошибка", f"Файл параметров не найден: {str(e)}")
            if not valid_reg:
                self.regression_file_path.set('regression_params.json')
            if not valid_class:
                self.classification_file_path.set('classification_params.json')
        except json.JSONDecodeError as e:
            messagebox.showerror("Ошибка", f"Ошибка формата JSON в файле параметров: {str(e)}")
            if not valid_reg:
                self.regression_file_path.set('regression_params.json')
            if not valid_class:
                self.classification_file_path.set('classification_params.json')
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке параметров моделей: {str(e)}")
            if not valid_reg:
                self.regression_file_path.set('regression_params.json')
            if not valid_class:
                self.classification_file_path.set('classification_params.json')

    def create_main_menu(self):
        """Создание главного меню приложения"""
        menubar = tk.Menu(self.root)

        # меню файла
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Загрузить данные регрессии", command=self.load_regression_data)
        file_menu.add_command(label="Загрузить данные классификации", command=self.load_classification_data)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        # меню анализа
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Вывести метрики", command=self.show_metrics)
        menubar.add_cascade(label="Анализ", menu=analysis_menu)

        # меню справки
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Инструкция", command=self.show_help)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def create_data_tab(self):
        """Создание вкладки с данными"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Данные")

        # Фрейм для информации о регрессионных данных
        reg_frame = ttk.LabelFrame(data_frame, text="Данные для регрессии")
        reg_frame.pack(padx=10, pady=5, fill='both', expand=True)

        self.reg_data_info = tk.Text(reg_frame, height=10, width=80)
        self.reg_data_info.pack(padx=10, pady=10, fill='both', expand=True)

        # Фрейм для информации о классификационных данных
        class_frame = ttk.LabelFrame(data_frame, text="Данные для классификации")
        class_frame.pack(padx=10, pady=5, fill='both', expand=True)

        self.class_data_info = tk.Text(class_frame, height=10, width=80)
        self.class_data_info.pack(padx=10, pady=10, fill='both', expand=True)

        # Кнопки загрузки данных
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(padx=10, pady=10, fill='x')

        ttk.Button(btn_frame, text="Загрузить данные регрессии", command=self.load_regression_data).pack(side='left',
                                                                                                         padx=10)
        ttk.Button(btn_frame, text="Загрузить данные классификации", command=self.load_classification_data).pack(
            side='left', padx=10)

    def create_regression_tab(self):
        """Создание вкладки для регрессии"""
        regression_frame = ttk.Frame(self.notebook)
        self.notebook.add(regression_frame, text="Регрессия")

        top_frame = ttk.Frame(regression_frame)
        top_frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(top_frame, text="Оценка выходной мощности блока:", font=("Arial", 12)).pack(side='left', padx=10)

        # фреймы для каждой модели (yn, yc, yв)
        models_frame = ttk.Frame(regression_frame)
        models_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Модель yn (низкие частоты)
        yn_frame = ttk.LabelFrame(models_frame, text="Модель для низких частот (yn)")
        yn_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        ttk.Button(yn_frame, text="Обучить модель", command=lambda: self.train_regression_model('yn')).pack(pady=5)
        self.yn_metrics = tk.Text(yn_frame, height=25, width=30)
        self.yn_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Модель yc (средние частоты)
        yc_frame = ttk.LabelFrame(models_frame, text="Модель для средних частот (yc)")
        yc_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        ttk.Button(yc_frame, text="Обучить модель", command=lambda: self.train_regression_model('yc')).pack(pady=5)
        self.yc_metrics = tk.Text(yc_frame, height=25, width=30)
        self.yc_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Модель yв (высокие частоты)
        yv_frame = ttk.LabelFrame(models_frame, text="Модель для высоких частот (yв)")
        yv_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        ttk.Button(yv_frame, text="Обучить модель", command=lambda: self.train_regression_model('yv')).pack(pady=5)
        self.yv_metrics = tk.Text(yv_frame, height=25, width=30)
        self.yv_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Настройка сетки
        models_frame.grid_columnconfigure(0, weight=1)
        models_frame.grid_columnconfigure(1, weight=1)
        models_frame.grid_columnconfigure(2, weight=1)

        #Выбор JSON-файла параметров
        default_path = os.path.join('regression_params.json')
        self.regression_file_path = tk.StringVar(value=default_path)

        # Создаём фрейм для выбора файла
        file_select_frame = ttk.Frame(regression_frame)
        file_select_frame.pack(padx=10, pady=(0, 10), fill='x')

        ttk.Label(file_select_frame, text="Файл с параметрами регрессионной модели:").pack(side='left', padx=(0, 5))

        file_entry = ttk.Entry(file_select_frame, textvariable=self.regression_file_path, width=60, state='readonly')
        file_entry.pack(side='left', padx=5, fill='x', expand=True)

        ttk.Button(file_select_frame, text="Выбрать файл", command=self.select_regression_file).pack(side='left',
                                                                                                     padx=5)

    def select_regression_file(self):
        """Открыть диалог для выбора JSON-файла параметров для регресии"""
        file_path = filedialog.askopenfilename(
            title="Выберите JSON-файл параметров регрессионных моделей",
            filetypes=[("JSON файлы", "*.json")]
        )
        if file_path:
            self.regression_file_path.set(file_path)
            self.load_model_parameters()

    def create_classification_tab(self):
        """Создание вкладки для классификации"""
        classification_frame = ttk.Frame(self.notebook)
        self.notebook.add(classification_frame, text="Классификация")

        top_frame = ttk.Frame(classification_frame)
        top_frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(top_frame, text="Оценка состояния блока усилителя мощности:", font=("Arial", 12)).pack(side='left',
                                                                                                         padx=10)
        # Создание для каждой модели (zn, zc, zв)
        models_frame = ttk.Frame(classification_frame)
        models_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Модель zn (низкие частоты)
        zn_frame = ttk.LabelFrame(models_frame, text="Модель для низких частот (zn)")
        zn_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        ttk.Button(zn_frame, text="Обучить модель", command=lambda: self.train_classification_model('zn')).pack(pady=5)
        self.zn_metrics = tk.Text(zn_frame, height=25, width=30)
        self.zn_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Модель zc (средние частоты)
        zc_frame = ttk.LabelFrame(models_frame, text="Модель для средних частот (zc)")
        zc_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        ttk.Button(zc_frame, text="Обучить модель", command=lambda: self.train_classification_model('zc')).pack(pady=5)
        self.zc_metrics = tk.Text(zc_frame, height=25, width=30)
        self.zc_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Модель zв (высокие частоты)
        zv_frame = ttk.LabelFrame(models_frame, text="Модель для высоких частот (zв)")
        zv_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        ttk.Button(zv_frame, text="Обучить модель", command=lambda: self.train_classification_model('zv')).pack(pady=5)
        self.zv_metrics = tk.Text(zv_frame, height=25, width=30)
        self.zv_metrics.pack(padx=5, pady=5, fill='both', expand=True)

        # Настройка сетки
        models_frame.grid_columnconfigure(0, weight=1)
        models_frame.grid_columnconfigure(1, weight=1)
        models_frame.grid_columnconfigure(2, weight=1)

        # Выбор JSON-файла параметров
        default_path = os.path.join('classification_params.json')
        self.classification_file_path = tk.StringVar(value=default_path)

        # Создаём фрейм для выбора файла
        file_select_frame = ttk.Frame(classification_frame)
        file_select_frame.pack(padx=10, pady=(0, 10), fill='x')

        ttk.Label(file_select_frame, text="Файл с параметрами классификационной модели:").pack(side='left', padx=(0, 5))

        file_entry = ttk.Entry(file_select_frame, textvariable=self.classification_file_path, width=60,
                               state='readonly')
        file_entry.pack(side='left', padx=5, fill='x', expand=True)

        ttk.Button(file_select_frame, text="Выбрать файл", command=self.select_classification_file).pack(side='left',
                                                                                                         padx=5)

    def select_classification_file(self):
        """Открыть диалог для выбора JSON-файла параметров для классификации"""
        file_path = filedialog.askopenfilename(
            title="Выберите JSON-файл параметров классификационных моделей",
            filetypes=[("JSON файлы", "*.json")]
        )
        if file_path:
            self.classification_file_path.set(file_path)
            self.load_model_parameters()

    def create_prediction_tab(self):
        """Создание вкладки прогнозирования"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="Прогнозирование")

        # Фрейм для ввода данных
        input_frame = ttk.LabelFrame(prediction_frame, text="Ввод параметров нового клистрона")
        input_frame.pack(padx=10, pady=10, fill='x')

        # Создание фрейма с сеткой для x1-x10
        x_frame = ttk.Frame(input_frame)
        x_frame.pack(padx=10, pady=10, fill='x')

        # Создание поля ввода для x1-x10
        self.x_entries = {}
        for i in range(10):
            row, col = divmod(i, 5)
            ttk.Label(x_frame, text=f"х{i + 1}:").grid(row=row, column=col * 2, padx=5, pady=5, sticky='e')
            self.x_entries[f'x{i + 1}'] = ttk.Entry(x_frame, width=10)
            self.x_entries[f'x{i + 1}'].grid(row=row, column=col * 2 + 1, padx=5, pady=5, sticky='w')

        # Поля для U и T
        ut_frame = ttk.Frame(input_frame)
        ut_frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(ut_frame, text="U:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.u_entry = ttk.Entry(ut_frame, width=10)
        self.u_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(ut_frame, text="T:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.t_entry = ttk.Entry(ut_frame, width=10)
        self.t_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        # Кнопка прогнозирования
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(padx=10, pady=10, fill='x')

        ttk.Button(buttons_frame, text="Заполнить примерными данными", command=self.fill_sample_data).pack(side='left',
                                                                                                           padx=10)
        ttk.Button(buttons_frame, text="Рассчитать прогноз", command=self.make_prediction).pack(side='right', padx=10)

        # Фрейм для вывода результатов
        results_frame = ttk.LabelFrame(prediction_frame, text="Результаты прогнозирования")
        results_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Текстовое поле для вывода результатов
        self.results_text = tk.Text(results_frame, height=12, width=80)
        self.results_text.pack(padx=10, pady=10, fill='both', expand=True)

    def load_regression_data(self):
        """Загрузка данных для регрессии"""
        file_path = filedialog.askopenfilename(title="Выберите файл данных для регрессии",
                                               filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv")])
        if file_path:
            try:
                # Определение типа файла по расширению
                if file_path.endswith('.csv'):
                    self.reg_data = pd.read_csv(file_path, encoding='utf-8')
                else:
                    self.reg_data = pd.read_excel(file_path)

                # Проверка наличие необходимых столбцов
                required_columns_reg = ['х1', 'х2', 'х3', 'х4', 'х5', 'х6', 'х7', 'х8', 'х9', 'х10', 'Yн', 'Yс', 'Yв']
                missing_columns = [col for col in required_columns_reg if col not in self.reg_data.columns]

                if missing_columns:
                    messagebox.showwarning("Предупреждение",
                                           f"В файле отсутствуют следующие столбцы: {', '.join(missing_columns)}")

                # Обновление инфы на вкладке данных
                info_text = f"Загружен файл: {file_path}\n"
                info_text += f"Количество строк: {len(self.reg_data)}\n"
                info_text += f"Колонки: {', '.join(self.reg_data.columns.tolist())}\n\n"

                # описательная статистика
                info_text += "Описательная статистика:\n"
                info_text += str(self.reg_data.describe())

                self.reg_data_info.delete(1.0, tk.END)
                self.reg_data_info.insert(tk.END, info_text)

                messagebox.showinfo("Успешно", "Данные для регрессии успешно загружены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def load_classification_data(self):
        """Загрузка данных для классификации"""
        file_path = filedialog.askopenfilename(title="Выберите файл данных для классификации",
                                               filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv")])
        if file_path:
            try:
                # Определение типа файла по расширению
                if file_path.endswith('.csv'):
                    self.class_data = pd.read_csv(file_path, encoding='utf-8')
                else:
                    self.class_data = pd.read_excel(file_path)

                # Проверка наличие необходимых столбцов
                required_columns_class = ['Yн', 'Yс', 'Yв', 'U', 'Т', 'Zн', 'Zс', 'Zв']
                missing_columns = [col for col in required_columns_class if col not in self.class_data.columns]

                if missing_columns:
                    messagebox.showwarning("Предупреждение",
                                           f"В файле отсутствуют следующие столбцы: {', '.join(missing_columns)}")

                # Обновление инфы на вкладке данных
                info_text = f"Загружен файл: {file_path}\n"
                info_text += f"Количество строк: {len(self.class_data)}\n"
                info_text += f"Колонки: {', '.join(self.class_data.columns.tolist())}\n\n"

                # описательную статистику
                info_text += "Описательная статистика:\n"
                info_text += str(self.class_data.describe())

                # распределение классов
                info_text += "\n\nРаспределение классов:\n"
                for col in ['Zн', 'Zс', 'Zв']:
                    if col in self.class_data.columns:
                        info_text += f"{col}: {self.class_data[col].value_counts().sort_index().to_dict()}\n"

                self.class_data_info.delete(1.0, tk.END)
                self.class_data_info.insert(tk.END, info_text)

                messagebox.showinfo("Успешно", "Данные для классификации успешно загружены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def train_regression_model(self, model_type):
        """Обучение модели регрессии для указанного типа (yn, yc, или yв)"""
        if self.reg_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные для регрессии")
            return

        # отображение экрана загрузки
        self.notebook.pack_forget()
        self.loading_frame.pack(fill='both', expand=True)
        self.progress.start()

        # Запуск обучения в отдельном потоке
        threading.Thread(target=self._train_regression_thread, args=(model_type,)).start()

    def _train_regression_thread(self, model_type):
        """Поток для обучения модели регрессии"""
        try:
            # Получить параметры из загруженного JSON
            params = self.regression_params.get(model_type)
            if not params:
                raise ValueError(f"Не найдены параметры для модели {model_type}")

            # Определить входные и выходную переменную
            X = self.reg_data[['х1', 'х2', 'х3', 'х4', 'х5', 'х6', 'х7', 'х8', 'х9', 'х10']]
            target_column = params['target_column']

            if target_column not in self.reg_data.columns:
                self.root.after(0,
                                lambda: messagebox.showerror("Ошибка", f"Столбец {target_column} не найден в данных"))
                self.root.after(0, self._hide_loading)
                return

            y = self.reg_data[target_column]

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=params.get('random_state_split', 42)
            )

            # Нормализация данных
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Сохраняем скейлер
            self.scalers_reg[model_type] = scaler

            # Сетка гиперпараметров и кросс-валидация
            cv = ShuffleSplit(
                n_splits=params.get('n_splits', 5),
                test_size=params.get('test_size_cv', 0.1),
                random_state=params.get('random_state_cv', 42)
            )

            # сетка гиперпараметров, которая содержит в себе гиперпараметры из JSON
            simplified_param_grid = {}

            for key in params['param_grid']:
                if key == 'hidden_layer_sizes':
                    simplified_param_grid[key] = [
                        tuple(x) if isinstance(x, (list, tuple)) else (x,)
                        for x in params['param_grid'][key]
                    ]
                else:
                    simplified_param_grid[key] = params['param_grid'][key]

            grid_search = GridSearchCV(
                estimator=MLPRegressor(random_state=params.get('random_state_mlp', 42), max_iter=300),
                param_grid=simplified_param_grid,
                cv=cv,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1,
                verbose=0
            )

            # Обучение модели
            grid_search.fit(X_train_scaled, y_train)

            # Лучшая модель
            best_model = grid_search.best_estimator_
            self.regression_models[model_type] = best_model
            print(best_model)

            # Прогнозирование на тестовом наборе
            y_pred = best_model.predict(X_test_scaled)

            # Вычисление метрик
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Сохранение метрик
            self.regression_metrics[model_type] = {
                'MAPE': mape,
                'RMSE': rmse,
                'best_params': grid_search.best_params_
            }

            # Обновление интерфейса
            self.root.after(0, lambda: self._update_regression_ui(model_type))

            # Скрываем экран загрузки
            self.root.after(0, self._hide_loading)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка",
                                                            f"Ошибка при обучении модели {model_type}: {str(e)}"))
            self.root.after(0, self._hide_loading)

    def _update_regression_ui(self, model_type):
        """Обновление интерфейса после обучения модели регрессии"""
        metrics = self.regression_metrics.get(model_type, {})
        text_widget = getattr(self, f"{model_type}_metrics")

        text_widget.delete(1.0, tk.END)
        if metrics:
            text_widget.insert(tk.END, f"Модель: {self.code_names[model_type]}\n")
            text_widget.insert(tk.END, f"MAPE: {metrics['MAPE']:.2f}%\n")
            text_widget.insert(tk.END, f"RMSE: {metrics['RMSE']:.2f}\n")
            text_widget.insert(tk.END, "\nЛучшие параметры:\n")
            for param, value in metrics['best_params'].items():
                text_widget.insert(tk.END, f"{param}: {value}\n")
        else:
            text_widget.insert(tk.END, "Модель не обучена")

    def train_classification_model(self, model_type):
        """Обучение модели классификации для указанного типа (zn, zc, или zв)"""
        if self.class_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные для классификации")
            return

        # Показываем экран загрузки
        self.notebook.pack_forget()
        self.loading_frame.pack(fill='both', expand=True)
        self.progress.start()

        # Запуск обучения в отдельном потоке
        threading.Thread(target=self._train_classification_thread, args=(model_type,)).start()

    @staticmethod
    def balance_classes(X, y):
        """Балансировка классов с помощью SMOTE и вычисление весов классов"""
        try:
            # Рассчитываем веса классов
            classes = np.unique(y)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

            # Используем SMOTE для балансировки (только если есть более 1 образца для каждого класса)
            smote = SMOTE(k_neighbors=min(5, len(y) - 1), random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            return X_resampled, y_resampled, class_weight_dict
        except ValueError as e:
            print(f"Ошибка балансировки: {str(e)}")
            return X, y, {cls: 1.0 for cls in np.unique(y)}

    def _train_classification_thread(self, model_type):
        """Поток для обучения модели классификации"""
        try:
            # Получаем параметры из загруженного JSON
            params = self.classification_params.get(model_type)
            if not params:
                raise ValueError(f"Не найдены параметры для модели {model_type}")

            # Определяем входные и выходную переменную
            X = self.class_data[['Yн', 'Yс', 'Yв', 'U', 'Т']]
            target_column = params['target_column']

            if target_column not in self.class_data.columns:
                self.root.after(0,
                                lambda: messagebox.showerror("Ошибка",
                                                             f"Столбец {target_column} не найден в данных"))
                self.root.after(0, self._hide_loading)
                return

            y = self.class_data[target_column]

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=params.get('random_state_split', 42),
                stratify=y
            )

            # Нормализация данных
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Сохраняем скейлер
            self.scalers_class[model_type] = scaler

            # Балансировка классов с помощью SMOTE
            try:
                smote = SMOTE(k_neighbors=1, random_state=params.get('random_state_smote', 42))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            except ValueError as e:
                print(f"Ошибка SMOTE: {str(e)}. Используем оригинальные данные.")
                X_train_balanced, y_train_balanced = X_train_scaled, y_train

            # Логирование информации о балансировке
            print(f"\nБалансировка для {model_type}:")
            print("До балансировки:", np.bincount(y_train))
            print("После балансировки:", np.bincount(y_train_balanced))

            # Сетка гиперпараметров и кросс-валидация
            cv = StratifiedKFold(
                n_splits=params.get('n_splits', 5),
                shuffle=True,
                random_state=params.get('random_state_cv', 42)
            )

            # формирование simplified_param_grid
            simplified_param_grid = {}
            for key in params['param_grid']:
                if key == 'hidden_layer_sizes':
                    simplified_param_grid[key] = [
                        tuple(x) if isinstance(x, (list, tuple)) else (x,)
                        for x in params['param_grid'][key]
                    ]
                else:
                    simplified_param_grid[key] = params['param_grid'][key]

            grid_search = GridSearchCV(
                estimator=MLPClassifier(
                    random_state=params.get('random_state_mlp', 42)
                ),
                param_grid=simplified_param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train_balanced, y_train_balanced)

            # Лучшая модель
            best_model = grid_search.best_estimator_
            self.classification_models[model_type] = best_model

            # Прогнозирование на тестовом наборе
            y_pred = best_model.predict(X_test_scaled)

            # Вычисление метрик
            f1 = f1_score(y_test, y_pred, average='weighted') # weighted означает, что набор данных несбалансированный.
            report = classification_report(y_test, y_pred, target_names=self.class_mapping.values())

            # Сохранение метрик
            self.classification_metrics[model_type] = {
                'F1-score': f1,
                'classification_report': report,
                'best_params': grid_search.best_params_,
                'class_distribution': {
                    'before': np.bincount(y_train).tolist(),
                    'after': np.bincount(y_train_balanced).tolist()
                }
            }

            # Вывод информации в консоль
            print("\nОтчёт по классификации для", model_type)
            print(report)
            print(f"F-мера ({model_type}): {f1:.4f}")

            # Обновление интерфейса
            self.root.after(0, lambda: self._update_classification_ui(model_type))
            self.root.after(0, self._hide_loading)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка",
                                                            f"Ошибка при обучении модели {model_type}: {str(e)}"))
            self.root.after(0, self._hide_loading)
            import traceback
            traceback.print_exc()

    def _update_classification_ui(self, model_type):
        """Обновление интерфейса после обучения модели классификации"""
        metrics = self.classification_metrics.get(model_type, {})
        text_widget = getattr(self, f"{model_type}_metrics")

        text_widget.delete(1.0, tk.END)
        if metrics:
            text_widget.insert(tk.END, f"Модель: {self.code_names[model_type]}\n")
            text_widget.insert(tk.END, f"F1-score: {metrics['F1-score']:.2f}\n")
            # text_widget.insert(tk.END, "\nОтчет классификации:\n")
            # text_widget.insert(tk.END, metrics['classification_report'])
            text_widget.insert(tk.END, "\nЛучшие параметры:\n")
            for param, value in metrics['best_params'].items():
                text_widget.insert(tk.END, f"{param}: {value}\n")
        else:
            text_widget.insert(tk.END, "Модель не обучена")

    def _hide_loading(self):
        """Скрытие экрана загрузки"""
        self.progress.stop()
        self.loading_frame.pack_forget()
        self.notebook.pack(padx=10, pady=10, fill='both', expand=True)

    def show_metrics(self):
        """Показ всех метрик в отдельном окне"""
        if not self.regression_metrics and not self.classification_metrics:
            messagebox.showwarning("Предупреждение", "Сначала обучите модели")
            return

        # новое окно
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Метрики всех моделей")
        metrics_window.geometry("800x600")

        # текстовое поле
        text_frame = ttk.Frame(metrics_window)
        text_frame.pack(padx=10, pady=10, fill='both', expand=True)

        text_area = tk.Text(text_frame, wrap=tk.WORD)
        text_area.pack(side=tk.LEFT, fill='both', expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scrollbar.set)

        # метрики регрессии
        if self.regression_metrics:
            text_area.insert(tk.END, "=== МЕТРИКИ РЕГРЕССИИ ===\n\n")
            for model_type, metrics in self.regression_metrics.items():
                text_area.insert(tk.END, f"Модель: {self.code_names[model_type]}\n")
                text_area.insert(tk.END, f"MAPE: {metrics['MAPE']:.2f}%\n")
                text_area.insert(tk.END, f"RMSE: {metrics['RMSE']:.2f}\n")
                text_area.insert(tk.END, "Лучшие параметры:\n")
                for param, value in metrics['best_params'].items():
                    text_area.insert(tk.END, f"  {param}: {value}\n")
                text_area.insert(tk.END, "\n")

        # метрики классификации
        if self.classification_metrics:
            text_area.insert(tk.END, "\n=== МЕТРИКИ КЛАССИФИКАЦИИ ===\n\n")
            for model_type, metrics in self.classification_metrics.items():
                text_area.insert(tk.END, f"Модель: {self.code_names[model_type]}\n")
                text_area.insert(tk.END, f"F1-score: {metrics['F1-score']:.2f}\n")
                text_area.insert(tk.END, "Отчет классификации:\n")
                text_area.insert(tk.END, metrics['classification_report'])
                text_area.insert(tk.END, "Лучшие параметры:\n")
                for param, value in metrics['best_params'].items():
                    text_area.insert(tk.END, f"  {param}: {value}\n")
                text_area.insert(tk.END, "\n")

    def fill_sample_data(self):
        """Заполнение полей ввода примерными данными"""
        sample_data = {
            'x1': 9.5, 'x2': 10.95, 'x3': 4.3, 'x4': 93.2, 'x5': 3.9,
            'x6': 9.5, 'x7': 274, 'x8': 31, 'x9': 0.95, 'x10': 50.53,
            'u': 10.8, 't': 73
        }

        for key, entry in self.x_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(sample_data[key]))

        self.u_entry.delete(0, tk.END)
        self.u_entry.insert(0, str(sample_data['u']))

        self.t_entry.delete(0, tk.END)
        self.t_entry.insert(0, str(sample_data['t']))

    def make_prediction(self):
        """Выполнение прогнозирования на основе введенных данных"""
        try:
            # Проверка, что все модели обучены
            if None in self.regression_models.values() or None in self.classification_models.values():
                messagebox.showwarning("Предупреждение", "Сначала обучите все модели")
                return

            # Собираем данные для регрессии
            x_values = []
            for i in range(1, 11):
                entry = self.x_entries.get(f'x{i}')
                if not entry:
                    messagebox.showerror("Ошибка", f"Не заполнено поле x{i}")
                    return
                try:
                    x_values.append(float(entry.get()))
                except ValueError:
                    messagebox.showerror("Ошибка", f"Некорректное значение в поле x{i}")
                    return

            # numpy массив и нормализуем
            X_reg = np.array([x_values])
            X_reg_scaled = {}
            for model_type in ['yn', 'yc', 'yv']:
                X_reg_scaled[model_type] = self.scalers_reg[model_type].transform(X_reg)

            # Прогнозируем значения мощности
            y_pred = {}
            for model_type in ['yn', 'yc', 'yv']:
                y_pred[model_type] = self.regression_models[model_type].predict(X_reg_scaled[model_type])[0]

            # Собираем данные для классификации
            try:
                u = float(self.u_entry.get())
                t = float(self.t_entry.get())
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные значения U или T")
                return

            X_class = np.array([[y_pred['yn'], y_pred['yc'], y_pred['yv'], u, t]])
            X_class_scaled = {}
            for model_type in ['zn', 'zc', 'zv']:
                X_class_scaled[model_type] = self.scalers_class[model_type].transform(X_class)

            # Прогноз состояния
            z_pred = {}
            for model_type in ['zn', 'zc', 'zv']:
                z_pred[model_type] = self.classification_models[model_type].predict(X_class_scaled[model_type])[0]

            # сборка результатов
            result_text = "=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ===\n\n"
            result_text += "1. Прогнозируемая выходная мощность:\n"
            result_text += f"  - Низкие частоты (Yн): {y_pred['yn']:.2f}\n"
            result_text += f"  - Средние частоты (Yс): {y_pred['yc']:.2f}\n"
            result_text += f"  - Высокие частоты (Yв): {y_pred['yv']:.2f}\n\n"

            result_text += "2. Прогнозируемое состояние блока:\n"
            for model_type in ['zn', 'zc', 'zv']:
                state = self.class_mapping.get(z_pred[model_type], "неизвестно")
                result_text += f"  - {self.code_names[model_type]}: {state}\n"

            # Вывод результатов
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при прогнозировании: {str(e)}")

    def show_about(self):
        """Показ информации о программе"""
        about_text = """Анализатор блоков усилителей мощности клистронов\n
            Версия 1.0\n
            Программа предназначена для анализа и прогнозирования характеристик 
            блока усилителя мощности клистронов на основе нейросетевых моделей.\n
            Разработчик: Зюзин Георгий, УлГТУ\n
            © 2025 УлГТУ"""
        messagebox.showinfo("О программе", about_text)

    def show_help(self):
        """Показ инструкции по использованию программы"""
        help_text = HELP_TEXT = """
        Инструкция для приложения
        
        1. Подготовка данных  
        Перед использованием приложения необходимо:
        
          1.1. Для регрессии (прогнозирование мощности Yн, Yс, Yв):  
            - Подготовить файл (CSV/Excel) с колонками:  
              х1, х2, х3, х4, х5, х6, х7, х8, х9, х10, Yн, Yс, Yв
        
          1.2. Для классификации (оценка состояния Zн, Zс, Zв):  
            - Подготовить файл (CSV/Excel) с колонками:  
              Yн, Yс, Yв, U, Т, Zн, Zс, Zв  
              (где Zн, Zс, Zв — метки классов: 1=хорошее, 2=удовл., 3=неудовл.)
        
        2. Загрузка данных  
          - Меню → Файл → Загрузить данные регрессии (выбрать файл с x1-x10, Yн/Yс/Yв).  
          - Меню → Файл → Загрузить данные классификации (выбрать файл с Yн/Yс/Yв, U, T, Zн/Zс/Zв).
        
        3. Обучение моделей  
          3.1. Регрессия (вкладка Регрессия):  
            - Обучить модели для Yн, Yс, Yв (кнопки "Обучить модель").  
          3.2. Классификация (вкладка Классификация):  
            - Обучить модели для Zн, Zс, Zв (кнопки "Обучить модель").
        
          Примечание:  
            - Перед обучением можно изменить гиперпараметры в JSON-файлах (кнопка "Выбрать файл" на каждой вкладке).
        
          Минимальная структура JSON (для классификации аналогично для zn, zc, zv):
        
          {
            "yn": { "param_grid": { ... } },
            "yc": { "param_grid": { ... } },
            "yv": { "param_grid": { ... } }
          }
        
        4. Прогнозирование (вкладка Прогнозирование)  
          - Ввести параметры нового клистрона:  
            x1-x10 (технические характеристики).  
            U, T (доп. параметры).  
          - Нажать "Рассчитать прогноз".  
          Результат:  
            - Прогноз мощности (Yн, Yс, Yв).  
            - Оценка состояния (Zн, Zс, Zв).  
          Тест: Кнопка "Заполнить примерными данными" подставит тестовые значения.
        
        5. Просмотр метрик  
          - На вкладках Регрессия и Классификация отображаются:  
            - MAPE и RMSE (для регрессии).  
            - F1-score (для классификации).  
          - Полный отчет: Меню → Анализ → Вывести метрики.
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("Инструкция")
        help_window.geometry("1000x700")

        text_frame = ttk.Frame(help_window)
        text_frame.pack(padx=10, pady=10, fill='both', expand=True)

        text_area = tk.Text(text_frame, wrap=tk.WORD)
        text_area.pack(side=tk.LEFT, fill='both', expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scrollbar.set)

        text_area.insert(tk.END, help_text)
        text_area.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = KlystronAnalyzer(root)
    root.mainloop()