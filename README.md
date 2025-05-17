Клонирование
```python
git clone https://github.com/zh0rchik/diplom.git
```
Зависимости проекта
```python
pip install -r requirements.txt
```
Собрать проект
```python
pyinstaller --onefile --windowed --add-data "classification_params.json;." --add-data "regression_params.json;." --collect-data imblearn main.py
```

<hr>
Чтобы не забыть как добавить новые либы

```python
pip freeze > requirements.txt
```
<hr>

