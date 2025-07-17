Проблема с передачей большого словаря `verification_data` между шагами конвейера заключается в создании **скрытых зависимостей** и **неявного контракта**.

### **Как это работает сейчас (Проблема)**

1.**Инициализация:** В `VerificationPipeline` создается начальный словарь `verification_data`.
2.**Передача:** Этот словарь, обернутый в `VerificationContext`, передается в первый шаг (`ScreenshotParsingStep`).
3.**Модификация:** `ScreenshotParsingStep` выполняет свою работу и **добавляет в словарь новые ключи**, например, `"fact_hierarchy"` и `"image_analysis"`. Он *подразумевает*, что эти данные понадобятся кому-то дальше.
4.**Зависимость:** Следующий шаг, `FactCheckingStep`, **ожидает**, что в словаре `verification_data` уже есть ключ `"fact_hierarchy"`. Он не объявляет эту зависимость явно, а просто пытается получить доступ к `context.verification_data["fact_hierarchy"]`.

Это приводит к нескольким рискам:

* **Хрупкость кода:** Если вы в `ScreenshotParsingStep` переименуете ключ `"fact_hierarchy"` в `"facts"`, `FactCheckingStep` сломается с ошибкой `KeyError` во время выполнения. Статические анализаторы кода не смогут отловить эту ошибку.
* **Отсутствие безопасности типов:** Какого типа данные лежат по ключу `"fact_hierarchy"`? Это `list`, `dict` или кастомный объект? Код, использующий эти данные, просто надеется на правильный тип, что может привести к `TypeError` или `AttributeError`, если структура данных изменится.
* **Сложность понимания:** Чтобы понять, какие данные нужны шагу `FactCheckingStep`, разработчику нужно прочитать не только его код, но и код всех предыдущих шагов, чтобы выяснить, кто и какие ключи добавляет в словарь.

-----

### **Как это можно улучшить (Решение с Pydantic)**

Вместо словаря используется строго типизированная модель данных Pydantic, которая описывает состояние всего процесса верификации.

#### **Шаг 1: Определите модель состояния**

Создайте Pydantic-модель, которая будет служить единым источником правды для данных верификации. Она будет эволюционировать по мере выполнения конвейера.

```python
# Файл: backend/agent/models/verification_state.py

from pydantic import BaseModel
from typing import Optional, List
from .fact import FactHierarchy  # Предполагаем, что у вас есть такие модели
from .fact_checking_models import FactCheckResult
from .post_analysis_result import PostAnalysisResult

class VerificationState(BaseModel):
    """
    Описывает полное состояние процесса верификации.
    Поля являются опциональными, так как они заполняются на разных шагах.
    """
    # Входные данные
    session_id: str
    user_prompt: Optional[str] = None
    image_data: bytes # или другой тип для изображения

    # Результаты шагов
    fact_hierarchy: Optional[FactHierarchy] = None
    fact_checking_results: Optional[List[FactCheckResult]] = None
    post_analysis: Optional[PostAnalysisResult] = None
    
    # Можно добавить и другие поля, которые сейчас лежат в словаре
    # ...
```

#### **Шаг 2: Измените конвейер и шаги**

Теперь конвейер оперирует не словарем, а экземпляром этой модели.

**Было (в `pipeline_steps.py`):**

```python
# Абстрактный шаг
class VerificationStep(ABC):
    @abstractmethod
    async def execute(self, context: VerificationContext) -> None:
        pass

# Пример использования
class FactCheckingStep(VerificationStep):
    async def execute(self, context: VerificationContext) -> None:
        fact_hierarchy = context.verification_data["fact_hierarchy"]
        # ... работа с fact_hierarchy
        context.verification_data["fact_checking_results"] = results
```

**Стало:**

```python
# Файл: backend/agent/models/verification_state.py (добавим к модели выше)
# Теперь каждый шаг принимает и возвращает обновленное состояние.
# Это делает поток данных явным и предсказуемым.

# Абстрактный шаг
class VerificationStep(ABC):
    @abstractmethod
    async def execute(self, state: VerificationState) -> VerificationState:
        pass

# Пример использования
class FactCheckingStep(VerificationStep):
    async def execute(self, state: VerificationState) -> VerificationState:
        if state.fact_hierarchy is None:
            raise ValueError("Fact hierarchy is not populated before fact checking.")

        # ... работа с state.fact_hierarchy (IDE дает подсказки!)
        
        state.fact_checking_results = results
        return state
```

### **Преимущества нового подхода**

✅ **Надежность и Явный Контракт:** Код становится самодокументируемым. Любой шаг, принимающий `VerificationState`, явно показывает, с какими данными он может работать. Зависимости становятся очевидными.

✅ **Безопасность типов и автодополнение:** Pydantic гарантирует, что в `state.fact_hierarchy` будет объект типа `FactHierarchy` (или `None`). Ваша IDE будет предоставлять автодополнение для атрибутов (`state.fact_hierarchy.main_fact...`), что резко снижает количество ошибок.

✅ **Упрощение рефакторинга:** Если вы решите переименовать поле `fact_hierarchy` в `extracted_facts`, вы можете сделать это с помощью инструментов рефакторинга в IDE, и изменение безопасно применится по всему проекту.

✅ **Раннее обнаружение ошибок:** Если один шаг попытается записать в состояние данные неверного типа, Pydantic немедленно вызовет ошибку валидации, а не позволит неверным данным "отравить" последующие шаги.

Этот рефакторинг не меняет логику работы, но делает поток данных между компонентами системы **прозрачным, надежным и легким для поддержки**.
