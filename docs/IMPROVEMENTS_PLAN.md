# План улучшений (простой видеоредактор на генерациях)

Этот документ — продуктовый и инженерный roadmap для форка **LTX Desktop**: простой видеоредактор, где сборка/монтаж тесно связаны с генерациями (T2V/I2V/A2V/retake), с учётом **всех платформ**.

## Контекст

Архитектура: Electron (main/preload) + React/TS (renderer) + Python/FastAPI (backend).

## Принципы фокуса (чтобы не превратиться в «тяжёлый NLE»)

- **Shot-based workflow** как первичный UX: шоты → вариации → промоут в таймлайн → контекстная генерация из таймлайна.
- **Надёжность проекта** важнее «больших фич»: нельзя терять работу, нельзя превращать проект в свалку ассетов.
- **Кроссплатформенность**: пути/URL, файловые операции, доступ к корзине/трешу и диалоги должны работать одинаково.

## Каталог улучшений (roadmap)

Ниже пункты сгруппированы по смыслу. Для каждого — кратко: **что / зачем / MVP**.

### A) Shot workflow и вариации (ядро продукта)

#### A1) Очередь генераций внутри проекта

- **Что**: очередь задач генерации (видео/изображения/retake), управляемая из проекта.
- **Зачем**: пользователь набивает 10–20 задач и уходит — приложение “пашет” само.
- **MVP**:
  - очередь задач: add / cancel / retry
  - batch variations одного шота по seed / duration / resolution
  - «pause after current» (остановиться после текущей)
  - (позже) reorder drag-and-drop

#### A2) Storyboard / shot-based workflow

- **Статус**: базовый UI- и модельный слой **частично реализован** — добавлены типы `Shot`/`ShotVariant`, поле `shots` в `Project` и простой `ShotBoard` в Project view.
- **Что**: список шотов внутри проекта как промежуточный слой между Gen Space и Timeline.
- **Зачем**: проект становится сценой/сценарием, а не просто “папкой с попытками”.
- **MVP**:
  - shot list: prompt, refs, seed, aspect, duration, notes
  - “generate next variation”
  - “promote to timeline”
  - простое сравнение версий шота (2-up compare)

#### A3) Variations / branching для ассетов

- **Статус**: подготовлен слой `ShotVariant` и поле `canonicalVariantId` в `Shot`; фактическое дерево вариаций по ассетам и UI для выбора canonical всё ещё в планах.
- **Что**: дерево вариантов для шота/ассета + явная parent→child связь («из чего получено»).
- **Зачем**: творчество не превращается в плоскую свалку файлов.
- **MVP**:
  - дерево вариантов (shot_03: v1/v2/v3…)
  - “make canonical” (пометить лучший как основной)
  - side-by-side compare (2-up)

### B) Timeline-aware generation (вторая ключевая опора)

#### B1) Timeline-aware generation

- **Что**: команды генерации, понимающие контекст таймлайна.
- **Зачем**: editor — не только сборка, но и место “контекстной генерации”.
- **MVP**:
  - “generate clip for selected gap”
  - “extend previous clip into gap”
  - “bridge between clip A and clip B”

#### B2) Extend / loop / prepend по времени

- **Что**: продление клипа без пересчёта всего.
- **Зачем**: практично при ограниченных длительностях модели/API.
- **MVP**:
  - extend +2s / +4s
  - seamless loop mode
  - prepend frames (добавить перед клипом)

### C) Prompt/consistency UX (быстро окупается)

#### C1) Prompt memory и reusable prompt blocks

- **Что**: библиотека переиспользуемых блоков промпта (positive/negative) + project defaults.
- **Зачем**: не копировать каждый раз “cinematic realism…”, быстрее итерации, меньше ошибок.
- **MVP**:
  - сохранённые blocks (positive/negative)
  - project-level defaults
  - стратегия merge (порядок/дедуп/конфликты)

#### C2) Consistency packs (Character/Style/Location cards)

- **Что**: явные сущности проекта, которые можно “применить к шоту”.
- **Зачем**: консистентность персонажей/стиля без огромных копипаст‑промптов.
- **MVP**:
  - Character card (refs + описание + запреты)
  - Style/Location cards
  - “apply to shot”

#### C3) Reference board (moodboard)

- **Что**: несколько референсов на шот одновременно, с ролями и весами.
- **Зачем**: для видео часто важнее доска референсов, чем одно поле prompt.
- **MVP**:
  - 2–8 картинок
  - роли: style / character / composition / environment
  - веса

### D) Модели/железо/режимы (расширяет аудиторию)

#### D1) VRAM-aware presets и adaptive local mode

- **Что**: “Fit my hardware” режим, который подбирает параметры под железо.
- **Зачем**: больше реальной аудитории, меньше “запуск не поместился”.
- **MVP**:
  - авто‑подбор resolution/duration/model/offload где возможно
  - предупреждение “этот запуск не поместится” до старта
  - degraded modes вместо жёсткого отказа (где применимо)

#### D2) Модельный хаб внутри приложения

- **Что**: управляемый список моделей/вариантов.
- **Зачем**: понятность “что ставить”, “сколько весит”, “какое качество/скорость”.
- **MVP**:
  - список моделей + size + VRAM requirement + speed/quality label
  - install/remove
  - per-project default model

### E) Multi-backend (power users; заметная ценность, но не MVP)

#### E1) Backend target: local / cloud API / custom server

- **Что**: выбор backend target и работа с удалённой очередью задач.
- **Зачем**: запускать генерации на своём сервере/ноде (private inference).
- **MVP**:
  - сохранение нескольких endpoints
  - статус сервера/доступных моделей
  - remote job queue
  - sync completed assets обратно в проект

### F) Управление библиотекой и «гигиена проекта»

#### F1) Asset intelligence: поиск и умные коллекции

- **Что**: теги, поиск, smart folders/filters.
- **Зачем**: на больших проектах без этого невозможно работать.
- **MVP**:
  - tags
  - search по prompt/filename/date/model
  - smart filters: favorites/failed/unused/variants/retakes

#### F2) Удаление/архивация/очистка outputs

- **Что**: полноценные операции “delete/move to trash/cleanup”.
- **Зачем**: уменьшить хаос в outputs и дать нормальные ежедневные операции.
- **MVP**:
  - delete asset (move to trash)
  - cleanup unused outputs/cache
  - consolidate outputs into project folder
  - relink broken assets

### G) Надёжность и совместимость проектов (нельзя терять работу)

#### G1) Файловое хранение проектов + схема + миграции

- **Статус**: базовый слой **реализован** — проекты теперь сохраняются на диск через Electron, есть `projectSchemaVersion` и миграция из `localStorage`.
- **Что**: перейти от `localStorage` к проектам на диске с версионированием схемы.
- **Зачем**: размер/надёжность/переносимость, возможность миграций и бэкапов.
- **MVP**:
  - один файл на проект + `projectSchemaVersion` ✅
  - миграторы vN→vN+1
  - импорт/экспорт проекта (формат/архив)

#### G2) Missing media detection + Relink

- **Что**: детектор отсутствующих файлов + мастер перепривязки.
- **Зачем**: файлы перемещаются, `file://` ломается — проект не должен “умирать”.
- **MVP**:
  - scan missing media при открытии проекта
  - batch‑relink по выбранной папке

#### G3) Crash recovery + версии проекта

- **Что**: автосейв/снапшоты/восстановление после краша.
- **Зачем**: редактор и генерации могут падать; работа должна восстанавливаться.
- **MVP**:
  - автосейв каждые N секунд / событийный
  - восстановление при следующем запуске
  - хранить несколько последних версий (ring buffer)

#### G4) Diagnostics / “Report issue” + redaction секретов

- **Что**: единый “диагностический бандл” для саппорта/issue.
- **Зачем**: ускоряет triage и уменьшает “не могу воспроизвести”.
- **MVP**:
  - собрать версии/платформу/режим/статус моделей/последние логи
  - redaction: исключить/замаскировать API keys и чувствительные данные

## Инженерный backlog (текущее состояние кода/инфры)

Эти пункты полезны, но их приоритет регулируется продуктовым фокусом и текущими задачами.

### 1) Видеоредактор: `VideoEditor.tsx` слишком большой

- Файл: `frontend/views/VideoEditor.tsx` (очень крупный компонент).
- Риски: сложность поддержки, регрессы, производительность (лишние перерендеры), рост времени на изменения.
- Идея: декомпозиция на подкомпоненты + выделение стейта (timeline/UI) в отдельные хуки/контекст.

### 3) Отсутствие фронтенд‑тестов

- Сейчас: фронтенд‑тестов нет.
- Идея: Vitest + React Testing Library + jsdom, начать с утилит и ключевых хуков.

### 4) Безопасность dev‑режима: `webSecurity` отключён

- Файл: `electron/window.ts` (в dev выключается `webSecurity`).
- Риск: ослабление безопасности и возможные неожиданные эффекты в поведении.
- Идея: держать `webSecurity: true` и полагаться на CORS/настройки бэкенда для dev.

### 5) Интернационализация (i18n)

- Сейчас: строки захардкожены в JSX, местами встречается русский текст без системы переводов.
- Идея: `react-i18next`, словари `en/ru`, постепенное выносение строк в ключи.

### 6) Доступность (a11y)

- Сейчас: мало `aria-*`, `role`, не везде есть нормальный keyboard flow.
- Идея: базовая доступность модалок/меню (role, aria-label/description, ESC, focus trap, таб‑навигация).

### 8) Прогресс генерации: polling → push

- Сейчас: polling прогресса в рендерере.
- Идея: WebSocket/SSE для прогресса генерации, меньше нагрузки и более точный прогресс.

### 9) Проверка места на диске на первом запуске

- Сейчас: есть задел, но нет реального расчёта свободного места.
- Идея: IPC `statfs` (Node) и отображение доступного места в онбординге.

### 10) «Скрытые» фичи (Effects / IC‑LoRA)

- Сейчас: в коде встречаются `EFFECTS HIDDEN` / `IC-LORA HIDDEN` (закомментированные участки).
- Идея: вернуть через feature flags (вместо комментирования), оценить готовность.

### 11) Lint/format инфраструктура

- Сейчас: нет ESLint/Prettier в toolchain.
- Идея: добавить ESLint + Prettier + lint step в CI.

### 12) UX обновлений приложения

- Сейчас: auto-updater есть, но UI сценарий «обновление доступно / скачивается / установить и перезапустить» ограничен.
- Идея: UI уведомления и управляемый рестарт.

## R&D / long-term направления (не в основном roadmap)

Эти идеи сильные, но заметно увеличивают сложность UI/ML/пайплайнов. Их стоит рассматривать после стабилизации ядра.

- **In/outpaint и masked retake**: маска по кадру, заморозка фона, propagation, feather/grow.
- **Ключевые кадры и interpolation**: A/B/C keyframes, контроль перехода, фиксация персонажа/камеры.
- **Multi-pass generation**: Fast Preview / Balanced / Final Quality (несколько проходов качества).
- **LoRA / adapters / style modules**: стек адаптеров, веса, project-level stack.
- **Sound-driven монтаж для A2V**: beat detection, auto-cut, intensity linked to energy.

## Приоритеты (3 тира)

### Верхний тир (максимальная ценность для «простого генеративного редактора»)

- **G1: Файловое хранение проектов + schema version + миграции**
- **A1: Очередь генераций внутри проекта (MVP: add/cancel/batch variations + pause after current)**
- **A2: Storyboard/shot list + promote to timeline**
- **A3: Variations/branching + compare + make canonical**
- **B1–B2: Timeline-aware generation + extend/loop/prepend**
- **F2 + G2: Удаление/архивация/cleanup outputs + relink broken assets**
- **G3: Crash recovery (автосейв + восстановление)**

### Средний тир (быстро усиливает UX и снижает поддержку)

- **C1: Prompt blocks + project defaults**
- **C2: Character/Style/Location cards (простая версия)**
- **C3: Reference board для шотов**
- **F1: Asset intelligence (tags/search/smart filters)**
- **D1: VRAM-aware presets / Fit my hardware**
- **D2: Model hub**
- **G4: Diagnostics/report issue + redaction секретов**

### Нижний тир (позже / по мере зрелости)

- **E1: Backend target: custom server + remote queue + sync** (сильно, но требует много UX/стабильности)
- Пункты, которые вы попросили держать в нижнем тире приоритетов:
  - **Декомпозиция `VideoEditor.tsx`**
  - **Фронтенд‑тесты**
  - **Безопасность dev‑режима: `webSecurity`**
  - **i18n**
  - **a11y**
- **11: ESLint/Prettier + CI lint**
- **12: UX обновлений приложения**
- **R&D направления** (см. раздел выше)

