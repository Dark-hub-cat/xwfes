# freetoken-igpu

Пересборка идей [FreeToken](https://github.com/FlashML-org/FreeToken) для архитектуры
**без дискретного GPU**: обычный CPU + интегрированная графика (iGPU), с возможностью
полностью отключить iGPU. Целевая платформа — ноутбуки класса AMD Ryzen 7 7500U /
5700U (Radeon 610M/680M, общая LPDDR5-память).

Ядро — самостоятельный C++20 движок (GGUF → MoE-инференс), написанный как прототип
алгоритмов; параллельно заложен путь гибридизации с llama.cpp для продакшн-квантов.

## Что уже работает

- GGUF v2/v3 (mmap без загрузки в RAM), F32 / F16 / BF16 / Q8_0 / Q4_0 / Q4_1
- MoE-граф qwen3-moe / llama-family (router top-k, shared expert, GQA, RoPE NeoX/Norm)
- **Фрагментная загрузка экспертов**: каждый эксперт гонится через устройство
  кусками нейронов (по умолчанию ~16 МБ окно), double-buffer staging, дефрагментация
  quant-блоков по строкам — работает и на CPU, и на iGPU
- Гетерогенный планировщик CPU/iGPU (`--igpu auto|on|off`, переключение на лету `/igpu`)
- **Регулятор нагрузки GPU/iGPU** (duty-cycle + термозащита + защита вывода на экран)
- Спекулятивное декодирование в стиле вашего MLSD: `ngram` (порт llama-ngram),
  `draft` (модель-драфт), `dflash` (блочный драфт с раундами уточнения),
  `mlsd` = MultiDrafter(ngram+dflash) — аналог `draft-simple,ngram-mod,draft-mtp`
- PrefixCache — аналог Agent State Cache (переиспользование KV префиксов диалога)
- Тесты зелёные: парсер, токенизатор, кванты, эквивалентность чанкера,
  верификатор спекуляции, ngram, сквозная генерация

## Соответствие FreeToken ↔ здесь

| Идея FreeToken | Реализация |
|---|---|
| Эксперты не держатся в VRAM целиком | `ExpertStreamer`: окна по N нейронов |
| Prefetch следующего слоя | staging-окно + бюджет `--staging-mb` |
| Динамическое распределение CPU/GPU | `HeteroScheduler::expert_device()` |
| Agent State Cache | `PrefixCache` (блоки 32 токена, LRU) |
| Desktop GUI | roadmap |

Отличие от дискретной версии: у iGPU память общая с ОС, PCIe-стадии нет вообще —
«загрузка кусочками» решает проблему лимита carve-out BIOS (обычно 512 МБ–4 ГБ),
а не пропускной способности шины.

## Регулятор нагрузки (не изнашиваем GPU, экран продолжает работать)

iGPU обычно питает дисплей: если отдать ему 100%, рабочий стол начнёт заикаться,
ноут уйдёт в троттлинг. Поэтому:

- `--gpu-util 65` — duty-cycle: после каждого окна экспертов вставляются паузы,
  чтобы средняя загрузка ускорителя не превышала заданных процентов
- `--gpu-temp-limit 85` — при достижении порога (Linux hwmon / Windows PDH-утилизация)
  эффективный потолок автоматически снижается шагами по 10% и восстанавливается,
  когда температура падает
- `--no-governor` — отключить (на свой страх)

По умолчанию регулятор включён и ограничивает 65%.

## Сборка

### На GitHub (если на ПК нет компилятора)

1. Создайте репозиторий на github.com и загрузите содержимое этой папки
   (web-интерфейс → Add file → Upload files, либо `git push`).
2. Вкладка **Actions** → workflow `build` запустится сам (или Run workflow вручную).
3. Через ~10–15 минут в завершившемся прогоне скачайте артефакт:
   - `ft-windows-x64.zip` — базовая сборка (CPU-only)
   - `ft-windows-x64-vulkan.zip` — сборка с iGPU-бэкендом (+ `shaders/moe_ffn_chunk.spv`,
     кладите рядом с exe или запускайте из корня проекта)

### Локально (w64devkit, без админ-прав)

```bat
set PATH=%CD%\_toolchain\w64devkit\bin;%PATH%
cmake -S freetoken-igpu -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build\ft-tests.exe
```

Для Vulkan-сборки нужен Vulkan SDK:
`glslangValidator -V shaders/moe_ffn_chunk.comp -o shaders/moe_ffn_chunk.spv`
и `cmake -DFT_WITH_VULKAN=ON`.

## Запуск

```bat
:: CPU-only, без спекуляции
ft-cli.exe --model models\Qwen3-30B-A3B-Q4_K_M.gguf --igpu off --threads 8

:: ngram-спекуляция (zero-cost drafts)
ft-cli.exe --model model.gguf --spec ngram --spec-k 5

:: MLSD-режим: ngram + dflash-драфтер на маленькой модели
ft-cli.exe --model big.gguf --draft-model small.gguf --spec mlsd --spec-k 6

:: с iGPU (эксперты фрагментами через Radeon)
ft-cli.exe --model model.gguf --igpu on --gpu-util 60 --chunk-neurons 2048
```

REPL-команды: `/stats`, `/reset`, `/igpu on|off`, `/quit`.
Ключевые флаги: `--ctx`, `--temp/--top-k/--top-p/--seed`, `-n`,
`--chunk-neurons`, `--staging-mb`, `--p-min`, `--p-split`, `--dflash-rounds`.

## Бюджет под Ryzen 7 7500U (16 ГБ LPDDR5)

| Модель | Файл | RAM | Ожидание |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | ~18 ГБ | впритык на 32 ГБ, не для 16 ГБ | единицы t/s |
| Qwen3-8B Q4_K_M (dense) | ~5 ГБ | комфортно на 16 ГБ | ~5–8 t/s CPU |
| Qwen3-4B Q4_K_M | ~2.5 ГБ | комфортно | ~10+ t/s |

Radeon 610M (2 CU RDNA2) добавит немного: её сила — освободить CPU от attention
и части FFN, а не разогнать генерацию. Честные ориентиры — CPU-замеры ±20%.

## Ограничения и roadmap

1. **Кванты K-серии** (Q4_K/Q5_K/Q6_K/IQ) ядро пока не читает — реальные модели в
   этих квантах откроются в гибриде с llama.cpp (его бэкенды уже умеют iGPU-Vulkan);
   наш слой планировщика/чанкера переносится туда как модуль.
2. Vulkan-бэкенд компилируется в CI, но требует обкатки на живом железе.
3. DFlash — аппроксимация (итерационные раунды на каузальной мини-LM); обученный
   двунаправленный драфтер + MTP-головы — следующий шаг (нужны MTP-GGUF).
4. Attention O(n²) без FlashAttention; down-проекция плотного FFN однопоточная.
5. Токенизатор прототипный (жадный матчинг+BPE).

## Структура

```
include/ft/   types gguf quant backend cpu_backend vulkan_backend chunker
              scheduler throttle graph kv_cache sampler tokenizer engine spec
src/          реализация + main.cpp + tests/
shaders/      moe_ffn_chunk.comp → .spv (Vulkan compute, фазы A/B)
.github/workflows/build.yml
```
