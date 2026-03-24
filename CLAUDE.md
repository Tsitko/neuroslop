# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Учебно-экспериментальный проект по исследованию архитектур нейронных сетей с оптимизацией под Apple Metal. Проект не имеет конечной цели — это площадка для экспериментов.

**Язык:** предпочтительно Swift / Metal Shading Language для максимальной интеграции с GPU, C/C++ допустим для низкоуровневых экспериментов.

## Build & Test

```bash
swift build                    # сборка (debug)
swift build -c release         # сборка (release) — для бенчмарков
swift run neuroslop            # тренировка (usage: neuroslop [naive|accelerate] [train_subset] [test_subset] [epochs])
swift run Tests                # тесты (executable, не swift test — нет Xcode)
swift run Benchmark            # профилирование
swift run -c release neuroslop accelerate  # быстрая тренировка
```

**Debug vs Release:** debug build в ~24x медленнее release (bounds checking, ARC, no inlining). Всегда бенчмаркать в release. Тесты можно в debug.

Reference бенчмарки (PyTorch, MLX): `python3 Reference/bench_pytorch.py [train_n] [test_n] [epochs]`

## Architecture

Swift Package с тремя targets:
- **NeuroslopCore** — библиотека: Matrix (row-major Float32), слои, сеть, бэкенды вычислений
- **neuroslop** — CLI entry point
- **Benchmark** — бенчмарки по операциям

Ключевая абстракция — `ComputeBackend` протокол: единый интерфейс для matmul, activations, element-wise ops. Реализации:
- `CPUBackend` — naive loops, baseline
- `AccelerateBackend` — cblas_sgemm + vDSP
- `MetalBackend` — GPU compute shaders (шейдеры вкомпилированы как Swift string, компилируются в runtime)

`OptimizedMLP` — обходит протокол, pre-allocated буферы + fused ops для максимальной скорости на CPU.

`Layer` протокол — расширяемый интерфейс для гетерогенных слоёв (Dense, будущие KAN/Fourier/Rational). `DenseLayer` конформит.

`Matrix` — row-major `[Float]`, совместим с Metal буферами и Accelerate (cblas_sgemm).

## Project Rules

- **Мониторинг памяти обязателен.** Все тесты, бенчмарки и тренировочные циклы должны отслеживать и логировать потребление памяти (RSS). Особенно критично для MacBook Air 32GB. При работе с Metal — отслеживать allocated GPU memory.

## Hardware

- **Mac Studio M4 Max 128GB** — основная машина для быстрой проверки гипотез
- **MacBook Air M4 32GB** — тестирование оптимизаций и проверка работы на ограниченных ресурсах

## Roadmap (примерный, корректируется по ходу)

1. **Перцептрон и вариации** — базовый перцептрон, случайный перцептрон, перцептрон Гамба, перцептрон с диаметром; комбинации этих подходов. Низкоуровневая реализация, понимание основ.
2. **Свёрточные сети (CNN)** — простые архитектуры, эксперименты с фильтрами и pooling.
3. **Трансформеры** — если дойдём, attention mechanisms, positional encoding.

На каждом этапе — Metal-оптимизация: compute shaders, memory layout, batch processing.

## Metal Optimization Notes

- M4 Max: 40-core GPU, 128GB unified memory, hardware ray tracing (не для нас, но unified memory — ключевое)
- M4 (Air): 10-core GPU, 32GB unified memory
- Unified memory позволяет избежать CPU<->GPU копирования — использовать MTLBuffer с shared storage mode
- Metal Performance Shaders (MPS) — готовые ядра для матричных операций, свёрток
- Для кастомных операций — Metal compute shaders (.metal файлы)
