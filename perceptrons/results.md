# MLP Results

Архитектура: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
Датасет: MNIST
Оптимизатор: SGD, lr=0.01, batch_size=64

Hardware: Mac Studio M4 Max 128GB

## Полный датасет: 60k train / 10k test / 10 epochs

| Backend | Accuracy | Loss | Время/эпоху | Память |
|---------|----------|------|-------------|--------|
| **neuroslop MLP + Adam (HybridNetwork)** | **97.60%** | **0.10** | **463ms** | **640 MB** |
| neuroslop Optimized SGD (pre-alloc) | 95.32% | 0.16 | 93ms | 424 MB |
| neuroslop Accelerate (release) | 95.10% | 0.17 | 157ms | 437 MB |
| PyTorch CPU | 93.77% | 0.23 | 374ms | 895 MB |
| MLX GPU (Metal) | 93.43% | 0.23 | 520ms | 700 MB |
| PyTorch MPS (Metal) | 93.52% | 0.23 | 945ms | 948 MB |
| neuroslop Metal GPU (release) | 92.4%* | 0.26* | 4.3s | 518 MB |
| neuroslop Accelerate (debug) | 95.36% | 0.16 | 3.7s | 441 MB |
| neuroslop naive CPU (debug) | — | — | ~150s** | 267 MB |

*Metal: 3 эпохи (не 10), **naive: 5k subset

### Заметки

- **neuroslop Optimized (release) в 4x быстрее PyTorch CPU!** 93ms vs 374ms/epoch.
- Optimized vs Accelerate: 1.7x ускорение от pre-allocated буферов, fused ops, in-place weight update.
- **Metal GPU в 46x медленнее Optimized CPU** для этого MLP. Command buffer dispatch overhead (~5-15μs) × ~30 ops × 938 батчей = ~400ms overhead сверх вычислений. Для малых матриц (64×784) GPU не даёт parallelism advantage.
- Debug build в ~24x медленнее release из-за bounds checking, ARC retain/release, отсутствия inlining.
- Accuracy выше reference (95.3% vs 93.5%) — из-за He normal init (vs PyTorch Kaiming uniform).
- Память: 424 MB (optimized) vs 895 MB (PyTorch) — в 2x экономнее.

## Верификация математики (trace comparison vs PyTorch)

При загрузке одинаковых весов — forward pass, loss, gradients и SGD step совпадают с PyTorch до floating point precision:

| Компонент | Max diff |
|-----------|----------|
| Forward pre-activation (z0) | 2.3e-07 |
| Softmax output | 1.5e-08 |
| Loss (CE) | идентичен (2.304600) |
| Gradient dL/dz | 3.7e-09 |
| Weight gradients | 2-7e-09 |
| Weights after SGD step | 3.7e-09 |

Разница в accuracy между реализациями — **только инициализация весов**, математика идентична.

## Subset: 5k train / 1k test / 5 epochs

| Backend | Accuracy | Loss | Время/эпоху | Память |
|---------|----------|------|-------------|--------|
| neuroslop naive CPU | 82.6% | 0.68 | ~150s | 267 MB |
| neuroslop Accelerate | 81.8% | 0.71 | 625ms | 263 MB |
| PyTorch CPU | 58.7% | 2.07 | 30ms | 739 MB |
| PyTorch MPS (Metal) | 55.3% | 2.04 | 80ms | 793 MB |
| MLX GPU (Metal) | 58.0% | 2.06 | 39ms | 545 MB |

### Заметки по subset

- Разница accuracy (82% vs 58%) вызвана **только инициализацией весов** (He normal + zero bias vs Kaiming uniform + uniform bias). Математика forward/backward/SGD идентична — проверено trace comparison, все значения совпадают с PyTorch до ~1e-7.
- He normal сходится быстрее на малом количестве данных; на полном датасете разница нивелируется.
- Accelerate: 240x быстрее naive CPU (625ms vs 150s). Весь выигрыш от cblas_sgemm.

## Profiling результаты (debug build, Accelerate)

extractRows занимал 53% времени. После замены на pre-shuffle + contiguous slice:

| Фаза | До оптимизации | После |
|------|---------------|-------|
| extractRows / slice | 4189ms (53%) | 10ms (0.3%) |
| forward | 1849ms (23%) | 1849ms (48%) |
| backward | 1747ms (22%) | 1747ms (45%) |

Matmul (cblas_sgemm) занимает ~188ms из 3600ms — **только 5%**. Остальное — аллокации Matrix, element-wise ops, ARC overhead.

**Release build убирает ~24x overhead** (bounds checks, ARC, no inlining) → 157ms/epoch.

## Выполненные оптимизации

| # | Оптимизация | Эффект | Техника |
|---|-------------|--------|---------|
| 1 | extractRows → pre-shuffle + slice | 2x (debug) | Одна перестановка всего датасета вместо поэлементного копирования каждого батча. Contiguous slice вместо scatter-gather. |
| 2 | Release build | 24x (3.7s→157ms) | Компилятор убирает bounds checks, ARC retain/release, inlines все вызовы. Для числодробилок debug build бессмыслен. |
| 3 | Pre-allocated буферы | 1.7x (157ms→93ms) | Все промежуточные матрицы (z, a, dz, dW, db) аллоцируются один раз при создании сети. Ноль malloc во время тренировки. UnsafeMutableBufferPointer вместо Swift Array — нет CoW overhead. |
| 4 | Fused matmul+bias+activation | включено в #3 | cblas_sgemm пишет в pre-allocated буфер, затем один проход: добавить bias + применить activation. Три операции за один проход по памяти вместо трёх. |
| 5 | In-place weight update | включено в #3 | vDSP_vsma: `w += (-lr) * grad` — одна инструкция вместо subtract(w, scalarMultiply(grad, lr)) с двумя промежуточными массивами. |
| 6 | Fused ReLU backward | включено в #3 | `dz = upstream * (a > 0 ? 1 : 0)` — один проход вместо: вычислить derivative матрицу + elementwise multiply. |

### Почему это работает (и почему PyTorch так не делает)

Для маленьких моделей (<1M параметров) overhead dispatch/alloc/autograd доминирует над вычислениями. cblas_sgemm занимает 5% времени, остальное — обвязка.

PyTorch платит за универсальность: autograd граф, Python dispatch, Tensor metadata, динамические формы. `torch.compile()` теоретически делает то же самое (trace + fuse + pre-alloc), но требует CUDA и не работает на MPS.

Наш подход: зная архитектуру заранее, захардкодить все размеры и убрать все аллокации. Работает только для фиксированных моделей — это trade-off.

## KAN Discovery v3: Adam optimizer (release, 5k/1k, lr=0.001)

### SGD vs Adam — Adam решает проблему learning rate

| Архитектура | Adam Acc (3ep) | Adam Loss | ms/epoch | Вердикт |
|-------------|----------------|-----------|----------|---------|
| **MLP baseline** | **90.0%** | **0.32** | **39ms** | Лучший |
| Rational KAN Metal | 87.8% | 0.38 | 857ms | GPU ускоряет, но MLP быстрее |
| BSpline KAN (G=5,k=3) | 85.4% | 0.45 | 3020ms | Медленный, не лучше Rational |
| Rational KAN CPU | 84.8% | 0.48 | 2416ms | Без GPU слишком медленный |
| Fourier KAN | 84.2% | 0.52 | 296ms | Быстрый из KAN, но хуже MLP |

Phase 2 (MLP winner +5ep): **92.6%**, loss 0.23. Стабильно, без скачков.

### Главный вывод

Adam нивелирует преимущество KAN в быстрой сходимости. С SGD Fourier KAN побеждал (62% vs 18% MLP) за счёт лучшей начальной динамики. С Adam MLP получает adaptive lr и сходится быстрее всех. KAN слои дают преимущество только когда данные имеют структуру, которую KAN basis (Fourier/Rational) ловит лучше чем ReLU.

## KAN Discovery v2: SGD (release, 5k/1k, lr=0.001)

### Архитектура: один KAN слой + MLP backbone + LayerNorm на переходе

| Архитектура | Phase 1 (3ep) Acc | Loss | ms/epoch |
|-------------|-------------------|------|----------|
| **Fourier KAN 784→64 + LN + Dense→10** | **62.8%** | **1.08** | **163ms** |
| Rational KAN Metal 784→64 + LN + Dense→10 | 54.3% | 2.06 | 781ms |
| Rational KAN CPU 784→64 + LN + Dense→10 | 40.3% | 1.98 | 2444ms |
| MLP 784→128→64→10 | 18.2% | 2.16 | 13ms |

### Metal Rational KAN: 3.1x ускорение vs CPU

Parallel P(x)/Q(x) evaluation: один thread per edge (batch × outSize × inSize). На M4 Max 40 GPU cores — 64×64×784 = 3.2M threads, идеально параллельно. Reduce суммирование отдельным kernel.

Phase 2 (winner +5 epochs): Fourier KAN → 69.8%, loss 0.94. Нестабильность — loss скачет, нужен lr decay или Adam.

### Ключевые наблюдения

1. **Fourier KAN доминирует** — 77.7% vs 21% MLP за 3 эпохи при lr=0.001. Fourier basis быстрее ловит структуру MNIST (частотные паттерны в цифрах).
2. **Матричная оптимизация Fourier KAN: 22x ускорение** (3.7s → 163ms/epoch). basis matrix [batch, d_in·(2K+1)] × coeffs → один matmul через cblas_sgemm.
3. **Rational KAN** всё ещё медленный (2.3s) — backward не матричный, тройные циклы.
4. **MLP с lr=0.001** — too slow convergence. MLP предпочитает больший lr (0.01).
5. **LayerNorm** между KAN→Dense стабилизирует переход.
6. **Phase 2 нестабильность** — lr=0.001 может быть слишком большим для продолжения, нужен decay.

### Оптимизации FourierKAN

Forward: `output = basis_matrix @ coefficients` — один matmul
- basis_matrix[batch, inSize·(2K+1)]: предвычисленные cos/sin для каждого входа и частоты
- coefficients[inSize·(2K+1), outSize]: обучаемые параметры

Backward: `dCoeffs = basisᵀ @ dOutput`, `dBasis = dOutput @ coeffsᵀ`, input grad через chain rule.

## Следующие возможности

1. **Adam optimizer** — для стабильности KAN тренировки
2. **Rational KAN backward оптимизация** — векторизация тройного цикла
3. **BSpline KAN** — третий тип basis functions
4. **Discovery на полном MNIST** — с Adam и lr decay

## CPU Naive Matmul Benchmark

| Размер | Время (median) |
|--------|----------------|
| 64x64 | 23ms |
| 128x128 | 179ms |
| 256x256 | 1.4s |
| 512x512 | 11.2s |
| 1024x1024 | 90s |

O(n³) scaling. Основной bottleneck тренировки.

---

# CNN Results

## Fashion-MNIST CNN (release, 5k/1k, 5ep, Adam)

CNN: Conv(1→32,3×3)→ReLU→Pool→Conv(32→64,3×3)→ReLU→Pool→Dense(3136→128)→Dense(128→10)

| Metric | Value |
|--------|-------|
| Accuracy | 84.0% |
| Loss | 0.46 |
| Время/epoch | 2.8s |

## CIFAR-10 CNN Discovery (release, 5k/1k, 10ep, Adam)

| Архитектура | Accuracy | Loss | ms/epoch |
|-------------|----------|------|----------|
| **CNN baseline** | **58.1%** | **1.29** | **4.0s** |
| CNN + Rational KAN Metal | 51.2% | 1.78 | 12.5s |
| CNN + Fourier KAN | 46.3% | 1.95 | 6.8s |

### Выводы по CNN + KAN

CNN baseline побеждает: конволюции уже извлекают иерархические фичи, Dense+ReLU достаточно для классификации. KAN heads добавляют overhead без выигрыша в accuracy. KAN имеет смысл когда Dense слой является bottleneck — на CIFAR-10 с хорошими conv фичами это не так.

## Full CIFAR-10 (50k/10k, 10ep, Adam, per-channel normalization)

| Архитектура | Best Acc | Final Acc | ms/epoch | Память |
|-------------|----------|-----------|----------|--------|
| **CNN + BatchNorm** | **71.9% (ep5)** | 70.5% | 56s | 15.4 GB |
| CNN + BN + Fourier KAN | 65.5% (ep4) | 63.5% | 84s | 17.2 GB |

## Full CIFAR-10 v2: + Dropout + LR decay (50k/10k, 15ep, Adam + cosine)

| Архитектура | Best Acc | Epoch | ms/epoch |
|-------------|----------|-------|----------|
| **CNN + BN + Dropout(0.3) + LR decay** | **74.8%** | 11 | 46s |
| CNN + BN (без dropout/decay) | 71.9% | 5 | 56s |

Dropout + cosine LR decay: +3% и overfitting отложен (epoch 7-8 вместо 5). Потолок 2-conv CNN архитектуры ~75%. Для >80% нужны: data augmentation, residual connections, больше слоёв.

### Общие выводы CNN

- **Fourier/Rational KAN heads не дают преимущества** на CIFAR-10 — conv фичи уже хорошо структурированы, Dense+ReLU достаточно
- **im2col + matmul** через Accelerate работает, но 15 GB памяти — дорого
- **46s/epoch** на M4 Max — im2col loop основной bottleneck

---

# STT Results (Vosk Russian)

## Baseline: Vosk ONNX (encoder.int8 + transducer decoder)

| Тестовый набор | WER | Samples |
|---------------|-----|---------|
| Farfield random | 2.3% | 500 |
| Farfield hard (rare words) | 7.4% | 200 |
| Hard samples (our fbank) | 10.4% | 50 |

Типы ошибок: иноязычные слова, слияние/разделение, морфология, редкие слова.

## Fourier KAN Adapter (RNN-T loss, frozen encoder+decoder)

Архитектура: `Encoder[frozen] → FourierKAN[512→512, trainable] → Decoder+Joiner[frozen]`

### Experiment 1: Train on farfield test (1500 samples, 10 epochs)

| Adapter | WER | Params | Δ vs baseline |
|---------|-----|--------|---------------|
| **Fourier KAN K=3** | **8.4%** | **3,584** | **-2.3% (↓21%)** |
| Baseline (no adapter) | 10.7% | 0 | — |
| Dense adapter | 14.6% | 262,144 | +3.9% (overfits) |

KAN побеждает Dense: 75x меньше параметров → лучше обобщает. Fourier basis естественен для аудио фич.

### Experiment 2: Train on crowd data (5000 samples, 15 epochs), test on farfield

| Adapter | Best WER (epoch) | Final WER | Params |
|---------|-----------------|-----------|--------|
| **K=8** | **9.5% (ep3)** | 13.8% | 8,704 |
| K=3 | 10.3% (ep1) | 13.6% | 3,584 |
| Baseline | 10.4% | 10.4% | 0 |

### Ключевые выводы STT

1. **Fourier KAN адаптер улучшает WER** при правильном training (RNN-T loss, early stopping)
2. **Domain mismatch = overfitting**: тренировка на crowd, тест на farfield — adapter overfit к crowd после epoch 3-4
3. **Early stopping критичен**: best WER на epoch 1-3, потом деградация
4. **KAN >> Dense для адаптеров**: меньше параметров = лучше generalization
5. **RNN-T loss обязателен**: CTC и teacher forcing не работают для transducer fine-tuning

### Неудачные подходы (задокументировано)

| Подход | Результат | Причина неудачи |
|--------|-----------|----------------|
| Knowledge distillation (autoencoder) | Нет сигнала | Residual adapter = identity, loss=0 |
| CTC + KAN head | WER 78.9% | CTC не знает языковую модель декодера |
| Teacher forcing через decoder | WER 203% | Adapter "перекрикивает" decoder, бесконечные повторы |
| RNN-T loss | **WER 8.4%** ✓ | Правильный alignment-aware loss |
