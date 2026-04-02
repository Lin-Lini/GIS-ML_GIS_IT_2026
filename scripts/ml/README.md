# ML MVP pipeline v2 for GIS_IT_2026

Это обновлённая версия ML-контура после разбора слабых мест первой итерации.

Что исправлено по сути:
- weak labels больше не опираются на literature hit как на прямой positive-сигнал;
- negative class теперь строится через многолетнюю стабильность участка, а не через почти недостижимый one-shot negative;
- валидация пишет предупреждения по тонким inside/outside сэмплам, чтобы не выдавать шаткие годы за серьёзное доказательство;
- в scoring и export добавлены `pred_logit`, `pred_catboost`, `pred_ml_mean`, `pred_ml_final` и `ml_minus_baseline`.

## Этапы

1. `01_build_parcel_year_table.py`
   - собирает parcel-year таблицу из `results/analytics/*_parcel_stats.csv`
   - добавляет temporal features

2. `01b_feature_diagnostics.py`
   - optional
   - missing share, Spearman correlation, high-corr pairs

3. `02_make_weak_labels.py`
   - positive = выраженный деградационный сигнал по текущим и динамическим признакам
   - negative = текущая низкая активность + многолетняя стабильность участка
   - literature mask используется только как ограничение для negative и слабый весовой бонус, а не как прямой target

4. `03_run_hdbscan.py`
   - кластеризация режимов деградации

5. `04_train_prob_models.py`
   - обучает logit и CatBoost
   - пишет OOF-метрики и full-table predictions
   - выбирает `pred_ml_final` автоматически по CV среди logit/catboost

6. `05_validate_model_outputs.py`
   - внутренняя метрика по weak labels
   - внешняя mask-based validation с warning-файлом для thin samples

7. `06_export_front_predictions.py`
   - экспорт parcel-level ML score для фронта
   - добавляет disagreement-поля относительно baseline

## Что запускать

Минимальный рабочий контур:
1. `01_build_parcel_year_table.py`
2. `02_make_weak_labels.py`
3. `03_run_hdbscan.py`
4. `04_train_prob_models.py`
5. `05_validate_model_outputs.py`
6. `06_export_front_predictions.py`

## Важная оговорка

Даже после этого апдейта ML всё ещё сравнивается с `risk_score` как baseline. Нельзя честно заявлять, что ML победил baseline, пока внешняя validation не покажет устойчивый выигрыш, а не пару красивых чисел из перекошенной разметки. Такие уж у нас традиции у данных: сначала врут, потом ломают презентацию.
