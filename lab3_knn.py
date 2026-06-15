"""
Лабораторная 3 — KNN: подготовка выборок, кросс-валидация, подбор гиперпараметров
Следуйте методичке: train_test_split, GridSearchCV и RandomizedSearchCV, две стратегии CV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint


def main():
    # 1. Загрузка и подготовка данных
    data = datasets.load_wine()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}')

    # 2. Обучение исходной модели
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
    pipe.fit(X_train, y_train)
    y_pred_base = pipe.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f'\nБазовый KNN (k=5) точность: {acc_base:.4f}')

    # 3. GridSearchCV
    print('\n=== Поиск по сетке (Grid Search) ===')
    param_grid = {
        'knn__n_neighbors': list(range(1, 21)),
        'knn__weights': ['uniform', 'distance']
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f'Лучшие параметры (Grid): {grid.best_params_}')
    print(f'Лучший CV-скор: {grid.best_score_:.4f}')
    y_pred_grid = grid.best_estimator_.predict(X_test)
    acc_grid = accuracy_score(y_test, y_pred_grid)
    print(f'Точность на тесте (Grid): {acc_grid:.4f}')

    # 4. RandomizedSearchCV с двумя стратегиями CV
    print('\n=== Случайный поиск (Randomized Search) с KFold ===')
    param_dist = {
        'knn__n_neighbors': randint(1, 30),
        'knn__weights': ['uniform', 'distance']
    }
    rand_kfold = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=20, cv=KFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1, random_state=0, verbose=0
    )
    rand_kfold.fit(X_train, y_train)
    print(f'Лучшие параметры (Random, KFold): {rand_kfold.best_params_}')
    print(f'Лучший CV-скор: {rand_kfold.best_score_:.4f}')
    y_pred_rand_kfold = rand_kfold.best_estimator_.predict(X_test)
    acc_rand_kfold = accuracy_score(y_test, y_pred_rand_kfold)
    print(f'Точность на тесте (Random, KFold): {acc_rand_kfold:.4f}')

    print('\n=== Случайный поиск (Randomized Search) со Stratified KFold ===')
    rand_stratified = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        n_jobs=-1, random_state=0, verbose=0
    )
    rand_stratified.fit(X_train, y_train)
    print(f'Лучшие параметры (Random, StratifiedKFold): {rand_stratified.best_params_}')
    print(f'Лучший CV-скор: {rand_stratified.best_score_:.4f}')
    y_pred_rand_stratified = rand_stratified.best_estimator_.predict(X_test)
    acc_rand_stratified = accuracy_score(y_test, y_pred_rand_stratified)
    print(f'Точность на тесте (Random, StratifiedKFold): {acc_rand_stratified:.4f}')

    # 5. Графики
    # График 1: Зависимость accuracy от k (из GridSearch)
    results_df = pd.DataFrame(grid.cv_results_)
    params_uniform = []
    scores_uniform = []
    params_distance = []
    scores_distance = []
    
    for k_val in range(1, 21):
        mask_uni = (results_df['param_knn__n_neighbors'] == k_val) & (results_df['param_knn__weights'] == 'uniform')
        mask_dist = (results_df['param_knn__n_neighbors'] == k_val) & (results_df['param_knn__weights'] == 'distance')
        
        if mask_uni.any():
            params_uniform.append(k_val)
            scores_uniform.append(results_df[mask_uni]['mean_test_score'].values[0])
        if mask_dist.any():
            params_distance.append(k_val)
            scores_distance.append(results_df[mask_dist]['mean_test_score'].values[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(params_uniform, scores_uniform, marker='o', label='uniform (равномерная)', linewidth=2, markersize=6)
    ax.plot(params_distance, scores_distance, marker='s', label='distance (расстояние)', linewidth=2, markersize=6)
    ax.axvline(grid.best_params_['knn__n_neighbors'], color='red', linestyle='--', alpha=0.7, label='Лучший k')
    ax.set_xlabel('Количество соседей (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CV точность', fontsize=12, fontweight='bold')
    ax.set_title('Производительность KNN в зависимости от k (GridSearchCV)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lab3_k_selection.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 2: Сравнение моделей
    models_names = ['Базовый\n(k=5)', 'GridSearch', 'Random\n(KFold)', 'Random\n(Stratified)']
    accuracies = [acc_base, acc_grid, acc_rand_kfold, acc_rand_stratified]
    colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_names, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Точность на тесте', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение моделей KNN', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab3_model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 3: Classification report
    print(f'\n=== Отчет классификации (лучшая модель) ===')
    print(classification_report(y_test, y_pred_grid, target_names=data.target_names))

    print('\n✓ Графики сохранены: lab3_*.png')


if __name__ == '__main__':
    main()
