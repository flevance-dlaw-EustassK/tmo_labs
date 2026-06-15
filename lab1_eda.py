"""
Лабораторная 1 — Разведочный анализ данных и визуализация
Следует выполнить согласно методичке: выбрать датасет (рекомендуется scikit-learn),
провести обзор данных, вычислить основные характеристики, построить визуализации
и корреляционные матрицы.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


def main():
    # Загрузим iris (рекомендуется без пропусков)
    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = pd.Categorical.from_codes(data.target, data.target_names)

    print("Датасет: Ирис (Iris Flower Dataset)")
    print(f"Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nОсновная статистика:")
    print(df.describe())

    # 1. Pairplot
    fig = sns.pairplot(df, hue='target', diag_kind='hist', palette='husl')
    plt.suptitle('Диаграмма рассеяния — Датасет Ирис', y=1.00)
    plt.tight_layout()
    plt.savefig('lab1_pairplot.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 2. Корреляционная матрица
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Матрица корреляции признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lab1_correlation.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 3. Распределение признаков по классам
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(df.select_dtypes(include=[np.number]).columns):
        for target in df['target'].unique():
            data_subset = df[df['target'] == target][col]
            axes[idx].hist(data_subset, alpha=0.6, label=target, bins=15)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Частота')
        axes[idx].set_title(f'Распределение {col}')
        axes[idx].legend()
    plt.tight_layout()
    plt.savefig('lab1_distributions.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 4. Boxplot для сравнения признаков по классам
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x='target', y=col, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{col} по классам')
    plt.tight_layout()
    plt.savefig('lab1_boxplots.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    print("\n✓ Графики сохранены: lab1_*.png")


if __name__ == '__main__':
    main()
