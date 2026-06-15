"""
Лабораторная 2 — Обработка пропусков, кодирование категорий, масштабирование
Следовать пунктам методички: выбрать датасет с категориальными признаками и пропусками.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # Используем датасет Titanic из seaborn (есть пропуски и категориальные признаки)
    df = sns.load_dataset('titanic')
    print('=== Информация о датасете ===')
    print(f'Размер: {df.shape}')
    print('\nСтолбцы с пропусками:')
    missing = df.isna().sum()
    print(missing[missing > 0])

    # 1. Визуализация пропусков
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_percent = (df.isna().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    missing_percent.plot(kind='barh', ax=ax, color='salmon')
    ax.set_xlabel('Процент пропусков (%)')
    ax.set_title('Анализ пропусков в данных', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lab2_missing_analysis.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 2. Обработка пропусков
    df_processed = df.copy()
    imputer = SimpleImputer(strategy='median')
    df_processed['age'] = imputer.fit_transform(df_processed[['age']])
    df_processed['fare'] = imputer.fit_transform(df_processed[['fare']])
    df_processed = df_processed.dropna(subset=['embarked'])

    # Визуализация: до и после
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df['age'].hist(bins=30, ax=axes[0], color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0].set_title('Распределение возраста (с пропусками)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Возраст')
    axes[0].set_ylabel('Частота')
    
    df_processed['age'].hist(bins=30, ax=axes[1], color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title('Распределение возраста (после импутации)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Возраст')
    axes[1].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig('lab2_imputation.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 3. Кодирование категориальных признаков
    cat_cols = ['sex', 'embarked', 'class']
    df_encoded = pd.get_dummies(df_processed[cat_cols], drop_first=True)
    print(f'\nРазмер закодированных категориальных признаков: {df_encoded.shape}')
    print('Закодированные столбцы:', df_encoded.columns.tolist())

    # Визуализация категорий
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, col in enumerate(cat_cols):
        df_processed[col].value_counts().plot(kind='bar', ax=axes[idx], color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Распределение "{col}"', fontweight='bold')
        axes[idx].set_ylabel('Количество')
        axes[idx].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('lab2_categorical.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # 4. Сборка таблицы признаков
    features = pd.concat([
        df_processed[['age', 'fare']].fillna(0),
        df_encoded
    ], axis=1)

    # 5. Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features.columns)

    # Визуализация: до и после масштабирования
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    features['age'].hist(bins=20, ax=axes[0], color='orange', edgecolor='black', alpha=0.7)
    axes[0].set_title('Возраст (до масштабирования)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Частота')
    
    X_scaled_df['age'].hist(bins=20, ax=axes[1], color='purple', edgecolor='black', alpha=0.7)
    axes[1].set_title('Возраст (после масштабирования)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Значение')
    axes[1].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig('lab2_scaling.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    print(f'\nРазмер подготовленной матрицы признаков: {X_scaled.shape}')
    print('✓ Графики сохранены: lab2_*.png')


if __name__ == '__main__':
    main()
