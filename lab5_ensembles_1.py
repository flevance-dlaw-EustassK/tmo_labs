"""
Лабораторная 5 — Ансамбли. Часть 1 (Bagging, RandomForest, AdaBoost, GradientBoosting)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def main():
    data = datasets.load_wine()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}\n')

    models = {
        'Bagging': BaggingClassifier(n_estimators=50, random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=0)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print(f'--- {name} ---')
        print(f'Точность: {acc:.4f}, Точность класса: {prec:.4f}, Полнота: {rec:.4f}, F1: {f1:.4f}')
        print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Конвертируем результаты в DataFrame
    results_df = pd.DataFrame(results)

    # График 1: Сравнение метрик всех моделей
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df['Model']))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Точность', color='skyblue', edgecolor='black')
    ax.bar(x - 0.5*width, results_df['Precision'], width, label='Точность класса', color='lightgreen', edgecolor='black')
    ax.bar(x + 0.5*width, results_df['Recall'], width, label='Полнота', color='lightcoral', edgecolor='black')
    ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', color='lightyellow', edgecolor='black')
    
    ax.set_ylabel('Оценка', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение ансамблевых моделей (датасет Wine)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('lab5_models_comparison.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 2: Accuracy сравнение (радарная диаграмма)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(results_df), endpoint=False).tolist()
    angles += angles[:1]  # замкнуть окружность
    
    accuracies = results_df['Accuracy'].tolist()
    accuracies += accuracies[:1]
    
    ax.plot(angles, accuracies, 'o-', linewidth=2, label='Точность', color='steelblue')
    ax.fill(angles, accuracies, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(results_df['Model'])
    ax.set_ylim([0, 1])
    ax.set_title('Ансамбли - лепестковая диаграмма точности', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('lab5_radar_comparison.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 3: Heatmap метрик
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Оценка'}, ax=ax, linewidths=1, linecolor='black')
    ax.set_title('Ансамбли - тепловая карта метрик', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lab5_metrics_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    print('\n✓ Графики сохранены: lab5_*.png')


if __name__ == '__main__':
    main()
