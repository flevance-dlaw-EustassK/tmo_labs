"""
Лабораторная 6 — Ансамбли. Часть 2 (Stacking, MLP, опционально GMDH)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def main():
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}\n')

    # Модель 1: Stacking
    print('=== Обучение Stacking классификатора ===')
    estimators = [
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('ab', AdaBoostClassifier(n_estimators=30, random_state=42))
    ]
    
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    stack.fit(X_train, y_train)
    y_pred_stack = stack.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    prec_stack = precision_score(y_test, y_pred_stack)
    rec_stack = recall_score(y_test, y_pred_stack)
    f1_stack = f1_score(y_test, y_pred_stack)
    
    print(f'Точность: {acc_stack:.4f}, Точность класса: {prec_stack:.4f}, Полнота: {rec_stack:.4f}, F1: {f1_stack:.4f}')
    print(classification_report(y_test, y_pred_stack, target_names=['Злокачественная', 'Доброкачественная']))

    # Модель 2: MLP
    print('=== Обучение MLP классификатора ===')
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    prec_mlp = precision_score(y_test, y_pred_mlp)
    rec_mlp = recall_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    
    print(f'Точность: {acc_mlp:.4f}, Точность класса: {prec_mlp:.4f}, Полнота: {rec_mlp:.4f}, F1: {f1_mlp:.4f}')
    print(classification_report(y_test, y_pred_mlp, target_names=['Злокачественная', 'Доброкачественная']))

    # Опционально: GMDH
    try:
        import gmdh
        print('\n=== Обучение модели GMDH ===')
        # Примечание: GMDH не поддерживает классификацию, только регрессию
        print('Примечание: библиотека GMDH доступна, но поддерживает только регрессию (не классификацию)')
    except ImportError:
        print('\nGMDH не установлена; установите через: pip install gmdh')

    # Результаты
    results = {
        'Model': ['Stacking', 'MLP'],
        'Accuracy': [acc_stack, acc_mlp],
        'Precision': [prec_stack, prec_mlp],
        'Recall': [rec_stack, rec_mlp],
        'F1-Score': [f1_stack, f1_mlp]
    }
    results_df = pd.DataFrame(results)

    # График 1: Сравнение метрик Stacking vs MLP
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df['Model']))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Точность', color='skyblue', edgecolor='black')
    ax.bar(x - 0.5*width, results_df['Precision'], width, label='Точность класса', color='lightgreen', edgecolor='black')
    ax.bar(x + 0.5*width, results_df['Recall'], width, label='Полнота', color='lightcoral', edgecolor='black')
    ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', color='lightyellow', edgecolor='black')
    
    ax.set_ylabel('Оценка', fontsize=12, fontweight='bold')
    ax.set_title('Stacking vs MLP - сравнение метрик', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('lab6_stacking_mlp_comparison.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 2: Матрица ошибок для обеих моделей
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stacking confusion matrix
    cm_stack = confusion_matrix(y_test, y_pred_stack)
    sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                cbar_kws={'label': 'Количество'}, xticklabels=['Зло\nкач.', 'Добро\nкач.'], 
                yticklabels=['Зло\nкач.', 'Добро\nкач.'])
    axes[0].set_title('Stacking - матрица ошибок', fontweight='bold')
    axes[0].set_ylabel('Истинный класс')
    axes[0].set_xlabel('Предсказанный класс')
    
    # MLP confusion matrix
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                cbar_kws={'label': 'Количество'}, xticklabels=['Зло\nкач.', 'Добро\nкач.'],
                yticklabels=['Зло\nкач.', 'Добро\nкач.'])
    axes[1].set_title('MLP - матрица ошибок', fontweight='bold')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_xlabel('Предсказанный класс')
    
    plt.tight_layout()
    plt.savefig('lab6_confusion_matrices.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 3: Метрики в виде heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Оценка'}, ax=ax, linewidths=1, linecolor='black')
    ax.set_title('Stacking vs MLP - тепловая карта метрик', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lab6_metrics_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 4: MLP обучение (loss curve если доступно)
    fig, ax = plt.subplots(figsize=(10, 6))
    if hasattr(mlp, 'loss_curve_'):
        ax.plot(mlp.loss_curve_, marker='o', linewidth=2, markersize=4, color='steelblue')
        ax.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
        ax.set_ylabel('Потери', fontsize=12, fontweight='bold')
        ax.set_title('Кривая обучения MLP', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lab6_mlp_loss.png', dpi=100, bbox_inches='tight')
        plt.close('all')

    print('\n✓ Графики сохранены: lab6_*.png')


if __name__ == '__main__':
    main()
