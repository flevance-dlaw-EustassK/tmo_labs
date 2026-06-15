"""
Лабораторная 4 — Линейные модели, SVM и деревья решений
Согласно методичке: обучить линейную модель, SVM и дерево; сравнить метрики; визуализировать дерево.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def main():
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Обучающая выборка: {X_train.shape}, Тестовая выборка: {X_test.shape}\n')

    # Словарь моделей
    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }

    results = []

    # Обучение и оценка
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
        print(classification_report(y_test, y_pred))

    # График 1: Сравнение метрик моделей
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df['Model']))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Точность', color='skyblue', edgecolor='black')
    ax.bar(x - 0.5*width, results_df['Precision'], width, label='Точность класса', color='lightgreen', edgecolor='black')
    ax.bar(x + 0.5*width, results_df['Recall'], width, label='Полнота', color='lightcoral', edgecolor='black')
    ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', color='lightyellow', edgecolor='black')
    
    ax.set_ylabel('Оценка', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение моделей: линейная регрессия vs SVM vs дерево решений', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('lab4_model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 2: Важность признаков в дереве решений
    dt = models['Decision Tree']
    importances = dt.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(data.feature_names)[indices])
    ax.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
    ax.set_title('Топ 15 важности признаков (дерево решений)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('lab4_feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # График 3: Визуализация дерева решений (с ограничением глубины)
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(dt, max_depth=4, feature_names=data.feature_names, 
              class_names=['Malignant', 'Benign'], filled=True, ax=ax, 
              fontsize=10, rounded=True)
    plt.tight_layout()
    plt.savefig('lab4_tree_visualization.png', dpi=100, bbox_inches='tight')
    plt.close('all')

    # Вывод правил дерева (для малого дерева)
    print('\n=== Правила дерева решений (глубина <= 3) ===')
    dt_simple = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt_simple.fit(X_train, y_train)
    tree_rules = export_text(dt_simple, feature_names=list(data.feature_names))
    print(tree_rules)

    print('\n✓ Графики сохранены: lab4_*.png')


if __name__ == '__main__':
    main()
