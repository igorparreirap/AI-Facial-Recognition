import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = 'orl_faces'
IMG_HEIGHT = 112
IMG_WIDTH = 92
N_COMPONENTS_PCA = 50
N_SUBJECTS = 40
N_IMAGES_PER_SUBJECT = 10

def load_data(base_path):
    X_data = []
    y_data = []
    
    print(f"Carregando dados de {base_path}...")
    
    try:
        subject_dirs = [d for d in os.listdir(base_path) if d.startswith('s')]
        subject_dirs.sort(key=lambda x: int(x[1:]))
        
        if len(subject_dirs) != N_SUBJECTS:
            print(f"Aviso: Esperado {N_SUBJECTS} sujeitos, mas {len(subject_dirs)} encontrados.")

        for subject_dir in subject_dirs:
            subject_label = int(subject_dir[1:])
            subject_path = os.path.join(base_path, subject_dir)
            
            if not os.path.isdir(subject_path):
                continue
                
            img_files = [f for f in os.listdir(subject_path) if f.endswith('.pgm')]
            img_files.sort(key=lambda x: int(x.split('.')[0]))

            for img_file in img_files:
                img_path = os.path.join(subject_path, img_file)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Erro ao ler a imagem: {img_path}")
                    continue
                
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                X_data.append(img_resized.flatten())
                y_data.append(subject_label)

        print(f"Carregamento concluído. {len(X_data)} imagens carregadas.")
        
        return np.array(X_data), np.array(y_data)

    except FileNotFoundError:
        print(f"Erro: O diretório '{base_path}' não foi encontrado.")
        print("Certifique-se de que a pasta 'orl_faces' está no mesmo diretório que este script.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o carregamento: {e}")
        return None, None

def get_subject_images(X, y, subject_label):
    indices = np.where(y == subject_label)[0]
    images = [X[i].reshape(IMG_HEIGHT, IMG_WIDTH) for i in indices]
    return images

def create_pipeline():
    pca = PCA(n_components=N_COMPONENTS_PCA, whiten=True, random_state=42)
    
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    
    pipeline = Pipeline(steps=[('pca', pca), ('svm', svm)])
    return pipeline

def run_test_image(pipeline, X_full, y_full, test_image_vector, test_label):
    
    predicted_label = pipeline.predict([test_image_vector])[0]
    probabilities = pipeline.predict_proba([test_image_vector])[0]
    
    classes = pipeline.classes_
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_classes = classes[top_5_indices]
    top_5_probs = probabilities[top_5_indices]

    print(f"\n--- Análise da Imagem de Teste (Sujeito Real: s{test_label}) ---")
    print(f"Previsão do Modelo: s{predicted_label}")
    print("Resultado: " + ("CORRETO" if predicted_label == test_label else "INCORRETO"))
    print("-" * 30)
    
    print("Top-5 Classes Mais Prováveis:")
    for i in range(len(top_5_classes)):
        print(f"  {i+1}. Sujeito s{top_5_classes[i]} (Confiança: {top_5_probs[i]:.2%})")
    
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Análise da Imagem de Teste - Real: s{test_label} | Previsto: s{predicted_label}", fontsize=16)
    
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(test_image_vector.reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
    ax1.set_title(f"Imagem de Teste (s{test_label})")
    ax1.axis('off')

    predicted_subject_images = get_subject_images(X_full, y_full, predicted_label)
    
    display_images = []
    count = 0
    for img in predicted_subject_images:
        if not np.array_equal(img.flatten(), test_image_vector):
            display_images.append(img)
            count += 1
        if count == 9:
            break
            
    if len(display_images) == 0 and len(predicted_subject_images) > 0:
         display_images = predicted_subject_images[:9]

    for i, img in enumerate(display_images):
        ax = plt.subplot(3, 4, i + 2)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Base (s{predicted_label})")
        ax.axis('off')
        if i >= 8:
            break
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_confusion_matrix(pipeline, X, y):
    print("\nCalculando Matriz de Confusão (5-Fold Cross-Validation)...")
    
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv_strategy)
    
    accuracy_cv = accuracy_score(y, y_pred_cv)
    print(f"Acurácia Média na Validação Cruzada (5-fold): {accuracy_cv:.2%}")
    
    cm = confusion_matrix(y, y_pred_cv)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=range(1, N_SUBJECTS + 1), 
                yticklabels=range(1, N_SUBJECTS + 1))
    plt.title(f'Matriz de Confusão (5-Fold CV) - Acurácia: {accuracy_cv:.2%}', fontsize=16)
    plt.xlabel('Classe Prevista (Sujeito)')
    plt.ylabel('Classe Verdadeira (Sujeito)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    tick_interval = 5
    plt.xticks(ticks=np.arange(0.5, N_SUBJECTS, tick_interval), labels=np.arange(1, N_SUBJECTS + 1, tick_interval))
    plt.yticks(ticks=np.arange(0.5, N_SUBJECTS, tick_interval), labels=np.arange(1, N_SUBJECTS + 1, tick_interval))

    plt.tight_layout()
    plt.show()

    return y, y_pred_cv, accuracy_cv

def plot_tsne(X, y):
    print("\nCalculando projeção t-SNE (isso pode levar alguns minutos)...")
    
    pca = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=30.0, max_iter=1000, 
                learning_rate='auto', init='pca', random_state=42)
    
    X_tsne = tsne.fit_transform(X_pca)
    
    plt.figure(figsize=(14, 10))
    cmap = plt.get_cmap('gist_ncar', N_SUBJECTS)
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, alpha=0.8)
    
    plt.title('Projeção t-SNE dos Vetores PCA (Colorido por Sujeito)', fontsize=16)
    plt.xlabel('Componente t-SNE 1')
    plt.ylabel('Componente t-SNE 2')
    
    cbar = plt.colorbar(scatter, ticks=range(1, N_SUBJECTS + 1))
    cbar.set_label('Sujeito (Classe)')
    
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data(BASE_DIR)
    
    if X is None:
        return

    pipeline = create_pipeline()
    
    print("\nTreinando o modelo SVM no conjunto de dados completo...")
    pipeline.fit(X, y)
    print("Treinamento concluído.")

    y_true_cv, y_pred_cv, acc_cv = plot_confusion_matrix(create_pipeline(), X, y)
    
    plot_tsne(X, y)

    print("\n--- Análise de Erros Frequentes (Validação Cruzada) ---")
    erros = np.where(y_true_cv != y_pred_cv)[0]
    print(f"Total de erros na CV: {len(erros)} / {len(y_true_cv)}")
    
    confusoes = {}
    for i in erros:
        par = (y_true_cv[i], y_pred_cv[i])
        confusoes[par] = confusoes.get(par, 0) + 1
    
    confusoes_ordenadas = sorted(confusoes.items(), key=lambda item: item[1], reverse=True)
    
    print("Maiores confusões (Real -> Previsto): Frequência")
    for (real, prev), freq in confusoes_ordenadas[:10]:
        print(f"  s{real} -> s{prev}: {freq} vez(es)")

    print("\n--- TESTE 1 (Acerto Esperado) ---")
    idx_teste_1 = 0
    run_test_image(pipeline, X, y, X[idx_teste_1], y[idx_teste_1])

    if len(erros) > 0:
        print("\n--- TESTE 2 (Erro da CV) ---")
        idx_teste_2 = erros[0]
        run_test_image(pipeline, X, y, X[idx_teste_2], y[idx_teste_2])
    else:
        print("\n--- TESTE 2 (Aleatório) ---")
        idx_teste_2 = np.random.randint(0, len(X))
        run_test_image(pipeline, X, y, X[idx_teste_2], y[idx_teste_2])

    print("\n--- TESTE 3 (Aleatório) ---")
    idx_teste_3 = np.random.randint(0, len(X))
    while idx_teste_3 == idx_teste_2:
        idx_teste_3 = np.random.randint(0, len(X))
    run_test_image(pipeline, X, y, X[idx_teste_3], y[idx_teste_3])


if __name__ == "__main__":
    main()