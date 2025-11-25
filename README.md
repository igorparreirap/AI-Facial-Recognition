# 🧑‍💻 IA-Reconhecimento-Facial-AED3: PCA (Eigenfaces) e SVM


### 🎯 Objetivo do Projeto
Desenvolver um sistema robusto de reconhecimento facial capaz de identificar 40 sujeitos diferentes a partir do dataset ORL Faces. O pipeline utiliza técnicas de aprendizado de máquina e visão computacional para tratar o problema de alta dimensionalidade em imagens.

---

## ⚙️ Metodologia e Pipeline

O projeto foi implementado em Python e segue as etapas de processamento e modelagem clássicas para reconhecimento facial:

### 1. Pré-processamento e Redução de Dimensionalidade (PCA)
Imagens de 92x112 pixels resultam em vetores de $\approx 10.000$ dimensões, tornando o treinamento ineficiente. A solução adotada foi:
* **Achatamento de Imagens:** As imagens são transformadas em vetores de características.
* **PCA (Principal Component Analysis):** O PCA foi aplicado para decompor as imagens em um subespaço de menor dimensão, mantendo a maior parte da variância.
    * **Eigenfaces:** Os vetores próprios (eigenvectors) do PCA representam as "faces características" (Eigenfaces) da base de dados.
    * **Seleção de Componentes:** Foram selecionados **50 Componentes Principais** (`N_COMPONENTS_PCA = 50`) para representar cada imagem, reduzindo drasticamente a dimensionalidade e o ruído.

### 2. Classificação (SVM com Kernel RBF)
Os dados transformados pelo PCA (as projeções nas Eigenfaces) foram usados para treinar um classificador de alta performance:
* **Máquinas de Vetores de Suporte (SVM):** Utilizado para mapear as classes faciais.
* **Kernel RBF (Radial Basis Function):** O kernel RBF foi escolhido para lidar com a separação não linear dos dados no espaço transformado.

### 3. Visualização de Dados (t-SNE)
A técnica **t-SNE (t-distributed Stochastic Neighbor Embedding)** foi aplicada aos dados de validação para visualizar a separação dos 40 sujeitos no espaço de duas dimensões, demonstrando a eficácia da transformação do PCA antes da classificação.

---

## 📈 Resultados e Avaliação

O pipeline de PCA + SVM foi avaliado usando **Cross-Validation (KFold)** com 10 folds para garantir que as métricas refletissem o desempenho real do modelo.

| Métrica | Valor | Interpretação |
| :--- | :--- | :--- |
| **Acurácia (CV)** | **95,75%** | O modelo classificou corretamente quase 96% das faces no conjunto de validação. |
| **Matriz de Confusão** | Detalhada em `reports/` | Revela as classes que são mais frequentemente confundidas (erros são raros e isolados). |

**Gráficos e Análises Salvas em `reports/`:**
* **Matriz de Confusão:** Detalhamento do desempenho da classificação.
* **Projeção t-SNE:** Gráfico de dispersão que mostra a clusterização das classes após a redução de dimensionalidade.

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
1.  **Python 3.x**
2.  **Dataset:** O projeto requer a base de dados **ORL Faces** (400 imagens).
    * Você deve baixar o dataset e colocar a pasta `orl_faces` na **raiz** do projeto (ao lado da pasta `src/`).

### 1. Instalação das Dependências
Instale as bibliotecas de Machine Learning e visão computacional usando o arquivo `requirements.txt`:
```bash
pip install -r requirements.txt