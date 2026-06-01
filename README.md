# Classificação Automática de Comentários NPS com Processamento de Linguagem Natural

## Visão Geral

Este projeto foi desenvolvido como trabalho final da disciplina de Processamento de Linguagem Natural.

O objetivo é automatizar a classificação de comentários de pesquisas NPS de um produto financeiro brasileiro, identificando qual área da empresa deve ser responsável pela análise do feedback recebido.

As categorias disponíveis são:

* Produto
* Suporte
* Serviços
* Pós-venda
* Geral

O pipeline contempla todas as etapas estudadas durante a disciplina:

* Pré-processamento textual
* Vetorização de textos
* Busca textual por similaridade
* Classificação supervisionada
* Extração de entidades
* Distância de Levenshtein
* Construção de grafo de conhecimento
* Deploy do modelo através de API
* Registro de experimentos utilizando MLFlow

---

# Estrutura do Projeto

```text
nps/

├── data
│   ├── raw
│   │   └── nps_data.csv
│   │
│   └── processed
│       ├── preprocessors
│       ├── vectorizers
│       ├── ner
│       └── graph
│
├── models
│
├── src
│   ├── preprocessors
│   ├── vectorizers
│   ├── training
│   ├── ner
│   ├── graph
│   └── api
│   └── data
│   └── model
│   └── pipeline
│   └── utils
│
├── preprocessing_analysis.py
├── vectorizers_analysis.py
├── docker-compose.yml
└── requirements.txt
```

---

# Dataset

O dataset utilizado contém aproximadamente 5.000 respostas de pesquisas NPS.

Features disponíveis:

| Coluna      | Descrição                         |
| ----------- | --------------------------------- |
| comentario  | Comentário informado pelo cliente |
| nota        | Nota NPS (0 a 10)                 |
| responsavel | Área responsável pela análise     |

---

# Instalação

## Criar ambiente virtual

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux:

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Baixar recursos do NLTK

```bash
python -m nltk.downloader stopwords

python -m nltk.downloader punkt

python -m nltk.downloader averaged_perceptron_tagger
```

---

# Pré-processamento

O projeto implementa diversos pré-processamentos para comparação de resultados.

## Executar análise dos pré-processadores

```bash
python preprocessing_analysis.py
```

Arquivos gerados:

```text
data/processed/preprocessors/
```

São gerados CSVs contendo o resultado de:

* Tokenização
* Normalização
* Stemming Porter
* Stemming Snowball
* Lemmatização
* Remoção de Stop-Words
* POS Tagging
* Noise Filter

Objetivo:

Permitir comparação visual entre diferentes abordagens de pré-processamento.

---

# Pipeline Final de Pré-processamento

O pipeline utilizado para treinamento do modelo é:

```text
Remoção de Stop-Words
↓
Normalização
↓
Stemming Snowball
↓
Noise Filter
```

O resultado final é salvo em:

```text
data/processed/nps_training.csv
```

---

# Vetorização

## Executar análise dos vetorizadores

```bash
python vectorizers_analysis.py
```

Arquivos gerados:

```text
data/processed/vectorizers/
```

São avaliadas as seguintes técnicas:

* Bag of Words
* TF-IDF
* TF-IDF + Bigram
* Word2Vec
* Similaridade por Cosseno
* Visualização t-SNE

Objetivo:

Comparar diferentes representações vetoriais para identificar a mais adequada para o problema.

---

# Treinamento dos Modelos

## Naive Bayes

```bash
python src/training/train_naive_bayes.py
```

Executa:

* Vetorização
* Treinamento
* Avaliação

Resultados exibidos:

* Classification Report
* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de Confusão

Também realiza:

* Salvamento do modelo
* Registro no MLFlow

---

## Regressão Logística

```bash
python src/training/train_logistic_regression.py
```

Resultados:

* Classification Report
* Matriz de Confusão

Objetivo:

Comparar desempenho com Naive Bayes.

---

## SVM - Random Search

```bash
python src/training/train_svm_random_search.py
```

Executa:

* RandomizedSearchCV
* Validação Cruzada

Objetivo:

Identificar regiões promissoras do espaço de hiperparâmetros.

---

## SVM - Grid Search

```bash
python src/training/train_svm_grid_search.py
```

Executa:

* GridSearchCV
* Validação Cruzada

Resultados:

* Melhor modelo encontrado
* Melhor conjunto de hiperparâmetros
* Classification Report
* Matriz de Confusão

---

# NER e Extração de Informação

## Extração de Entidades com spaCy

```bash
python src/ner/spacy_ner.py
```

Arquivo gerado:

```text
data/processed/ner/nps_ner_entities.csv
```

Objetivo:

Extrair entidades presentes nos comentários.

---

## Distância de Levenshtein

```bash
python src/ner/levenshtein_matcher.py
```

Arquivo gerado:

```text
data/processed/ner/levenshtein_entities.csv
```

Objetivo:

Padronizar entidades com erros ortográficos e abreviações.

Exemplos:

```text
probelma -> problema

platafroma -> plataforma

atendimnto -> atendimento
```

---

# Grafo de Conhecimento

## Construir Grafo

Pré-requisito:

```text
levenshtein_entities.csv
```

Executar:

```bash
python src/graph/graph_builder.py
```

Arquivos gerados:

```text
data/processed/graph/
```

Objetivo:

Criar uma rede de relacionamentos entre entidades encontradas nos comentários.

---

## Visualizar Grafo

```bash
python src/graph/graph_visualizer.py
```

Objetivo:

Exibir visualmente o grafo criado.

Também são calculadas métricas como:

* Degree Centrality
* Entidades mais importantes
* Relações mais frequentes

---

# API de Predição

Executar:

```bash
python src/api/api.py
```

A API será disponibilizada em:

```text
http://localhost:5000
```

Endpoint:

```http
POST /predict
```

Exemplo de requisição:

```json
{
    "comentario": "O suporte demorou para responder e precisei abrir vários chamados."
}
```

Exemplo de resposta:

```json
{
    "responsavel": "Suporte"
}
```

---

# MLFlow

A interface do MLFlow pode ser acessada em:

```text
http://localhost:5001
```

São registrados:

* Parâmetros
* Métricas
* Artefatos
* Modelo treinado

---

# Docker

## Subir todo o ambiente

```bash
docker-compose up --build
```

O processo executa:

1. Verifica existência do dataset processado
2. Executa pré-processamento se necessário
3. Treina o modelo Naive Bayes
4. Registra o experimento no MLFlow
5. Salva os artefatos do modelo
6. Inicializa a API

Serviços disponíveis:

| Serviço | URL                   |
| ------- | --------------------- |
| API     | http://localhost:5000 |
| MLFlow  | http://localhost:5001 |

---

# Artefatos Gerados

Modelos:

```text
models/
├── naive_bayes.pkl
├── ngram_vectorizer.pkl
└── label_encoder.pkl
```

Pré-processamento:

```text
data/processed/preprocessors/
```

Vetorização:

```text
data/processed/vectorizers/
```

NER:

```text
data/processed/ner/
```

Grafo:

```text
data/processed/graph/
```

---

# Fluxo Completo do Projeto

```text
Dataset Bruto
        ↓
Pré-processamento
        ↓
Vetorização
        ↓
Treinamento
        ↓
Avaliação
        ↓
NER
        ↓
Levenshtein
        ↓
Grafo de Conhecimento
        ↓
MLFlow
        ↓
API REST
```

---

# Autor

Pedro Duarte

Pós-Graduação em Engenharia de IA, Machine Learning e Deep Learning na Infnet

Disciplina: Processamento de Linguagem Natural