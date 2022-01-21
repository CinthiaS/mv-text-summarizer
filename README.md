# mv-text-summarizer


Steps

1. Segment Dataset
2. Extract Features: Extrai as features dos documentos segmentados e gera os rótulos das sentenças

  ```
  python src/main_extract_features.py
  ```
  
3. Create Dataset: Cria o dataset utilizado para treinamento dos algoritmos. Os dados serão normalizados e balanceados.
  
  ```
  python src/main_create_dataset.py 
  ```
  
 - Input: Matrizes de features e lista com o nome dos arquivos utilizados como test.
  
      dataset/introduction.csv
      dataset/materials.csv
      dataset/conclusion.csv
      dataset/indices_summ.csv
  
  ```
  Output Format: Dicionary = {X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: list,
                       y_test: list,
                       X_train_nf: pd.DataFrame,
                       X_test_nf: pd.DataFrame}
  ```
  
4. Create embeddings:  As matrixes são adicionadas no dataframe anterior

  ```
  python src/create_embeddings.py 
  ```

  ```
  Output Format: Dicionary = {X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: list,
                       y_test: list,
                       X_train_nf: pd.DataFrame,
                       X_test_nf: pd.DataFrame,
                       X_train_embbed: pd.DataFrame,
                       X_test_embbed: pd.DataFrame}
  ```
  
5. View Fusion:  As matrixes são adicionadas no dataframe anterior.

  ```
  python src/autoencoders.py 
  ```
  
  ```
  Output Format: Dicionary = {X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: list,
                       y_test: list,
                       X_train_nf: pd.DataFrame,
                       X_test_nf: pd.DataFrame,
                       X_train_embbed: pd.DataFrame,
                       X_test_embbed: pd.DataFrame,
                       X_train_f1: pd.DataFrame,
                       X_test_f1: pd.DataFrame}
  ```

7. Tunning

  ```
  python src/pipeline_tunning.py 
  ```

9. Train Classifiers

  ```
  python src/pipeline_classifiers.py 
  ```
11. Summarization and Evaluate

  ```
  python src/pipeline_summarization.py 
  ```
  
  All process can be executed running main.py
