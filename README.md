# mv-text-summarizer


Steps

1. Segment Dataset
2. Extract Features: Extrai as features dos documentos segmentados e gera os rótulos das sentenças

  ```
  python src/pipeline/main_extract_features.py
  ```
  
3. Create Dataset: Cria o dataset utilizado para treinamento dos algoritmos. Os dados serão normalizados e balanceados.
  
  ```
  python src/pipelines/main_create_dataset.py 
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
  Output Format: Dicionary = {X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: list,
                       y_test: list,
                       X_train_nf: pd.DataFrame,
                       X_test_nf: pd.DataFrame,
                       X_train_embbed: pd.DataFrame,
                       X_test_embbed: pd.DataFrame}
  ```
  
5. Fusão das visões:  As matrixes são adicionadas no dataframe anterior.

  ```
  python src/pipelines/main_autoencoders.py 
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

7. 
