# mv-text-summarizer


Steps

1. Segment Dataset

  ```
  python create_dataset/segmentation.py
  ```

3. Extract Features: Extracts features from segmented documents and generates sentence labels

  ```
  python src/main_extract_features.py
  ```
  
3. Create Dataset: Creates the dataset used for training the algorithms. The data will be normalized and balanced.
  
  ```
  python src/main_create_dataset.py 
  ```
  
 - Input: Feature matrices and list with the name of the files used as test.
  
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
  
4. Create embeddings:  Arrays are added in the previous dataframe

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
  
5. View Fusion:  Arrays are added in the previous dataframe.

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
