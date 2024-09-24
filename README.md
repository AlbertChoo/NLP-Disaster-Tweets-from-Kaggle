# NLP-Disaster-Tweets-from-Kaggle
To predict which `Tweets` are about real disasters and which ones are not

The `disaster-tweets-datasets` are in `.zip` file mode, unzip it to load the data, or find it on Kaggle, link: https://www.kaggle.com/c/nlp-getting-started/data

To prevent any error, I used old version of `tensorflow==2.14.0` and `tensorflow_hub==0.15.0`.

Basic Steps in `disaster-tweets.ipynb` file:
1. Load Data (check for missing data, value counts, etc)
2. Visualize Data, split train test datasets
3. Convert texts to numbers (Vectorization)
4. Build Model (Baseline, Dense, USE Transfer Learning)
5. Find Most Wrong Predictions
6. Making Predictions on Test dataset
7. Test runtime on different models
8. Make predictions on speciic format based on Kaggle
   - Save as `submission.csv` at last

Accuracy achieved 0.81734, Rank 261 upon successful submission. A lot of space to improve, how about stacked model? Conv1D? Fine-tuning with multiple layers? If runtime too long, try `batch()` and `prefetch()` datasets, from `tf.data.Dataset()` skills, to convert data to TensorFlow tensors.
