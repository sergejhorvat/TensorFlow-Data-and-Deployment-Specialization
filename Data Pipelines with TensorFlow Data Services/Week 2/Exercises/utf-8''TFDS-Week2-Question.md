# Classify Structured Data

## Import TensorFlow and Other Libraries


```python
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import feature_column

from os import getcwd
from sklearn.model_selection import train_test_split
```

## Use Pandas to Create a Dataframe

[Pandas](https://pandas.pydata.org/) is a Python library with many helpful utilities for loading and working with structured data. We will use Pandas to download the dataset and load it into a dataframe.


```python
filePath = f"{getcwd()}/../tmp2/heart.csv"
dataframe = pd.read_csv(filePath)
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0</td>
      <td>fixed</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3</td>
      <td>normal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2</td>
      <td>reversible</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0</td>
      <td>normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Split the Dataframe Into Train, Validation, and Test Sets

The dataset we downloaded was a single CSV file. We will split this into train, validation, and test sets.


```python
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
```

    193 train examples
    49 validation examples
    61 test examples


## Create an Input Pipeline Using `tf.data`

Next, we will wrap the dataframes with [tf.data](https://www.tensorflow.org/guide/datasets). This will enable us  to use feature columns as a bridge to map from the columns in the Pandas dataframe to features used to train the model. If we were working with a very large CSV file (so large that it does not fit into memory), we would use tf.data to read it from disk directly.


```python
# EXERCISE: A utility method to create a tf.data dataset from a Pandas Dataframe.

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    
    # Use Pandas dataframe's pop method to get the list of targets.
    labels = dataframe.pop('target')# YOUR CODE HERE
    
    # Create a tf.data.Dataset from the dataframe and labels.
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))# YOUR CODE HERE
    
    if shuffle:
        # Shuffle dataset.
        ds = ds.shuffle(buffer_size = len(dataframe))# YOUR CODE HERE
        
    # Batch dataset with specified batch_size parameter.
    ds = ds.batch(batch_size) # YOUR CODE HERE
    
    return ds
```


```python
batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## Understand the Input Pipeline

Now that we have created the input pipeline, let's call it to see the format of the data it returns. We have used a small batch size to keep the output readable.


```python
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )
```

    Every feature: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    A batch of ages: tf.Tensor([65 34 43 56 53], shape=(5,), dtype=int32)
    A batch of targets: tf.Tensor([0 0 1 0 0], shape=(5,), dtype=int32)


We can see that the dataset returns a dictionary of column names (from the dataframe) that map to column values from rows in the dataframe.

## Create Several Types of Feature Columns

TensorFlow provides many types of feature columns. In this section, we will create several types of feature columns, and demonstrate how they transform a column from the dataframe.


```python
# Try to demonstrate several types of feature columns by getting an example.
example_batch = next(iter(train_ds))[0]
```


```python
# A utility method to create a feature column and to transform a batch of data.
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column, dtype='float64')
    print(feature_layer(example_batch).numpy())
```

### Numeric Columns

The output of a feature column becomes the input to the model (using the demo function defined above, we will be able to see exactly how each column from the dataframe is transformed). A [numeric column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column) is the simplest type of column. It is used to represent real valued features. 


```python
# EXERCISE: Create a numeric feature column out of 'age' and demo it.
age = feature_column.numeric_column('age')# YOUR CODE HERE

demo(age)
```

    [[59.]
     [65.]
     [58.]
     [51.]
     [43.]]


In the heart disease dataset, most columns from the dataframe are numeric.

### Bucketized Columns

Often, you don't want to feed a number directly into the model, but instead split its value into different categories based on numerical ranges. Consider raw data that represents a person's age. Instead of representing age as a numeric column, we could split the age into several buckets using a [bucketized column](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column). 


```python
# EXERCISE: Create a bucketized feature column out of 'age' with
# the following boundaries and demo it.
boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

age_buckets = feature_column.bucketized_column(age,boundaries)# YOUR CODE HERE 

demo(age_buckets)
```

    [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]


Notice the one-hot values above describe which age range each row matches.

### Categorical Columns

In this dataset, thal is represented as a string (e.g. 'fixed', 'normal', or 'reversible'). We cannot feed strings directly to a model. Instead, we must first map them to numeric values. The categorical vocabulary columns provide a way to represent strings as a one-hot vector (much like you have seen above with age buckets). 

**Note**: You will probably see some warning messages when running some of the code cell below. These warnings have to do with software updates and should not cause any errors or prevent your code from running.


```python
# EXERCISE: Create a categorical vocabulary column out of the
# above mentioned categories with the key specified as 'thal'.
thal = feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])# YOUR CODE HERE

# EXERCISE: Create an indicator column out of the created categorical column.
thal_one_hot = feature_column.indicator_column(thal) # YOUR CODE HERE

demo(thal_one_hot)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/feature_column/feature_column_v2.py:4276: IndicatorColumn._variable_shape (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/feature_column/feature_column_v2.py:4331: VocabularyListCategoricalColumn._num_buckets (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
    [[0. 0. 1.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 1. 0.]]


The vocabulary can be passed as a list using [categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list), or loaded from a file using [categorical_column_with_vocabulary_file](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file).

### Embedding Columns

Suppose instead of having just a few possible strings, we have thousands (or more) values per category. For a number of reasons, as the number of categories grow large, it becomes infeasible to train a neural network using one-hot encodings. We can use an embedding column to overcome this limitation. Instead of representing the data as a one-hot vector of many dimensions, an [embedding column](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column) represents that data as a lower-dimensional, dense vector in which each cell can contain any number, not just 0 or 1. You can tune the size of the embedding with the `dimension` parameter.


```python
# EXERCISE: Create an embedding column out of the categorical
# vocabulary you just created (thal). Set the size of the 
# embedding to 8, by using the dimension parameter.

thal_embedding = feature_column.embedding_column(thal, dimension=8) # YOUR CODE HERE


demo(thal_embedding)
```

    [[-0.06833211 -0.10943139 -0.05789141 -0.49784955 -0.38791507  0.3202437
       0.37807843 -0.2375629 ]
     [-0.37373063 -0.18921189  0.4843987   0.17333704  0.41414964 -0.18652162
       0.4873144  -0.04992046]
     [-0.06833211 -0.10943139 -0.05789141 -0.49784955 -0.38791507  0.3202437
       0.37807843 -0.2375629 ]
     [-0.06833211 -0.10943139 -0.05789141 -0.49784955 -0.38791507  0.3202437
       0.37807843 -0.2375629 ]
     [-0.37373063 -0.18921189  0.4843987   0.17333704  0.41414964 -0.18652162
       0.4873144  -0.04992046]]


### Hashed Feature Columns

Another way to represent a categorical column with a large number of values is to use a [categorical_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket). This feature column calculates a hash value of the input, then selects one of the `hash_bucket_size` buckets to encode a string. When using this column, you do not need to provide the vocabulary, and you can choose to make the number of hash buckets significantly smaller than the number of actual categories to save space.


```python
# EXERCISE: Create a hashed feature column with 'thal' as the key and 
# 1000 hash buckets.
thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size = 1000)# YOUR CODE HERE

demo(feature_column.indicator_column(thal_hashed))
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/feature_column/feature_column_v2.py:4331: HashedCategoricalColumn._num_buckets (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


### Crossed Feature Columns
Combining features into a single feature, better known as [feature crosses](https://developers.google.com/machine-learning/glossary/#feature_cross), enables a model to learn separate weights for each combination of features. Here, we will create a new feature that is the cross of age and thal. Note that `crossed_column` does not build the full table of all possible combinations (which could be very large). Instead, it is backed by a `hashed_column`, so you can choose how large the table is.


```python
# EXERCISE: Create a crossed column using the bucketized column (age_buckets),
# the categorical vocabulary column (thal) previously created, and 1000 hash buckets.
crossed_feature = feature_column.crossed_column([age_buckets,thal], hash_bucket_size=1000)# YOUR CODE HERE

demo(feature_column.indicator_column(crossed_feature))
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/feature_column/feature_column_v2.py:4331: CrossedColumn._num_buckets (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be removed in a future version.
    Instructions for updating:
    The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


## Choose Which Columns to Use

We have seen how to use several types of feature columns. Now we will use them to train a model. The goal of this exercise is to show you the complete code needed to work with feature columns. We have selected a few columns to train our model below arbitrarily.

If your aim is to build an accurate model, try a larger dataset of your own, and think carefully about which features are the most meaningful to include, and how they should be represented.


```python
dataframe.dtypes
```




    age           int64
    sex           int64
    cp            int64
    trestbps      int64
    chol          int64
    fbs           int64
    restecg       int64
    thalach       int64
    exang         int64
    oldpeak     float64
    slope         int64
    ca            int64
    thal         object
    target        int64
    dtype: object



You can use the above list of column datatypes to map the appropriate feature column to every column in the dataframe.


```python
# EXERCISE: Fill in the missing code below
feature_columns = []

# Numeric Cols.
# Create a list of numeric columns. Use the following list of columns
# that have a numeric datatype: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca'].
numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']# YOUR CODE HERE

for header in numeric_columns:
    # Create a numeric feature column  out of the header.
    numeric_feature_column = feature_column.numeric_column(header)# YOUR CODE HERE    
    feature_columns.append(numeric_feature_column)

# Bucketized Cols.
# Create a bucketized feature column out of the age column (numeric column)
# that you've already created. Use the following boundaries:
# [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])# YOUR CODE HERE
feature_columns.append(age_buckets)

# Indicator Cols.
# Create a categorical vocabulary column out of the categories
# ['fixed', 'normal', 'reversible'] with the key specified as 'thal'.
thal = feature_column.categorical_column_with_vocabulary_list('thal',['fixed', 'normal', 'reversible'])  # YOUR CODE HERE

# Create an indicator column out of the created thal categorical column
thal_one_hot = feature_column.indicator_column(thal) # YOUR CODE HERE

feature_columns.append(thal_one_hot)

# Embedding Cols.
# Create an embedding column out of the categorical vocabulary you
# just created (thal). Set the size of the embedding to 8, by using
# the dimension parameter.
thal_embedding = feature_column.embedding_column(thal, dimension=8) # YOUR CODE HERE

feature_columns.append(thal_embedding)

# Crossed Cols.
# Create a crossed column using the bucketized column (age_buckets),
# the categorical vocabulary column (thal) previously created, and 1000 hash buckets.
crossed_feature = feature_column.crossed_column([age_buckets,thal], hash_bucket_size=1000) # YOUR CODE HERE

# Create an indicator column out of the crossed column created above to one-hot encode it.
crossed_feature = feature_column.indicator_column(crossed_feature) # YOUR CODE HERE

feature_columns.append(crossed_feature)
```

### Create a Feature Layer

Now that we have defined our feature columns, we will use a [DenseFeatures](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/DenseFeatures) layer to input them to our Keras model.


```python
# EXERCISE: Create a Keras DenseFeatures layer and pass the feature_columns you just created.
feature_layer = tf.keras.layers.DenseFeatures(feature_columns) # YOUR CODE HERE
```

Earlier, we used a small batch size to demonstrate how feature columns worked. We create a new input pipeline with a larger batch size.


```python
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

## Create, Compile, and Train the Model


```python
model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=100)
```

    WARNING:tensorflow:Layer sequential is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    Epoch 1/100
    7/7 [==============================] - 4s 589ms/step - loss: 4.1060 - accuracy: 0.5648 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
    Epoch 2/100
    7/7 [==============================] - 0s 46ms/step - loss: 1.3838 - accuracy: 0.5130 - val_loss: 0.6592 - val_accuracy: 0.5918
    Epoch 3/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.9247 - accuracy: 0.7098 - val_loss: 0.7245 - val_accuracy: 0.8163
    Epoch 4/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.8398 - accuracy: 0.7150 - val_loss: 0.7337 - val_accuracy: 0.5510
    Epoch 5/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.5479 - accuracy: 0.6788 - val_loss: 0.4087 - val_accuracy: 0.8163
    Epoch 6/100
    7/7 [==============================] - 0s 58ms/step - loss: 0.5432 - accuracy: 0.7358 - val_loss: 0.4381 - val_accuracy: 0.8163
    Epoch 7/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4930 - accuracy: 0.7409 - val_loss: 0.3717 - val_accuracy: 0.7959
    Epoch 8/100
    7/7 [==============================] - 0s 69ms/step - loss: 0.5009 - accuracy: 0.7150 - val_loss: 0.4543 - val_accuracy: 0.7551
    Epoch 9/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4854 - accuracy: 0.7098 - val_loss: 0.3605 - val_accuracy: 0.7959
    Epoch 10/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4747 - accuracy: 0.7513 - val_loss: 0.3676 - val_accuracy: 0.8163
    Epoch 11/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.5251 - accuracy: 0.7358 - val_loss: 0.8884 - val_accuracy: 0.4694
    Epoch 12/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.8683 - accuracy: 0.5285 - val_loss: 0.5924 - val_accuracy: 0.8163
    Epoch 13/100
    7/7 [==============================] - 0s 57ms/step - loss: 1.0666 - accuracy: 0.7202 - val_loss: 0.6908 - val_accuracy: 0.5510
    Epoch 14/100
    7/7 [==============================] - 0s 56ms/step - loss: 1.2996 - accuracy: 0.4663 - val_loss: 0.4027 - val_accuracy: 0.8163
    Epoch 15/100
    7/7 [==============================] - 0s 57ms/step - loss: 1.0121 - accuracy: 0.7202 - val_loss: 0.3386 - val_accuracy: 0.8163
    Epoch 16/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.8487 - accuracy: 0.5648 - val_loss: 0.4380 - val_accuracy: 0.7551
    Epoch 17/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.5652 - accuracy: 0.7306 - val_loss: 0.3304 - val_accuracy: 0.7959
    Epoch 18/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.5284 - accuracy: 0.7047 - val_loss: 0.4624 - val_accuracy: 0.7347
    Epoch 19/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4539 - accuracy: 0.7876 - val_loss: 0.3480 - val_accuracy: 0.8571
    Epoch 20/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4367 - accuracy: 0.7617 - val_loss: 0.4373 - val_accuracy: 0.7347
    Epoch 21/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4950 - accuracy: 0.7461 - val_loss: 0.3303 - val_accuracy: 0.7755
    Epoch 22/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.5299 - accuracy: 0.7202 - val_loss: 0.3589 - val_accuracy: 0.8367
    Epoch 23/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.4102 - accuracy: 0.8031 - val_loss: 0.4625 - val_accuracy: 0.7347
    Epoch 24/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4578 - accuracy: 0.8083 - val_loss: 0.3312 - val_accuracy: 0.7755
    Epoch 25/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4687 - accuracy: 0.7461 - val_loss: 0.3546 - val_accuracy: 0.8367
    Epoch 26/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3932 - accuracy: 0.8031 - val_loss: 0.5022 - val_accuracy: 0.6939
    Epoch 27/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.5142 - accuracy: 0.7565 - val_loss: 0.4558 - val_accuracy: 0.8163
    Epoch 28/100
    7/7 [==============================] - 0s 46ms/step - loss: 1.0208 - accuracy: 0.7202 - val_loss: 0.3828 - val_accuracy: 0.8163
    Epoch 29/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.5390 - accuracy: 0.7150 - val_loss: 0.5141 - val_accuracy: 0.6939
    Epoch 30/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4368 - accuracy: 0.7668 - val_loss: 0.3504 - val_accuracy: 0.8163
    Epoch 31/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4189 - accuracy: 0.7720 - val_loss: 0.4696 - val_accuracy: 0.7347
    Epoch 32/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4066 - accuracy: 0.8342 - val_loss: 0.3421 - val_accuracy: 0.7755
    Epoch 33/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.5679 - accuracy: 0.7254 - val_loss: 0.3303 - val_accuracy: 0.8571
    Epoch 34/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.3918 - accuracy: 0.8238 - val_loss: 0.3400 - val_accuracy: 0.8571
    Epoch 35/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3809 - accuracy: 0.7876 - val_loss: 0.3349 - val_accuracy: 0.8776
    Epoch 36/100
    7/7 [==============================] - 0s 69ms/step - loss: 0.3756 - accuracy: 0.8135 - val_loss: 0.4755 - val_accuracy: 0.7347
    Epoch 37/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.4433 - accuracy: 0.7979 - val_loss: 0.3812 - val_accuracy: 0.8163
    Epoch 38/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.6024 - accuracy: 0.7150 - val_loss: 0.4075 - val_accuracy: 0.7755
    Epoch 39/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4164 - accuracy: 0.8083 - val_loss: 0.3454 - val_accuracy: 0.7959
    Epoch 40/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4427 - accuracy: 0.7565 - val_loss: 0.3361 - val_accuracy: 0.8571
    Epoch 41/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.3708 - accuracy: 0.7979 - val_loss: 0.5151 - val_accuracy: 0.6735
    Epoch 42/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.5951 - accuracy: 0.6943 - val_loss: 0.4280 - val_accuracy: 0.8163
    Epoch 43/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.7658 - accuracy: 0.7202 - val_loss: 0.3520 - val_accuracy: 0.7959
    Epoch 44/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.4638 - accuracy: 0.7824 - val_loss: 0.4946 - val_accuracy: 0.6735
    Epoch 45/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.3935 - accuracy: 0.8497 - val_loss: 0.3485 - val_accuracy: 0.8163
    Epoch 46/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4377 - accuracy: 0.7513 - val_loss: 0.3667 - val_accuracy: 0.7551
    Epoch 47/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3557 - accuracy: 0.8394 - val_loss: 0.3292 - val_accuracy: 0.8367
    Epoch 48/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3581 - accuracy: 0.8083 - val_loss: 0.3174 - val_accuracy: 0.8367
    Epoch 49/100
    7/7 [==============================] - 0s 58ms/step - loss: 0.4823 - accuracy: 0.7358 - val_loss: 0.3286 - val_accuracy: 0.8776
    Epoch 50/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.4282 - accuracy: 0.8342 - val_loss: 0.3699 - val_accuracy: 0.7755
    Epoch 51/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.3679 - accuracy: 0.7927 - val_loss: 0.3187 - val_accuracy: 0.8367
    Epoch 52/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.3761 - accuracy: 0.8031 - val_loss: 0.3241 - val_accuracy: 0.8571
    Epoch 53/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4751 - accuracy: 0.7358 - val_loss: 0.3282 - val_accuracy: 0.8571
    Epoch 54/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.3490 - accuracy: 0.8290 - val_loss: 0.3322 - val_accuracy: 0.8776
    Epoch 55/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4155 - accuracy: 0.7824 - val_loss: 0.3054 - val_accuracy: 0.8776
    Epoch 56/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3788 - accuracy: 0.8187 - val_loss: 0.3625 - val_accuracy: 0.7959
    Epoch 57/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3683 - accuracy: 0.8031 - val_loss: 0.3405 - val_accuracy: 0.8163
    Epoch 58/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.4106 - accuracy: 0.8187 - val_loss: 0.3293 - val_accuracy: 0.8367
    Epoch 59/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.4191 - accuracy: 0.7772 - val_loss: 0.3084 - val_accuracy: 0.8776
    Epoch 60/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3626 - accuracy: 0.8031 - val_loss: 0.5397 - val_accuracy: 0.7143
    Epoch 61/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4872 - accuracy: 0.7668 - val_loss: 0.3598 - val_accuracy: 0.8163
    Epoch 62/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.5316 - accuracy: 0.7513 - val_loss: 0.5360 - val_accuracy: 0.7143
    Epoch 63/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.5015 - accuracy: 0.7668 - val_loss: 0.3107 - val_accuracy: 0.8367
    Epoch 64/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3806 - accuracy: 0.8135 - val_loss: 0.3381 - val_accuracy: 0.8163
    Epoch 65/100
    7/7 [==============================] - 0s 45ms/step - loss: 0.3721 - accuracy: 0.7979 - val_loss: 0.3340 - val_accuracy: 0.8367
    Epoch 66/100
    7/7 [==============================] - 0s 55ms/step - loss: 0.3484 - accuracy: 0.8497 - val_loss: 0.3476 - val_accuracy: 0.8163
    Epoch 67/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3393 - accuracy: 0.8446 - val_loss: 0.3330 - val_accuracy: 0.8367
    Epoch 68/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3531 - accuracy: 0.8342 - val_loss: 0.3312 - val_accuracy: 0.8367
    Epoch 69/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.3784 - accuracy: 0.8135 - val_loss: 0.4337 - val_accuracy: 0.7347
    Epoch 70/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.5579 - accuracy: 0.7409 - val_loss: 0.3213 - val_accuracy: 0.7959
    Epoch 71/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4792 - accuracy: 0.7720 - val_loss: 0.4176 - val_accuracy: 0.7347
    Epoch 72/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3755 - accuracy: 0.8135 - val_loss: 0.3109 - val_accuracy: 0.8571
    Epoch 73/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3433 - accuracy: 0.8135 - val_loss: 0.4537 - val_accuracy: 0.7347
    Epoch 74/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3627 - accuracy: 0.8497 - val_loss: 0.3365 - val_accuracy: 0.8367
    Epoch 75/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3639 - accuracy: 0.8187 - val_loss: 0.4048 - val_accuracy: 0.7551
    Epoch 76/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3370 - accuracy: 0.8446 - val_loss: 0.3327 - val_accuracy: 0.8367
    Epoch 77/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.3354 - accuracy: 0.8187 - val_loss: 0.3647 - val_accuracy: 0.7959
    Epoch 78/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3240 - accuracy: 0.8549 - val_loss: 0.3284 - val_accuracy: 0.8163
    Epoch 79/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3179 - accuracy: 0.8497 - val_loss: 0.3495 - val_accuracy: 0.7959
    Epoch 80/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3089 - accuracy: 0.8446 - val_loss: 0.3454 - val_accuracy: 0.8367
    Epoch 81/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.5948 - accuracy: 0.7306 - val_loss: 0.5160 - val_accuracy: 0.7143
    Epoch 82/100
    7/7 [==============================] - 0s 70ms/step - loss: 0.8661 - accuracy: 0.5389 - val_loss: 0.4627 - val_accuracy: 0.8163
    Epoch 83/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.7805 - accuracy: 0.7254 - val_loss: 0.3190 - val_accuracy: 0.8163
    Epoch 84/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4403 - accuracy: 0.7772 - val_loss: 0.3419 - val_accuracy: 0.8367
    Epoch 85/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.4375 - accuracy: 0.7668 - val_loss: 0.3181 - val_accuracy: 0.7959
    Epoch 86/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3637 - accuracy: 0.8290 - val_loss: 0.3427 - val_accuracy: 0.8367
    Epoch 87/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4173 - accuracy: 0.7824 - val_loss: 0.3312 - val_accuracy: 0.7959
    Epoch 88/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3553 - accuracy: 0.8083 - val_loss: 0.3956 - val_accuracy: 0.7959
    Epoch 89/100
    7/7 [==============================] - 0s 46ms/step - loss: 0.3646 - accuracy: 0.8497 - val_loss: 0.2990 - val_accuracy: 0.8980
    Epoch 90/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3468 - accuracy: 0.8238 - val_loss: 0.3476 - val_accuracy: 0.8163
    Epoch 91/100
    7/7 [==============================] - 0s 56ms/step - loss: 0.3357 - accuracy: 0.8446 - val_loss: 0.3231 - val_accuracy: 0.8571
    Epoch 92/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3407 - accuracy: 0.8187 - val_loss: 0.3246 - val_accuracy: 0.8776
    Epoch 93/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.3652 - accuracy: 0.8031 - val_loss: 0.5190 - val_accuracy: 0.7347
    Epoch 94/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.6508 - accuracy: 0.7254 - val_loss: 0.3098 - val_accuracy: 0.8367
    Epoch 95/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4930 - accuracy: 0.7461 - val_loss: 0.3333 - val_accuracy: 0.8163
    Epoch 96/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4469 - accuracy: 0.7927 - val_loss: 0.3062 - val_accuracy: 0.8776
    Epoch 97/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.5628 - accuracy: 0.7617 - val_loss: 0.4915 - val_accuracy: 0.8163
    Epoch 98/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4614 - accuracy: 0.7720 - val_loss: 0.5117 - val_accuracy: 0.7347
    Epoch 99/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4011 - accuracy: 0.8031 - val_loss: 0.3745 - val_accuracy: 0.7959
    Epoch 100/100
    7/7 [==============================] - 0s 57ms/step - loss: 0.4191 - accuracy: 0.7720 - val_loss: 0.3805 - val_accuracy: 0.7551





    <tensorflow.python.keras.callbacks.History at 0x7f65d9d6e208>




```python
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
```

    2/2 [==============================] - 0s 6ms/step - loss: 0.4020 - accuracy: 0.8033
    Accuracy 0.8032787


# Submission Instructions


```python
# Now click the 'Submit Assignment' button above.
```

# When you're done or would like to take a break, please run the two cells below to save your work and close the Notebook. This frees up resources for your fellow learners.


```javascript
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
```


```javascript
%%javascript
<!-- Shutdown and close the notebook -->
window.onbeforeunload = null
window.close();
IPython.notebook.session.delete();
```
