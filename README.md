# Product Matching

The aim of this project is to identify which products have been posted repeatedy from the data. The data itself is taken from [Shopee Product Matching](https://www.kaggle.com/competitions/shopee-product-matching/data), one of Kaggle competition. To follow along, please consider to see the link above.

## Goals and Methods
We can identify the repeatedly posted product from `posting_id`. There are two modalities of data, i.e., tabular and images. We can utilize the tabular data, image data, or both. From this phase, we could identify that we will use three scenarios:
- 1st: Predict using `image_phash` feature from tabular data.
- 2nd: Predict using extracted feature from CNN backbone, grouping the feature embeddings using KNN. This done by leverage the image data.
- 3rd: Predict using extracted feature from `title` with Tf-idf, and computes its similarity from those features.

## 1st
This method is just simply grouping by `label_group`, then aggregating the `posting_id` based on its unique value into a dictionary. We use the dictionary as a mapping function based on `label_group`.

## 2nd
We utilized EfficientNetB0 as a CNN backbone to extract features from the data and do the grouping with KNN. First, we preprocess the image data (batching and resizing). Due to the limitation of computing power, we processed the data in chunk fashion. The architecture for the deep learning models as follows:
- EfficientNetB0
- GlobalAveragePooling2D
- BatchNorm
- Dropout

We directly used the ImageNet weights, without any finetuning step. This is due to the competition rule, which disabling the internet when we submit the notebook. 
The next phase will rely on the KNN algorithm, to group the similar embeddings from our CNN.

## 3rd
The third scenario utilized the data from `title`. We think that we can take advantage of this feature. First, we vectorized the feature and get the text embeddings. The obtained text embeddings is then used to compute the similarity with the help of cosine similarity. We implement this method in a chunk manner due to the limitation of computational power. 

# Result
We obtained a public score of $0.7$ and private score of $0.688$ by combaining all of the three scenrios. The public score result for each scenario (uncombined) as follows:
- 1st: $0.55$
- 2nd: $0.65$
- 3rd: $0.66$