# ReplicaGH

> A collection of deep learning models for supervised detection of pull request clones

## Getting Started:
### Prerequisites:
* Python 3.7
* Tensorflow 1.12

Apart from Tensorflow and Keras for implementing the deep learning models, the implementation makes use of the classifiers and evaluation metrics implemented in the Scikit-learn library. Punkt Sentence Tokenizer and Word Tokenizer modules from the NLTK library are used to preprocess the corpora, and Gensim is used to pickle the embeddings into binary files. To perform some linear algebra computations and store the intermediate results, the popular Numpy package is used.

### Data:
The data being used for this project is stored in a separate repository here: https://github.com/achyudhk/ReplicaGH-Data. The complete dataset for training the embeddings has more than 1200 repositories and is not a part of this repository due to its size. Further, trained embeddings are also not a part of this repository due to their size and are available upon request.

### Models:
#### KimCNN
An implementation of the single-channel CNN architecture used by Kim et al. to encode pull request descriptions. Pull request descriptions from every pair of pull requests in a dataset sample is treated as a document and is tokenized into sentences. Every sentence is represented as a concatenation of 300-dimensional word vectors for each corresponding token from that sentence.  This resultant matrix is then passed through three temporal convolutional layers in parallel, with filter windows of size 3, 4 and 5. A temporal max-pooling operation is applied to these feature maps to retain the feature with the highest value in every map. All three outputs are concatenated and fed to a fully connected ReLU layer to create a document-level vector representation for the pull request description.

#### HybridCNN
A variation of KimCNN containing two CNN-based encoders in parallel: one for encoding representations from code and one for encoding natural language. 
![Architecture of the HybridCNN Model](https://github.com/achyudhk/ReplicaGH/raw/master/doc/hybrid_cnn.png)

#### GRU-CNN
A modified HybridCNN model that integrates the state-of-the-art code embeddings by Husain et al., extracted from a sequence-to-sequence model for predicting the docstring that corresponds to a Python code snippet.
![Architecture of the GRU-CNN Model](https://github.com/achyudhk/ReplicaGH/raw/master/doc/gru_cnn.png)

## Contributing:
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Ensure any install or build dependencies are removed before the end of the layer when doing a build. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.

## License:
This project is licensed under the GNU General Public License v3.0. Please see the LICENSE.md file for more details.

## References
* Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.
* Allamanis, Miltiadis, et al. "A survey of machine learning for big code and naturalness." ACM Computing Surveys (CSUR) 51.4 (2018): 81.
* Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.
* Yahav, Eran. "From Programs to Interpretable Deep Models and Back." International Conference on Computer Aided Verification. Springer, Cham, 2018.
* Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.
* Yu, Yue, et al. "A dataset of duplicate pull-requests in github." Proceedings of the 15th International Conference on Mining Software Repositories. ACM, 2018.
* Munaiah, Nuthan, et al. "Curating GitHub for engineered software projects." Empirical Software Engineering 22.6 (2017): 3219-3253.
