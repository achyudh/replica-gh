# Replica-GH

> A collection of deep learning models for supervised detection of pull request clones

## Getting Started:
### Prerequisites:
* Python 3.7
* Tensorflow 1.12

Apart from Tensorflow and Keras for implementing the deep learning models, the implementation makes use of the classifiers and evaluation metrics implemented in the Scikit-learn library. Punkt Sentence Tokenizer and Word Tokenizer modules from the NLTK library are used to preprocess the corpora, and Gensim is used to pickle the embeddings into binary files. To perform some linear algebra computations and store the intermediate results, the popular Numpy package is used.

### Data:
The data being used for this project is stored in a separate repository here: https://github.com/achyudhk/ReplicaGH-Data. The complete dataset for training the embeddings has more than 1200 repositories and is not a part of this repository due to its size. Further, trained embeddings are also not a part of this repository due to their size and are available upon request.

## Contributing:
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Ensure any install or build dependencies are removed before the end of the layer when doing a build. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.

## License:
This project is licensed under the GNU General Public License v3.0. Please see the LICENSE.md file for more details.

## References
* Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.
* Allamanis, Miltiadis, et al. "A survey of machine learning for big code and naturalness." ACM Computing Surveys (CSUR) 51.4 (2018): 81.
* Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.
* Yahav, Eran. "From Programs to Interpretable Deep Models and Back." International Conference on Computer Aided Verification. Springer, Cham, 2018.
* Yu, Yue, et al. "A dataset of duplicate pull-requests in github." Proceedings of the 15th International Conference on Mining Software Repositories. ACM, 2018.
* Munaiah, Nuthan, et al. "Curating GitHub for engineered software projects." Empirical Software Engineering 22.6 (2017): 3219-3253.
