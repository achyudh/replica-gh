from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Dropout, GlobalMaxPool1D, Dense, Input, Embedding, Concatenate
from tensorflow.python.keras.models import Model


class KimCNN:
    def __init__(self, embedding_map, tokenizer, model_config):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.embedding_map = embedding_map
        self.model = None

    def create_model(self):
        # Declaration for KimCNN-based encoder
        encoder_input = Input(shape=(self.model_config.max_sequence_len,), dtype='int32')
        embedding_layer = Embedding(len(self.tokenizer.word_index) + 1, self.model_config.word_embedding_dim,
                                    weights=[self.embedding_map], input_length=self.model_config.max_sequence_len,
                                    trainable=False)
        embedded_sequences = embedding_layer(encoder_input)

        l_conv1 = Conv1D(100, 3, activation='relu', padding='same')(embedded_sequences)
        l_pool1 = GlobalMaxPool1D()(l_conv1)
        l_conv2 = Conv1D(100, 4, activation='relu', padding='same')(embedded_sequences)
        l_pool2 = GlobalMaxPool1D()(l_conv2)
        l_conv3 = Conv1D(100, 5, activation='relu', padding='same')(embedded_sequences)
        l_pool3 = GlobalMaxPool1D()(l_conv3)
        l_concat1 = Concatenate()([l_pool1, l_pool2, l_pool3])
        encoder = Model(encoder_input, l_concat1)

        # Similarity classifier using the KimCNN-based encoder
        sequence_input1 = Input(shape=(self.model_config.max_sequence_len,), dtype='int32')
        sequence_input2 = Input(shape=(self.model_config.max_sequence_len,), dtype='int32')
        l_concat2 = Concatenate()([encoder(sequence_input1), encoder(sequence_input2)])
        l_dense1 = Dense(self.model_config.hidden_dim, activation='relu')(l_concat2)
        l_dropout1 = Dropout(self.model_config.dropout)(l_dense1)
        preds = Dense(self.model_config.num_classes, activation='softmax')(l_dropout1)
        self.model = Model([sequence_input1, sequence_input2], preds)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, train_x1, train_x2, train_y, evaluate_x1, evaluate_x2, evaluate_y, **kwargs):
        iteration = 0
        if 'iteration' in kwargs:
            iteration = kwargs['iteration']
        early_stopping_callback = EarlyStopping(patience=self.model_config.patience, monitor='val_acc')
        checkpoint_callback = ModelCheckpoint(filepath="data/checkpoints/kim_cnn/%s_%d.hdf5" %
                                                       (self.model_config.dataset, iteration),
                                                       monitor='val_acc', verbose=1, save_best_only=True)
        self.model.fit([train_x1, train_x2], train_y, validation_data=([evaluate_x1, evaluate_x2], evaluate_y),
                       epochs=self.model_config.epochs, batch_size=self.model_config.batch_size,
                       callbacks=[early_stopping_callback, checkpoint_callback])
        self.model.load_weights(filepath="data/checkpoints/kim_cnn/%s_%d.hdf5" % (self.model_config.dataset, iteration))

    def predict(self, predict_x1, predict_x2):
        return self.model.predict([predict_x1, predict_x2])

    def evaluate(self, evaluate_x1, evaluate_x2, evaluate_y):
        predict_y = self.predict(evaluate_x1, evaluate_x2).argmax(axis=1)
        evaluate_y = evaluate_y.argmax(axis=1)
        return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
                "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}

    def cross_val(self, data_x1, data_x2, data_y, n_splits=5):
        skf = StratifiedKFold(n_splits, shuffle=False, random_state=157)
        print("Performing cross validation (%d-fold)..." % n_splits)
        iteration = 1
        mean_accuracy = 0
        recall_list = [0 for _ in range(self.model_config.num_classes)]
        precision_list = [0 for _ in range(self.model_config.num_classes)]
        for train_index, test_index in skf.split(data_x1, data_y.argmax(axis=1)):
            self.create_model()
            print("Iteration %d of %d" % (iteration, n_splits))
            self.train(data_x1[train_index], data_x2[train_index], data_y[train_index],
                       data_x1[test_index], data_x2[test_index], data_y[test_index], iteration=iteration)
            metrics = self.evaluate(data_x1[test_index], data_x2[test_index], data_y[test_index])
            precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
            recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
            mean_accuracy += metrics['micro-average'][0]
            print("Precision, Recall, F_Score, Support")
            iteration += 1
            print(metrics)
        print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits,
                                                                         [precision/n_splits for precision in precision_list],
                                                                         [recall/n_splits for recall in recall_list]))
