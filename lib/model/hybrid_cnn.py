from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Dropout, GlobalMaxPool1D, Dense, Input, Embedding, Concatenate
from tensorflow.python.keras.models import Model


class HybridCNN:
    def __init__(self, word_embedding_map, code_embedding_map, word_tokenizer, code_tokenizer, model_config):
        self.model_config = model_config
        self.word_tokenizer = word_tokenizer
        self.code_tokenizer = code_tokenizer
        self.word_embedding_map = word_embedding_map
        self.code_embedding_map = code_embedding_map
        self.model = None

    def create_model(self):
        # Declaration for KimCNN-based word encoder
        word_encoder_input = Input(shape=(self.model_config.max_word_len,), dtype='int32')
        word_embedding_layer = Embedding(len(self.word_tokenizer.word_index) + 1, self.model_config.word_embedding_dim,
                                         weights=[self.word_embedding_map], input_length=self.model_config.max_word_len,
                                         trainable=False)
        embedded_word_sequences = word_embedding_layer(word_encoder_input)

        w_conv1 = Conv1D(100, 3, activation='relu', padding='same')(embedded_word_sequences)
        w_pool1 = GlobalMaxPool1D()(w_conv1)
        w_conv2 = Conv1D(100, 4, activation='relu', padding='same')(embedded_word_sequences)
        w_pool2 = GlobalMaxPool1D()(w_conv2)
        w_conv3 = Conv1D(100, 5, activation='relu', padding='same')(embedded_word_sequences)
        w_pool3 = GlobalMaxPool1D()(w_conv3)
        w_concat1 = Concatenate()([w_pool1, w_pool2, w_pool3])
        word_encoder = Model(word_encoder_input, w_concat1)

        # Declaration for KimCNN-based code encoder
        code_encoder_input = Input(shape=(self.model_config.max_code_len,), dtype='int32')
        code_embedding_layer = Embedding(len(self.code_tokenizer.word_index) + 1, self.model_config.code_embedding_dim,
                                         weights=[self.code_embedding_map], input_length=self.model_config.max_code_len,
                                         trainable=False)
        embedded_code_sequences = code_embedding_layer(code_encoder_input)

        c_conv1 = Conv1D(100, 3, activation='relu', padding='same')(embedded_code_sequences)
        c_pool1 = GlobalMaxPool1D()(c_conv1)
        c_conv2 = Conv1D(100, 4, activation='relu', padding='same')(embedded_code_sequences)
        c_pool2 = GlobalMaxPool1D()(c_conv2)
        c_conv3 = Conv1D(100, 5, activation='relu', padding='same')(embedded_code_sequences)
        c_pool3 = GlobalMaxPool1D()(c_conv3)
        c_concat1 = Concatenate()([c_pool1, c_pool2, c_pool3])
        code_encoder = Model(code_encoder_input, c_concat1)

        # Similarity classifier using the word and code encoders
        word_input1 = Input(shape=(self.model_config.max_word_len,), dtype='int32')
        word_input2 = Input(shape=(self.model_config.max_word_len,), dtype='int32')
        code_input1 = Input(shape=(self.model_config.max_code_len,), dtype='int32')
        code_input2 = Input(shape=(self.model_config.max_code_len,), dtype='int32')
        l_concat1 = Concatenate()([word_encoder(word_input1), word_encoder(word_input2),
                                   code_encoder(code_input1), code_encoder(code_input2)])
        l_dense1 = Dense(self.model_config.hidden_dim, activation='relu')(l_concat1)
        l_dropout1 = Dropout(self.model_config.dropout)(l_dense1)
        l_dense2 = Dense(self.model_config.hidden_dim, activation='relu')(l_dropout1)
        l_dropout2 = Dropout(self.model_config.dropout)(l_dense2)
        preds = Dense(self.model_config.num_classes, activation='softmax')(l_dropout2)
        self.model = Model([word_input1, word_input2, code_input1, code_input2], preds)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, train_xw1, train_xw2, train_xc1, train_xc2, train_y, evaluate_xw1, evaluate_xw2, evaluate_xc1,
              evaluate_xc2, evaluate_y, **kwargs):
        iteration = 0
        if 'iteration' in kwargs:
            iteration = kwargs['iteration']
        early_stopping_callback = EarlyStopping(patience=self.model_config.patience, monitor='val_acc')
        checkpoint_callback = ModelCheckpoint(filepath="data/checkpoints/hybrid_cnn/%s_%d.hdf5" %
                                                       (self.model_config.dataset, iteration),
                                                       monitor='val_acc', verbose=1, save_best_only=True)
        self.model.fit([train_xw1, train_xw2, train_xc1, train_xc2], train_y,
                       validation_data=([evaluate_xw1, evaluate_xw2, evaluate_xc1, evaluate_xc2], evaluate_y),
                       epochs=self.model_config.epochs, batch_size=self.model_config.batch_size,
                       callbacks=[early_stopping_callback, checkpoint_callback])
        self.model.load_weights(filepath="data/checkpoints/hybrid_cnn/%s_%d.hdf5" % (self.model_config.dataset, iteration))

    def predict(self, predict_xw1, predict_xw2, predict_xc1, predict_xc2):
        return self.model.predict([predict_xw1, predict_xw2, predict_xc1, predict_xc2])

    def evaluate(self, evaluate_xw1, evaluate_xw2, evaluate_xc1, evaluate_xc2, evaluate_y):
        predict_y = self.predict(evaluate_xw1, evaluate_xw2, evaluate_xc1, evaluate_xc2).argmax(axis=1)
        evaluate_y = evaluate_y.argmax(axis=1)
        return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
                "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}

    def cross_val(self, data_xw1, data_xw2, data_xc1, data_xc2, data_y, n_splits=5):
        skf = StratifiedKFold(n_splits, shuffle=False, random_state=157)
        print("Performing cross validation (%d-fold)..." % n_splits)
        iteration = 1
        mean_accuracy = 0
        recall_list = [0 for _ in range(self.model_config.num_classes)]
        precision_list = [0 for _ in range(self.model_config.num_classes)]
        for train_index, test_index in skf.split(data_xw1, data_y.argmax(axis=1)):
            self.create_model()
            print("Iteration %d of %d" % (iteration, n_splits))
            self.train(data_xw1[train_index], data_xw2[train_index], data_xc1[train_index], data_xc2[train_index],
                       data_y[train_index], data_xw1[test_index], data_xw2[test_index], data_xc1[test_index],
                       data_xc2[test_index], data_y[test_index], iteration=iteration)
            metrics = self.evaluate(data_xw1[test_index], data_xw2[test_index], data_xc1[test_index],
                                    data_xc2[test_index], data_y[test_index])
            precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
            recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
            mean_accuracy += metrics['micro-average'][0]
            print("Precision, Recall, F_Score, Support")
            iteration += 1
            print(metrics)
        print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits,
                                                                         [precision/n_splits for precision in precision_list],
                                                                         [recall/n_splits for recall in recall_list]))
