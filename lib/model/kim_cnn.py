from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Dropout, GlobalMaxPool1D, Dense, Input, Embedding, Concatenate
from tensorflow.python.keras.models import Model


class KimCNN:
    def __init__(self, embedding_map, embedding_dim, tokenizer, max_sequence_len, num_classes, dataset_name):
        self.dataset_name = dataset_name
        embedding_layer = Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_map],
                                    input_length=max_sequence_len, trainable=False)
        sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        l_conv1 = Conv1D(100, 3, activation='relu', padding='same')(embedded_sequences)
        l_pool1 = GlobalMaxPool1D()(l_conv1)
        l_conv2 = Conv1D(100, 4, activation='relu', padding='same')(embedded_sequences)
        l_pool2 = GlobalMaxPool1D()(l_conv2)
        l_conv3 = Conv1D(100, 5, activation='relu', padding='same')(embedded_sequences)
        l_pool3 = GlobalMaxPool1D()(l_conv3)

        l_concat = Concatenate()([l_pool1, l_pool2, l_pool3])
        l_dropout1 = Dropout(0.2)(l_concat)
        preds = Dense(num_classes, activation='softmax')(l_dropout1)
        self.model = Model(sequence_input, preds)

    def train(self, train_x, train_y, evaluate_x, evaluate_y):
        early_stopping_callback = EarlyStopping(patience=5, monitor='val_acc')
        checkpoint_callback = ModelCheckpoint(filepath="data/models/cnn/%s.hdf5" % self.dataset_name,
                                              monitor='val_acc', verbose=1, save_best_only=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.fit(train_x, train_y, validation_data=(evaluate_x, evaluate_y), epochs=20, batch_size=64,
                       callbacks=[early_stopping_callback, checkpoint_callback])
        self.model.load_weights("data/models/cnn/%s.hdf5" % self.dataset_name)

    def predict(self, predict_x):
        return self.model.predict(predict_x)

    def evaluate(self, evaluate_x, evaluate_y):
        predict_y = self.predict(evaluate_x).argmax(axis=1)
        evaluate_y = evaluate_y.argmax(axis=1)
        return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
                "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}

    def cross_val(self, data_x, data_y, num_classes, n_splits=5):
        skf = StratifiedKFold(n_splits, shuffle=False, random_state=157)
        print("Performing cross validation (%d-fold)..." % n_splits)
        iteration = 1
        mean_accuracy = 0
        recall_list = [0 for _ in range(num_classes)]
        precision_list = [0 for _ in range(num_classes)]
        for train_index, test_index in skf.split(data_x, data_y.argmax(axis=1)):
            print("Iteration %d of %d" % (iteration, n_splits))
            iteration += 1
            self.train(data_x[train_index], data_y[train_index], data_x[test_index], data_y[test_index])
            metrics = self.evaluate(data_x[test_index], data_y[test_index])
            precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
            recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
            mean_accuracy += metrics['micro-average'][0]
            print("Precision, Recall, F_Score, Support")
            print(metrics)
        print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits,
                                                                         [precision/n_splits for precision in precision_list],
                                                                         [recall/n_splits for recall in recall_list]))
