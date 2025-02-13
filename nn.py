"""
The main code for the recurrent and convolutional networks assignment.
See README.md for details.
"""
from typing import Tuple, List, Dict

import tensorflow


def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # Define the input layer with the specified input shape
    inputs = tensorflow.keras.layers.Input(shape=input_shape)

    # Add the first RNN layer with 128 units and tanh activation
    rnn_layer = tensorflow.keras.layers.SimpleRNN(
        units=128, activation='tanh', return_sequences=True)(inputs)

    # Add the second RNN layer with 64 units and tanh activation
    rnn_layer = tensorflow.keras.layers.SimpleRNN(
        units=64, activation='tanh', return_sequences=True)(rnn_layer)

    # Add a dense layer with 32 units and ReLU activation
    dense_layer = tensorflow.keras.layers.Dense(32, activation='relu')(rnn_layer)

    # Add the output dense layer with linear activation
    outputs = tensorflow.keras.layers.Dense(units=n_outputs, activation='linear')(dense_layer)

    # Compile the model using Adam optimizer and mean squared error loss
    model = tensorflow.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])

    # Define training parameters such as batch size and number of epochs
    fit_kwargs = {
        "batch_size": 16,
        "epochs": 50,
        "verbose": 1
    }

    return model, fit_kwargs

def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # Define the input layer with the specified input shape
    inputs = tensorflow.keras.layers.Input(shape=input_shape)

    # Add a Conv2D layer with 32 filters and 3x3 kernel
    x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Add a MaxPooling2D layer with 2x2 pooling size
    x = tensorflow.keras.layers.MaxPooling2D((2, 2))(x)

    # Add a Conv2D layer with 64 filters and 3x3 kernel
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Add another MaxPooling2D layer with 2x2 pooling size
    x = tensorflow.keras.layers.MaxPooling2D((2, 2))(x)

    # Flatten the feature map for input into dense layers
    x = tensorflow.keras.layers.Flatten()(x)

    # Add a dense layer with 128 units and ReLU activation
    x = tensorflow.keras.layers.Dense(128, activation='relu')(x)

    # Add the output dense layer with softmax activation for classification
    outputs = tensorflow.keras.layers.Dense(n_outputs, activation='softmax')(x)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model = tensorflow.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # Define the input layer with variable-length sequences
    inputs = tensorflow.keras.layers.Input(shape=(None,))

    # Add an Embedding layer to represent tokens with dense vectors
    embedding = tensorflow.keras.layers.Embedding(
        input_dim=len(vocabulary),
        output_dim=512
    )(inputs)

    # Add a Bidirectional LSTM layer for sequence modeling
    bidirectional_lstm = tensorflow.keras.layers.Bidirectional(
        tensorflow.keras.layers.LSTM(
            units=257,
            return_sequences=False,
            dropout=0.3
        )
    )(embedding)

    # Add a Dropout layer for regularization
    dropout = tensorflow.keras.layers.Dropout(0.3)(bidirectional_lstm)

    # Add the output dense layer with activation based on the task
    outputs = tensorflow.keras.layers.Dense(
        units=n_outputs,
        activation='sigmoid' if n_outputs == 1 else 'softmax',
        kernel_regularizer=tensorflow.keras.regularizers.l2(0.01)  # L2 regularization
    )(dropout)

    # Compile the model with Adam optimizer and appropriate loss function
    model = tensorflow.keras.models.Model(inputs, outputs)

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
    loss = 'binary_crossentropy' if n_outputs == 1 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model, {}

def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # Define the input layer with variable-length sequences
    inputs = tensorflow.keras.layers.Input(shape=(None,))

    # Add an Embedding layer to represent tokens with dense vectors
    x = tensorflow.keras.layers.Embedding(len(vocabulary), 128)(inputs)

    # Add a Conv1D layer for feature extraction
    x = tensorflow.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x)

    # Add a GlobalMaxPooling1D layer to reduce sequence dimensions
    x = tensorflow.keras.layers.GlobalMaxPooling1D()(x)

    # Add a Dropout layer for regularization
    x = tensorflow.keras.layers.Dropout(0.5)(x)

    # Add the output dense layer with sigmoid activation for binary classification
    outputs = tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')(x)

    # Compile the model with Adam optimizer and binary crossentropy loss
    model = tensorflow.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, {}
