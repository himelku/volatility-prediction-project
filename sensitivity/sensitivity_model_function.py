from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam


def create_model_sensitivity(
    input_shape,
    loss_function="mean_absolute_error",  # Default for volatility regression
    learning_rate=0.001,
    lstm_layers=2,
    activation_function="tanh",
    output_activation="linear",  # linear is best for continuous regression
    dropout_rate=0.1,
):
    """
    Creates an LSTM model for volatility prediction using intraday 15-min data.

    Parameters:
    - input_shape: tuple, shape of input (time_steps, features)
    - loss_function: loss function to compile with (default: 'mean_absolute_error')
    - learning_rate: learning rate for Adam optimizer
    - lstm_layers: number of stacked LSTM layers (default: 2)
    - activation_function: activation function for LSTM layers (default: 'tanh')
    - output_activation: activation for final Dense layer ('linear' recommended for regression)
    - dropout_rate: dropout rate applied after each LSTM layer

    Returns:
    - Compiled Keras Sequential model
    """
    model = Sequential()

    for i in range(lstm_layers):
        if i == 0:
            model.add(
                LSTM(
                    128,
                    return_sequences=True if lstm_layers > 1 else False,
                    activation=activation_function,
                    input_shape=input_shape,
                )
            )
        else:
            model.add(
                LSTM(
                    128,
                    return_sequences=False if i == lstm_layers - 1 else True,
                    activation=activation_function,
                )
            )
        model.add(Dropout(dropout_rate))

    # Output layer for regression (use linear activation)
    model.add(Dense(1, activation=output_activation))

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model
