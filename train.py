import pandas as pd
import os
import chess
import tensorflow as tf
import tensorboard
import numpy as np
from process import board_to_array
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected")

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("Built with CUDA: ", tf.test.is_built_with_cuda())
# tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Commented out to fix top_k issue

# Constants
TEST_SMALL_MODEL = True

BOARD_SHAPE = (8, 8, 19)  # Updated to 19 channels
if TEST_SMALL_MODEL == False:
    NUM_LAYERS = 10
    D_MODEL = 1024
    NUM_HEADS = 12
    DFF = 4096
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 200
else:
    # Small model is overfitting
    # NUM_LAYERS = 4
    # D_MODEL = 2048
    # NUM_HEADS = 8
    # DFF = 2048
    # DROPOUT_RATE = 0.1
    # BATCH_SIZE = 256


    # V1
    # NUM_LAYERS = 3
    # D_MODEL = 1024
    # NUM_HEADS = 6
    # DFF = 2048
    # DROPOUT_RATE = 0.2
    # BATCH_SIZE = 512
    # V2
    NUM_LAYERS = 6
    D_MODEL = 512
    NUM_HEADS = 6
    DFF = 768
    DROPOUT_RATE = 0.3
    BATCH_SIZE = 512


EPOCHS = 150
LEARNING_RATE = 0.0001  # Reduced from 0.001

def prepare_data(games_df, max_samples=10000):
    """Prepare chess data for training the transformer model."""
    X = []  # Board states
    y = []  # Next moves (as numbers)
    
    # Create move dictionaries from the dataset
    unique_moves = set()
    for idx, row in games_df.iterrows():
        moves = row['moves'].split()
        for move in moves:
            unique_moves.add(move)
    
    move_to_number = {move: i for i, move in enumerate(unique_moves)}
    number_to_move = {i: move for i, move in enumerate(unique_moves)}
    
    print(f"Total unique moves in dataset: {len(unique_moves)}")
    
    # Limit the number of games for memory efficiency
    sample_games = games_df.sample(min(len(games_df), max_samples))
    
    processed_games = 0
    failed_games = 0
    
    for idx, row in sample_games.iterrows():
        try:
            moves_list = row['moves'].split()
            # Skip games that are too short
            if len(moves_list) < 10:
                continue
            
            # Process game in chunks to create training examples
            valid_positions = 0
            board = chess.Board()
            
            for i in range(len(moves_list) - 1):
                try:
                    # Parse the current move
                    move = moves_list[i]
                    if move in move_to_number:
                        # Get current board state
                        board_state = board_to_array(board)
                        next_move = move_to_number.get(moves_list[i+1])
                        
                        if next_move is not None:
                            X.append(board_state)
                            y.append(next_move)
                            valid_positions += 1
                        
                        # Apply the move to the board
                        chess_move = board.parse_san(move)
                        board.push(chess_move)
                    else:
                        break
                        
                except (ValueError, AssertionError) as e:
                    # Skip this position if there's an error
                    break
            
            if valid_positions > 0:
                processed_games += 1
            else:
                failed_games += 1
                
        except Exception as e:
            failed_games += 1
            continue
    
    print(f"Successfully processed {processed_games} games, failed on {failed_games} games")
    
    if len(X) == 0:
        print("No valid training data generated!")
        return None, None, None, None, None, None
        
    X = np.array(X)  # Shape: (samples, 8, 8, 19)
    y = tf.keras.utils.to_categorical(y, num_classes=len(move_to_number))
    
    print(f"Generated {len(X)} training positions")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, move_to_number, number_to_move

def create_chess_transformer_model(num_moves):
    """Create a chess transformer model optimized for multi-channel board representation."""
    # Input is an 8x8x19 chessboard
    inputs = layers.Input(shape=BOARD_SHAPE)
    
    # Convolutional layers to extract spatial features
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Depthwise separable convolutions for efficiency
    x = layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu',
                              depthwise_initializer='he_normal',
                              pointwise_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(512, (3, 3), padding='same', activation='relu',
                              depthwise_initializer='he_normal',
                              pointwise_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu',
                              depthwise_initializer='he_normal',
                              pointwise_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(D_MODEL // 64, (1, 1), activation='relu',
                      kernel_initializer='he_normal')(x)
    
    # Reshape to sequence format for transformer
    x = layers.Reshape((64, D_MODEL // 64))(x)
    
    # Project to full d_model dimension with proper initialization
    x = layers.Dense(D_MODEL, kernel_initializer='glorot_uniform')(x)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=64, delta=1)
    position_embeddings = layers.Embedding(64, D_MODEL,
                                         embeddings_initializer='uniform')(positions)
    x = x + position_embeddings
    
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Multiple transformer blocks with improved initialization
    for _ in range(NUM_LAYERS):
        # Multi-head attention with proper scaling
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=D_MODEL // NUM_HEADS,
            dropout=DROPOUT_RATE,
            kernel_initializer='glorot_uniform'
        )(x, x)
        
        # Add & Norm with residual scaling
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network with proper initialization
        ffn_output = layers.Dense(DFF, activation='relu',
                                kernel_initializer='he_normal')(x)
        ffn_output = layers.Dense(D_MODEL,
                                kernel_initializer='glorot_uniform')(ffn_output)
        ffn_output = layers.Dropout(DROPOUT_RATE)(ffn_output)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers with proper initialization
    x = layers.Dense(512, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Output layer
    outputs = layers.Dense(num_moves, activation='softmax',
                          kernel_initializer='glorot_uniform')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    """Train the chess transformer model."""
    # Load data
    games_df = pd.read_csv('high_rated_games.csv')
    
    # Prepare data
    result = prepare_data(games_df)
    if result[0] is None:
        print("Failed to prepare training data. Exiting.")
        return None
        
    X_train, X_test, y_train, y_test, move_to_number, number_to_move = result
    
    print(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples")
    print(f"Board shape: {X_train.shape[1:]}, Moves: {y_train.shape[1]}")
    
    # Create the model
    model = create_chess_transformer_model(len(move_to_number))
    
    # Create optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # Gradient clipping
        epsilon=1e-7
    )
    
    # Custom top-k accuracy metric that works with mixed precision
    def top_5_accuracy(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, tf.cast(y_pred, tf.float32), k=5)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', top_5_accuracy]
    )
    
    # Model summary
    model.summary()
    
    # Create TensorBoard log directory with timestamp
    log_dir = f"logs/chess_transformer_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Log weight histograms every epoch
        write_graph=True,  # Visualize the model graph
        write_images=True,  # Log model weights as images
        update_freq='epoch',  # Log metrics every epoch
        profile_batch='500,520'  # Profile a few batches for performance analysis
    )
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Custom callback to log additional metrics
    class CustomMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Log current learning rate
            lr = float(self.model.optimizer.learning_rate)
            logs['learning_rate'] = lr
            
            # Log gradient norm (if available)
            if hasattr(self.model.optimizer, 'clipnorm'):
                logs['gradient_clipnorm'] = float(self.model.optimizer.clipnorm)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                'chess_transformer_model.keras', 
                save_best_only=True
            ),
            lr_scheduler,
            tensorboard_callback,
            CustomMetricsCallback()
        ]
    )
    
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view in browser, run: tensorboard --logdir={log_dir}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('chess_transformer_training_history.png')
    plt.show()
    
    return model, move_to_number, number_to_move

def predict_next_move(model, board_state, move_to_number, number_to_move):
    """Predict the next chess move using the trained model."""
    # Reshape board to match model input (add batch dimension)
    board_input = board_state.reshape(1, 8, 8, 19)
    
    # Get prediction
    pred = model.predict(board_input)[0]
    
    # Get top 5 move predictions
    top_indices = np.argsort(pred)[-5:][::-1]
    top_moves = [(number_to_move[idx], pred[idx]) for idx in top_indices]
    
    return top_moves

if __name__ == "__main__":
    model, move_to_number, number_to_move = train_model()
    
    # Example: prediction with a sample board
    # In practice, you would use a real board state
    sample_board = np.zeros((8, 8, 19))  # placeholder with 19 channels
    print("Top predicted moves:", predict_next_move(model, sample_board, move_to_number, number_to_move))