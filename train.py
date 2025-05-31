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
import pickle
import json

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
    # NUM_LAYERS = 6
    # D_MODEL = 512
    # NUM_HEADS = 6
    # DFF = 768
    # DROPOUT_RATE = 0.3
    # BATCH_SIZE = 512
    # V3
    NUM_LAYERS = 6
    D_MODEL = 1024
    NUM_HEADS = 6
    DFF = 2048
    DROPOUT_RATE = 0.25
    BATCH_SIZE = 64


EPOCHS = 150
LEARNING_RATE = 0.0001  # Reduced from 0.001

def prepare_data(games_df, max_samples=10_000):
    """Prepare chess data for training the transformer model."""
    X = []  # Board states
    y = []  # Next moves (as numbers)

    # check if the unique moves are already saved
    TokenizerSaved = False
    if os.path.exists('move_to_number.pkl') and os.path.exists('number_to_move.pkl'):
        print("Loading existing move dictionaries...")
        with open('move_to_number.pkl', 'rb') as f:
            move_to_number = pickle.load(f)
        with open('number_to_move.pkl', 'rb') as f:
            number_to_move = pickle.load(f)
        TokenizerSaved = True
        print(f"Loaded {len(move_to_number)} unique moves from dictionary")
    else:
        print("No existing move dictionaries found, creating new ones...")
        # Create move dictionaries from the dataset with validation
        unique_moves = set()
        print("Extracting unique moves from dataset...")
        
        for idx, row in games_df.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx} games, found {len(unique_moves)} unique moves")
            
            try:
                moves = row['moves'].split()
                # Validate moves using chess library
                board = chess.Board()
                
                for move_san in moves:
                    try:
                        chess_move = board.parse_san(move_san)
                        if chess_move in board.legal_moves:
                            unique_moves.add(move_san)
                            board.push(chess_move)
                        else:
                            break  # Invalid move, skip rest of game
                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                        break  # Invalid move notation, skip rest of game
            except Exception:
                continue  # Skip problematic games
        
        # Keep original structure - no special tokens
        move_to_number = {move: i for i, move in enumerate(sorted(unique_moves))}
        number_to_move = {i: move for i, move in enumerate(sorted(unique_moves))}

        # save move dictionaries for later use
        with open('move_to_number.pkl', 'wb') as f:
            pickle.dump(move_to_number, f)
        with open('number_to_move.pkl', 'wb') as f:
            pickle.dump(number_to_move, f)
        
        print(f"Total unique moves in dataset: {len(unique_moves)}")
    
    # Limit the number of games for memory efficiency
    sample_games = games_df.sample(min(len(games_df), max_samples))
    
    processed_games = 0
    failed_games = 0
    total_positions = 0
    
    print("Processing games to create training data...")
    for idx, row in sample_games.iterrows():
        try:
            moves_list = row['moves'].split()
            # Skip games that are too short
            if len(moves_list) < 10:
                failed_games += 1
                continue
            
            # Process game to create training examples
            valid_positions = 0
            board = chess.Board()
            
            for i in range(len(moves_list) - 1):
                try:
                    current_move = moves_list[i]
                    next_move = moves_list[i + 1]
                    
                    # Check if both moves are in dictionary
                    if current_move not in move_to_number or next_move not in move_to_number:
                        break
                    
                    # Validate current move is legal before applying
                    chess_move = board.parse_san(current_move)
                    if chess_move not in board.legal_moves:
                        break
                    
                    # Get current board state BEFORE applying the move
                    board_state = board_to_array(board)
                    next_move_token = move_to_number[next_move]
                    
                    X.append(board_state)
                    y.append(next_move_token)
                    valid_positions += 1
                    
                    # Apply the move for next iteration
                    board.push(chess_move)
                        
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                    # Skip this position and break from current game
                    break
            
            if valid_positions > 0:
                processed_games += 1
                total_positions += valid_positions
            else:
                failed_games += 1
                
        except Exception:
            failed_games += 1
            continue
        
        # Print progress every 1000 games
        if (processed_games + failed_games) % 1000 == 0:
            print(f"Processed {processed_games + failed_games} games: "
                  f"{processed_games} successful, {failed_games} failed, "
                  f"{total_positions} positions created")
    
    print(f"Successfully processed {processed_games} games, failed on {failed_games} games")
    print(f"Total training positions: {total_positions}")
    
    if len(X) == 0:
        print("No valid training data generated!")
        return None, None, None, None, None, None
        
    print("Converting to numpy arrays...")
    X = np.array(X, dtype=np.float32)  # Shape: (samples, 8, 8, 19)
    y = np.array(y, dtype=np.int32)    # Keep as integers for sparse categorical crossentropy
    
    print(f"Generated {len(X)} training positions")
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Memory usage: X = {X.nbytes / (1024**3):.2f} GB, y = {y.nbytes / (1024**2):.2f} MB")
    
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

def save_complete_model(model, move_to_number, number_to_move, history=None, model_name="chess_transformer_complete"):
    """Save the complete model with all necessary data for loading."""
    
    # Create directory for the complete model
    model_dir = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model in Keras format (recommended for Keras 3)
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)  # Remove save_format argument
    
    # Calculate best metrics from training history
    best_metrics = {}
    if history is not None and hasattr(history, 'history'):
        hist = history.history
        
        # Best validation metrics (primary) - updated metric names
        if 'val_loss' in hist:
            best_val_loss_epoch = np.argmin(hist['val_loss'])
            best_metrics['best_val_loss'] = float(np.min(hist['val_loss']))
            best_metrics['best_val_loss_epoch'] = int(best_val_loss_epoch)
        
        if 'val_sparse_categorical_accuracy' in hist:
            best_val_acc_epoch = np.argmax(hist['val_sparse_categorical_accuracy'])
            best_metrics['best_val_accuracy'] = float(np.max(hist['val_sparse_categorical_accuracy']))
            best_metrics['best_val_accuracy_epoch'] = int(best_val_acc_epoch)
        
        if 'val_top_5_accuracy' in hist:
            best_val_top5_epoch = np.argmax(hist['val_top_5_accuracy'])
            best_metrics['best_val_top_5_accuracy'] = float(np.max(hist['val_top_5_accuracy']))
            best_metrics['best_val_top_5_accuracy_epoch'] = int(best_val_top5_epoch)
        
        # Best training metrics (for reference) - updated metric names
        if 'loss' in hist:
            best_metrics['best_train_loss'] = float(np.min(hist['loss']))
            best_metrics['best_train_loss_epoch'] = int(np.argmin(hist['loss']))
        
        if 'sparse_categorical_accuracy' in hist:
            best_metrics['best_train_accuracy'] = float(np.max(hist['sparse_categorical_accuracy']))
            best_metrics['best_train_accuracy_epoch'] = int(np.argmax(hist['sparse_categorical_accuracy']))
        
        if 'top_5_accuracy' in hist:
            best_metrics['best_train_top_5_accuracy'] = float(np.max(hist['top_5_accuracy']))
            best_metrics['best_train_top_5_accuracy_epoch'] = int(np.argmax(hist['top_5_accuracy']))
        
        # Final epoch metrics - updated metric names
        best_metrics['final_epoch'] = len(hist['loss']) - 1 if 'loss' in hist else 0
        if 'val_loss' in hist:
            best_metrics['final_val_loss'] = float(hist['val_loss'][-1])
        if 'val_sparse_categorical_accuracy' in hist:
            best_metrics['final_val_accuracy'] = float(hist['val_sparse_categorical_accuracy'][-1])
    
    # Save model metadata
    metadata = {
        "num_moves": len(move_to_number),
        "board_shape": BOARD_SHAPE,
        "model_architecture": "chess_transformer",
        "tensorflow_version": tf.__version__,
        "save_date": datetime.datetime.now().isoformat(),
        "model_parameters": {
            "num_layers": NUM_LAYERS,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "dff": DFF,
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS
        },
        "training_metrics": best_metrics
    }
    
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training history if available
    if history is not None and hasattr(history, 'history'):
        with open(os.path.join(model_dir, "training_history.pkl"), 'wb') as f:
            pickle.dump(history.history, f)
    
    print(f"Complete model saved to: {model_dir}")
    if best_metrics:
        print("Best metrics achieved:")
        for metric, value in best_metrics.items():
            if not metric.endswith('_epoch'):
                print(f"  {metric}: {value:.4f}")
    
    return model_dir

def load_complete_model(model_dir):
    """Load the complete model with all necessary data."""
    
    # Load the model
    model_path = os.path.join(model_dir, "model")
    model = tf.keras.models.load_model(model_path)
    
    # Load move dictionaries
    with open(os.path.join(model_dir, "move_to_number.pkl"), 'rb') as f:
        move_to_number = pickle.load(f)
    
    with open(os.path.join(model_dir, "number_to_move.pkl"), 'rb') as f:
        number_to_move = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded model from: {model_dir}")
    print(f"Model metadata: {metadata}")
    
    return model, move_to_number, number_to_move, metadata

def train_model(model=None):
    """Train the chess transformer model."""
    # Load data
    games_df = pd.read_csv('high_rated_games.csv')
    print(f"Loaded {len(games_df)} games from dataset")
    
    # Prepare data
    result = prepare_data(games_df, max_samples=20_000)
    del games_df  # Free memory after loading data
    if result[0] is None:
        print("Failed to prepare training data. Exiting.")
        return None
        
    X_train, X_test, y_train, y_test, move_to_number, number_to_move = result
    
    print(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples")
    print(f"Board shape: {X_train.shape[1:]}, Number of possible moves: {len(move_to_number)}")
    
    # Use passed model or create new one
    if model is not None:
        print("Using provided model for training...")
        # Verify the model has the correct output shape
        expected_output_shape = len(move_to_number)
        actual_output_shape = model.output.shape[-1]
        
        if actual_output_shape != expected_output_shape:
            print(f"Warning: Model output shape ({actual_output_shape}) doesn't match vocabulary size ({expected_output_shape})")
            print("Creating new model with correct vocabulary size...")
            model = create_chess_transformer_model(len(move_to_number))
        else:
            print(f"Model output shape matches vocabulary size: {expected_output_shape}")
    else:
        print("Creating new model...")
        model = create_chess_transformer_model(len(move_to_number))
    
    # Create optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # Gradient clipping
        epsilon=1e-7
    )
    
    # Custom top-k accuracy metric that works with sparse labels
    def top_5_accuracy(y_true, y_pred):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
    
    # Compile the model with sparse categorical crossentropy
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Use sparse version to avoid one-hot encoding
        metrics=['sparse_categorical_accuracy', top_5_accuracy]
    )
    
    # Model summary
    model.summary()

    # save model at start of training
    model.save('chess_transformer_model.keras')
    print("Initial model saved as 'chess_transformer_model.keras'")
    
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
    
    # Save the complete model with training history
    model_dir = save_complete_model(model, move_to_number, number_to_move, history)
    
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view in browser, run: tensorboard --logdir={log_dir}")
    print(f"Complete model saved to: {model_dir}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
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
    # Define the custom metric function before loading
    def top_5_accuracy(y_true, y_pred):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
    
    # load train model "trained.keras" from "models" directory
    print("--- Starting training process ---")
    model = None
    if os.path.exists('models/trained.keras'):
        try:
            print("Loading existing model...")
            # Load model with custom objects
            # model = tf.keras.models.load_model(
            #     'models/trained.keras',
            #     custom_objects={'top_5_accuracy': top_5_accuracy}
            # )
            # Load latest model
            model = tf.keras.models.load_model(
                'chess_transformer_model.keras',
                custom_objects={'top_5_accuracy': top_5_accuracy}
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will create new model instead.")
            model = None
    else:
        print("No existing model found, will create new one.")

    model, move_to_number, number_to_move = train_model(model=model)    
    # Example: prediction with a sample board
    # In practice, you would use a real board state
    sample_board = np.zeros((8, 8, 19))  # placeholder with 19 channels
    print("Top predicted moves:", predict_next_move(model, sample_board, move_to_number, number_to_move))
    
    # Example of how to load the model later
    print("\n--- Example of loading model ---")
    # Uncomment these lines to test loading:
    # loaded_model, loaded_move_to_number, loaded_number_to_move, metadata = load_complete_model("path_to_saved_model")
    # print("Loaded model successfully!")