import pandas as pd
import os
import chess
import tensorflow as tf
import numpy as np
from process import construct_board_from_moves, move_to_number, number_to_move
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Constants
BOARD_SHAPE = (8, 8, 1)
NUM_LAYERS = 12
D_MODEL = 1024
NUM_HEADS = 8
DFF = 4096
DROPOUT_RATE = 0.1
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

def prepare_data(games_df, max_samples=10000):
    """Prepare chess data for training the transformer model."""
    X = []  # Board states
    y = []  # Next moves (as numbers)
    
    # Limit the number of games for memory efficiency
    sample_games = games_df.sample(min(len(games_df), max_samples))
    
    for idx, row in sample_games.iterrows():
        moves_list = row['moves'].split()
        # Skip games that are too short
        if len(moves_list) < 10:
            continue
            
        # Process game in chunks to create training examples
        for i in range(len(moves_list) - 1):
            current_moves = moves_list[:i+1]
            board_state, _ = construct_board_from_moves(current_moves)
            next_move = move_to_number.get(moves_list[i+1])
            
            if next_move is not None:
                X.append(board_state)
                y.append(next_move)
                
    X = np.array(X).reshape(-1, 8, 8, 1)  # Add channel dimension
    y = tf.keras.utils.to_categorical(y, num_classes=len(move_to_number))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_chess_transformer_model(num_moves):
    """Create a simplified chess transformer model using TensorFlow's built-in components."""
    # Input is an 8x8x1 chessboard
    inputs = layers.Input(shape=(8, 8, 1))
    
    # Reshape to sequence and project to d_model
    x = layers.Reshape((64, 1))(inputs)  # (batch_size, 64, 1)
    x = layers.Dense(D_MODEL)(x)  # (batch_size, 64, d_model)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=64, delta=1)
    position_embeddings = layers.Embedding(64, D_MODEL)(positions)
    x = x + position_embeddings
    
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Multiple transformer blocks
    for _ in range(NUM_LAYERS):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=D_MODEL // NUM_HEADS,
            dropout=DROPOUT_RATE
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        ffn_output = layers.Dense(DFF, activation='relu')(x)
        ffn_output = layers.Dense(D_MODEL)(ffn_output)
        ffn_output = layers.Dropout(DROPOUT_RATE)(ffn_output)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Output layer
    outputs = layers.Dense(num_moves, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    """Train the chess transformer model."""
    # Load data
    games_df = pd.read_csv('high_rated_games.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(games_df)
    
    print(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples")
    print(f"Board shape: {X_train.shape[1:]}, Moves: {y_train.shape[1]}")
    
    # Create the model
    model = create_chess_transformer_model(len(move_to_number))
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                'chess_transformer_model.h5', 
                save_best_only=True
            )
        ]
    )
    
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
    
    return model

def predict_next_move(model, board_state):
    """Predict the next chess move using the trained model."""
    # Reshape board to match model input
    board_input = board_state.reshape(1, 8, 8, 1)
    
    # Get prediction
    pred = model.predict(board_input)[0]
    
    # Get top 5 move predictions
    top_indices = np.argsort(pred)[-5:][::-1]
    top_moves = [(number_to_move[idx], pred[idx]) for idx in top_indices]
    
    return top_moves

if __name__ == "__main__":
    model = train_model()
    
    # Example: prediction with a sample board
    # In practice, you would use a real board state
    sample_board = np.zeros((8, 8))  # placeholder
    print("Top predicted moves:", predict_next_move(model, sample_board))