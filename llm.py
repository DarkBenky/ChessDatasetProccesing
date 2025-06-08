import pandas as pd
import os
import tensorflow as tf
import tensorboard
import numpy as np
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import pickle
import json
import re
from collections import Counter
import wandb
import pathlib

# Add Hugging Face datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
    print("✅ Hugging Face datasets available")
except ImportError:
    DATASETS_AVAILABLE = False
    print("❌ Hugging Face datasets not available. Install with: pip install datasets")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gups:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"🖥️  Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU detected")

print(f"🔧 TensorFlow version: {tf.__version__}")
print(f"🖥️  GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPUs detected")
print(f"🔧 Built with CUDA: {tf.test.is_built_with_cuda()}")

# Constants
TEST_SMALL_MODEL = True

# Text processing constants
MAX_SEQUENCE_LENGTH = 1024  # Increased from 512 for longer context
VOCAB_SIZE = 50000  # Increased vocabulary size
MIN_WORD_FREQ = 2  # Minimum frequency for word inclusion

if TEST_SMALL_MODEL == False:
    # Very large model configuration for production
    NUM_LAYERS = 24
    D_MODEL = 2048
    NUM_HEADS = 32
    DFF = 8192
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 16
else:
    # Large model configuration (much bigger than before)
    NUM_LAYERS = 4  # Increased from 32 to 48 layers
    D_MODEL = 512   # Increased from 1024 to 2048
    NUM_HEADS = 6  # Increased from 16 to 32 heads
    DFF = 1024       # Increased from 2048 to 8192
    DROPOUT_RATE = 0.1  # Reduced dropout for better performance
    BATCH_SIZE = 8   # Reduced batch size due to larger model

EPOCHS = 3  # Increased epochs for better training
LEARNING_RATE = 0.001  # Reduced learning rate for stability

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Add spaces around punctuation
    text = re.sub(r'([.!?,:;])', r' \1 ', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.!?,:;]', '', text)
    
    return text.strip()

# Try to use a fast subword tokenizer
try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
    print("✅ Using advanced subword BPE tokenizer")
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️  Tokenizers library not installed; using word-level tokenizer")
    print("💡 For better performance, install with: pip install tokenizers")

def build_tokenizer(texts, vocab_size=VOCAB_SIZE, min_freq=MIN_WORD_FREQ):
    """Build tokenizer from text data (BPE if available)."""
    if TOKENIZERS_AVAILABLE:
        print("Building BPE subword tokenizer…")
        # train a Byte-Level BPE on your corpus
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            texts,
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
        )
        # save vocab/merges for later reuse
        os.makedirs("tokenizer_bpe", exist_ok=True)
        tokenizer.save_model("tokenizer_bpe")
        return tokenizer

    # fallback to your original word-level
    print("Building word-level tokenizer…")
    all_words = []
    for t in texts:
        all_words.extend(preprocess_text(t).split())
    counts = Counter(all_words)
    filtered = [w for w, c in counts.items() if c >= min_freq]
    most_common = sorted(filtered, key=lambda w: counts[w], reverse=True)[:vocab_size-4]
    vocab = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN] + most_common
    word_to_id = {w:i for i,w in enumerate(vocab)}
    id_to_word = {i:w for w,i in word_to_id.items()}
    return (word_to_id, id_to_word)

def tokenize_text(text, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """Convert text to token IDs via BPE (or word-level fallback)."""
    if TOKENIZERS_AVAILABLE and isinstance(tokenizer, ByteLevelBPETokenizer):
        # wrap with start/end
        raw = f"{START_TOKEN} {text} {END_TOKEN}"
        encoded = tokenizer.encode(raw)
        ids = encoded.ids[:max_length]
        # pad
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        return ids + [pad_id] * (max_length - len(ids))

    # fallback: tokenizer is (word_to_id, id_to_word)
    if isinstance(tokenizer, tuple) and len(tokenizer) == 2:
        word_to_id, _ = tokenizer
    else:
        # Handle case where tokenizer might be just word_to_id dict
        word_to_id = tokenizer
    
    words = preprocess_text(text).split()
    ids = [word_to_id.get(START_TOKEN)]
    for w in words[: max_length-2]:
        ids.append(word_to_id.get(w, word_to_id[UNK_TOKEN]))
    ids.append(word_to_id.get(END_TOKEN))
    # pad
    pad = word_to_id[PAD_TOKEN]
    return ids + [pad] * (max_length - len(ids))

def prepare_text_data(texts, max_samples=400_000):  # Increased max samples
    """Prepare text data for training the LLM."""
    print("Preparing text data...")
    
    # Check if tokenizer is already saved
    tokenizer_saved = False
    if os.path.exists('word_to_id.pkl') and os.path.exists('id_to_word.pkl'):
        print("Loading existing tokenizer...")
        with open('word_to_id.pkl', 'rb') as f:
            word_to_id = pickle.load(f)
        with open('id_to_word.pkl', 'rb') as f:
            id_to_word = pickle.load(f)
        tokenizer = (word_to_id, id_to_word)
        tokenizer_saved = True
        print(f"Loaded tokenizer with {len(word_to_id)} tokens")
    else:
        print("Building new tokenizer...")
        # Sample texts for tokenizer building if dataset is large
        sample_texts = texts[:min(len(texts), 200_000)]  # Increased sample size
        tokenizer = build_tokenizer(sample_texts)
        
        # Extract word_to_id and id_to_word from tokenizer
        if isinstance(tokenizer, tuple):
            word_to_id, id_to_word = tokenizer
        else:
            # Handle BPE tokenizer case - create mappings
            vocab_size = tokenizer.get_vocab_size()
            word_to_id = {}
            id_to_word = {}
            for i in range(vocab_size):
                token = tokenizer.id_to_token(i)
                if token:
                    word_to_id[token] = i
                    id_to_word[i] = token
        
        # Save tokenizer
        with open('word_to_id.pkl', 'wb') as f:
            pickle.dump(word_to_id, f)
        with open('id_to_word.pkl', 'wb') as f:
            pickle.dump(id_to_word, f)
    
    # Limit samples for memory efficiency
    sample_texts = texts[:min(len(texts), max_samples)]
    
    print("Tokenizing texts...")
    X = []  # Input sequences
    y = []  # Target sequences (shifted by 1)
    
    processed_count = 0
    for i, text in enumerate(sample_texts):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(sample_texts)} texts")
        
        try:
            # Tokenize text - pass the correct tokenizer format
            token_ids = tokenize_text(text, tokenizer)
            
            # Skip if text is too short
            if len([t for t in token_ids if t != word_to_id[PAD_TOKEN]]) < 10:
                continue
            
            # Create input-target pairs for language modeling
            # For proper causal language modeling, we need input and target to be the same length
            # Input: tokens[:-1], Target: tokens[1:] but both padded to MAX_SEQUENCE_LENGTH
            input_seq = token_ids[:-1]  # Remove last token
            target_seq = token_ids[1:]  # Remove first token
            
            # Ensure both sequences are exactly MAX_SEQUENCE_LENGTH
            while len(input_seq) < MAX_SEQUENCE_LENGTH:
                input_seq.append(word_to_id[PAD_TOKEN])
            while len(target_seq) < MAX_SEQUENCE_LENGTH:
                target_seq.append(word_to_id[PAD_TOKEN])
            
            # Truncate if too long
            input_seq = input_seq[:MAX_SEQUENCE_LENGTH]
            target_seq = target_seq[:MAX_SEQUENCE_LENGTH]
            
            X.append(input_seq)
            y.append(target_seq)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing text {i}: {e}")
            continue
    
    print(f"Successfully processed {processed_count} texts")
    
    if len(X) == 0:
        print("No valid training data generated!")
        return None, None, None, None, None, None
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Generated {len(X)} training sequences")
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Memory usage: X = {X.nbytes / (1024**2):.2f} MB, y = {y.nbytes / (1024**2):.2f} MB")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, word_to_id, id_to_word

def create_llm_transformer_model(vocab_size, max_length=MAX_SEQUENCE_LENGTH):
    """Create a transformer-based language model."""
    print(f"Creating LARGE transformer model with {NUM_LAYERS} layers, {D_MODEL} d_model, {NUM_HEADS} heads")
    
    # Input is a sequence of token IDs
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # Token embedding with larger dimensions
    embedding = layers.Embedding(
        vocab_size, 
        D_MODEL,
        mask_zero=True,  # Enable masking for padding tokens
        embeddings_initializer='uniform'
    )(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=max_length, delta=1)
    position_embeddings = layers.Embedding(
        max_length, 
        D_MODEL,
        embeddings_initializer='uniform'
    )(positions)
    
    # Add embeddings
    x = embedding + position_embeddings
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Create static causal attention mask for autoregressive generation
    causal_mask = tf.linalg.band_part(tf.ones((max_length, max_length)), -1, 0)
    causal_mask = tf.cast(causal_mask, tf.float32)
    
    # Multiple transformer blocks with deeper architecture
    for layer_idx in range(NUM_LAYERS):
        # Print progress every 8 layers
        if layer_idx % 8 == 0:
            print(f"Building transformer layer {layer_idx + 1}/{NUM_LAYERS}")
        
        # Multi-head attention with more heads
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=D_MODEL // NUM_HEADS,
            dropout=DROPOUT_RATE,
            kernel_initializer='glorot_uniform'
        )(x, x, attention_mask=causal_mask)
        
        # Add & Norm with layer scaling for deeper models
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network with much larger intermediate dimension
        ffn_output = layers.Dense(
            DFF, 
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        
        # Additional intermediate layer for more capacity
        ffn_output = layers.Dense(
            DFF // 2,
            activation='relu',
            kernel_initializer='he_normal'
        )(ffn_output)
        
        ffn_output = layers.Dense(
            D_MODEL,
            kernel_initializer='glorot_uniform'
        )(ffn_output)
        ffn_output = layers.Dropout(DROPOUT_RATE)(ffn_output)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Add residual scaling for very deep models
        if NUM_LAYERS >= 24:
            # Scale residual connections for better gradient flow
            x = layers.Lambda(lambda tensor: tensor * (2.0 / NUM_LAYERS) ** 0.5)(x)
    
    # Output projection to vocabulary
    outputs = layers.Dense(
        vocab_size,
        kernel_initializer='glorot_uniform'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Print model size information
    total_params = model.count_params()
    print(f"Created model with {total_params:,} parameters ({total_params / 1e6:.1f}M parameters)")
    
    return model

def save_complete_llm_model(model, word_to_id, id_to_word, history=None, model_name="llm_transformer_complete"):
    """Save the complete LLM model with all necessary data."""
    
    # Create directory for the complete model
    model_dir = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)
    
    # Save tokenizer
    with open(os.path.join(model_dir, "word_to_id.pkl"), 'wb') as f:
        pickle.dump(word_to_id, f)
    with open(os.path.join(model_dir, "id_to_word.pkl"), 'wb') as f:
        pickle.dump(id_to_word, f)
    
    # Calculate best metrics from training history
    best_metrics = {}
    if history is not None and hasattr(history, 'history'):
        hist = history.history
        
        # Best validation metrics
        if 'val_loss' in hist:
            best_val_loss_epoch = np.argmin(hist['val_loss'])
            best_metrics['best_val_loss'] = float(np.min(hist['val_loss']))
            best_metrics['best_val_loss_epoch'] = int(best_val_loss_epoch)
            best_metrics['best_val_perplexity'] = float(np.exp(np.min(hist['val_loss'])))
        
        if 'val_sparse_categorical_accuracy' in hist:
            best_val_acc_epoch = np.argmax(hist['val_sparse_categorical_accuracy'])
            best_metrics['best_val_accuracy'] = float(np.max(hist['val_sparse_categorical_accuracy']))
            best_metrics['best_val_accuracy_epoch'] = int(best_val_acc_epoch)
        
        # Best training metrics
        if 'loss' in hist:
            best_metrics['best_train_loss'] = float(np.min(hist['loss']))
            best_metrics['best_train_loss_epoch'] = int(np.argmin(hist['loss']))
            best_metrics['best_train_perplexity'] = float(np.exp(np.min(hist['loss'])))
        
        if 'sparse_categorical_accuracy' in hist:
            best_metrics['best_train_accuracy'] = float(np.max(hist['sparse_categorical_accuracy']))
            best_metrics['best_train_accuracy_epoch'] = int(np.argmax(hist['sparse_categorical_accuracy']))
    
    # Save model metadata
    metadata = {
        "vocab_size": len(word_to_id),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "model_architecture": "llm_transformer",
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
        "training_metrics": best_metrics,
        "special_tokens": {
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "start_token": START_TOKEN,
            "end_token": END_TOKEN
        }
    }
    
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training history if available
    if history is not None and hasattr(history, 'history'):
        with open(os.path.join(model_dir, "training_history.pkl"), 'wb') as f:
            pickle.dump(history.history, f)
    
    print(f"Complete LLM model saved to: {model_dir}")
    if best_metrics:
        print("Best metrics achieved:")
        for metric, value in best_metrics.items():
            if not metric.endswith('_epoch'):
                print(f"  {metric}: {value:.4f}")
    
    return model_dir

def generate_text(model, prompt, word_to_id, id_to_word, max_length=100, temperature=1.0):
    """Generate text using the trained LLM."""
    # Create tokenizer tuple for consistency
    tokenizer = (word_to_id, id_to_word)
    
    # Tokenize the prompt
    prompt_tokens = tokenize_text(prompt, tokenizer)
    
    # Remove padding and get actual prompt length
    prompt_tokens = [t for t in prompt_tokens if t != word_to_id[PAD_TOKEN]]
    
    # Ensure we don't exceed max sequence length
    if len(prompt_tokens) >= MAX_SEQUENCE_LENGTH:
        prompt_tokens = prompt_tokens[-(MAX_SEQUENCE_LENGTH - 1):]
    
    generated = prompt_tokens.copy()
    
    for _ in range(max_length):
        # Prepare input (pad to MAX_SEQUENCE_LENGTH)
        input_seq = generated + [word_to_id[PAD_TOKEN]] * (MAX_SEQUENCE_LENGTH - len(generated))
        input_seq = input_seq[:MAX_SEQUENCE_LENGTH]
        
        # Predict next token
        input_array = np.array([input_seq])
        predictions = model.predict(input_array, verbose=0)[0]
        
        # Get the last non-padded position
        last_pos = min(len(generated) - 1, MAX_SEQUENCE_LENGTH - 1)
        next_token_logits = predictions[last_pos]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
            probabilities = tf.nn.softmax(next_token_logits).numpy()
            next_token = np.random.choice(len(probabilities), p=probabilities)
        else:
            next_token = np.argmax(next_token_logits)
        
        # Stop if we hit end token
        if next_token == word_to_id[END_TOKEN]:
            break
        
        generated.append(next_token)
        
        # Stop if we've reached maximum sequence length
        if len(generated) >= MAX_SEQUENCE_LENGTH:
            break
    
    # Convert back to text
    words = []
    for token_id in generated:
        word = id_to_word.get(token_id, UNK_TOKEN)
        if word in [PAD_TOKEN, START_TOKEN, END_TOKEN]:
            continue
        # strip BPE whitespace marker
        word = word.replace('Ġ', '')
        words.append(word)

    return ' '.join(words)

def train_llm_model(text_data, model=None):
    """Train the LLM transformer model."""
    print(f"Starting LARGE LLM training with {len(text_data)} text samples")
    
    # Initialize wandb with large model configuration
    wandb.init(
        project="llm-transformer-training",
        config={
            "model_architecture": "large_transformer_llm",
            "model_size": "large",
            "num_layers": NUM_LAYERS,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "dff": DFF,
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "min_word_freq": MIN_WORD_FREQ,
            "test_small_model": TEST_SMALL_MODEL,
            "dataset_size": len(text_data)
        }
    )
    
    # Prepare data with larger sample size
    result = prepare_text_data(text_data, max_samples=100_000)
    if result[0] is None:
        print("Failed to prepare training data. Exiting.")
        wandb.finish()
        return None
    
    X_train, X_test, y_train, y_test, word_to_id, id_to_word = result
    
    print(f"Training on {X_train.shape[0]} samples, validating on {X_test.shape[0]} samples")
    print(f"Sequence length: {X_train.shape[1]}, Vocabulary size: {len(word_to_id)}")
    
    # Log data preparation metrics
    wandb.log({
        "data_preparation/train_samples": X_train.shape[0],
        "data_preparation/validation_samples": X_test.shape[0],
        "data_preparation/sequence_length": X_train.shape[1],
        "data_preparation/vocabulary_size": len(word_to_id),
        "data_preparation/memory_usage_train_gb": X_train.nbytes / (1024**3),
        "data_preparation/memory_usage_val_gb": X_test.nbytes / (1024**3),
    })
    
    # Use passed model or create new one
    if model is not None:
        # verify loaded model matches new max length
        expected_dim = MAX_SEQUENCE_LENGTH
        actual_dim = model.input_shape[1]
        if actual_dim != expected_dim:
            print(f"⚠️ Loaded model input dim {actual_dim} != expected {expected_dim}, rebuilding model")
            model = create_llm_transformer_model(len(word_to_id))
            wandb.log({"model/recreated_for_shape_mismatch": True})
        else:
            print("✅ Using provided model with matching input shape")
            wandb.log({"model/reloaded_existing": True})
    else:
        print("🆕 Creating new LARGE LLM model...")
        model = create_llm_transformer_model(len(word_to_id))
        wandb.log({"model/created_new": True})
    
    # Create optimizer with gradient clipping for large model stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,
        epsilon=1e-7,
        beta_1=0.9,
        beta_2=0.95  # Different beta_2 for large models
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Log model parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    model_size_gb = total_params * 4 / (1024**3)  # Assuming float32
    
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/non_trainable_parameters": total_params - trainable_params,
        "model/estimated_size_gb": model_size_gb,
        "model/parameters_millions": total_params / 1e6
    })
    
    print(f"Large model created with {total_params:,} parameters ({total_params/1e6:.1f}M)")
    print(f"Estimated model size: {model_size_gb:.2f} GB")
    
    # Save initial model
    model.save('llm_transformer_model.keras')
    print("Initial large model saved as 'llm_transformer_model.keras'")
    
    # Create TensorBoard log directory
    log_dir = f"logs/llm_transformer_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Custom callback for wandb logging with proper step tracking
    class WandbCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.step_offset = 0
            
        def on_train_begin(self, logs=None):
            # Set the step offset to avoid conflicts with data preparation logs
            self.step_offset = wandb.run.step if wandb.run else 0
            
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                # Log training metrics with proper step calculation
                log_dict = {}
                for key, value in logs.items():
                    log_dict[f"training/{key}"] = value
                
                # Calculate and log perplexity
                if 'loss' in logs:
                    log_dict['training/perplexity'] = np.exp(logs['loss'])
                if 'val_loss' in logs:
                    log_dict['training/val_perplexity'] = np.exp(logs['val_loss'])
                
                # Log current learning rate
                lr = float(self.model.optimizer.learning_rate)
                log_dict['training/learning_rate'] = lr
                
                # Log gradient statistics for large model monitoring
                if hasattr(self.model.optimizer, 'clipnorm'):
                    log_dict['training/gradient_clipnorm'] = float(self.model.optimizer.clipnorm)
                
                # Use proper step calculation
                training_step = self.step_offset + epoch + 1
                wandb.log(log_dict, step=training_step)
    
    # Callbacks with adjustments for large model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=2,  # Less frequent for large models
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,  # Less aggressive reduction
        patience=5,  # More patience for large models
        min_lr=1e-8,
        verbose=1
    )
    
    # Custom callback for perplexity
    class PerplexityCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                if 'loss' in logs:
                    logs['perplexity'] = np.exp(logs['loss'])
                if 'val_loss' in logs:
                    logs['val_perplexity'] = np.exp(logs['val_loss'])
    
    # Train the model with adjusted settings for large model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),  # More patience
            tf.keras.callbacks.ModelCheckpoint(
                'llm_transformer_model.keras',
                save_best_only=True,
                save_weights_only=False
            ),
            lr_scheduler,
            tensorboard_callback,
            PerplexityCallback(),
            WandbCallback()  # Use the fixed callback
        ]
    )
    
    # Log final training metrics
    final_metrics = {}
    if 'loss' in history.history:
        final_metrics['final/best_train_loss'] = min(history.history['loss'])
        final_metrics['final/final_train_loss'] = history.history['loss'][-1]
        final_metrics['final/best_train_perplexity'] = np.exp(min(history.history['loss']))
    
    if 'val_loss' in history.history:
        final_metrics['final/best_val_loss'] = min(history.history['val_loss'])
        final_metrics['final/final_val_loss'] = history.history['val_loss'][-1]
        final_metrics['final/best_val_perplexity'] = np.exp(min(history.history['val_loss']))
    
    if 'sparse_categorical_accuracy' in history.history:
        final_metrics['final/best_train_accuracy'] = max(history.history['sparse_categorical_accuracy'])
        final_metrics['final/final_train_accuracy'] = history.history['sparse_categorical_accuracy'][-1]
    
    if 'val_sparse_categorical_accuracy' in history.history:
        final_metrics['final/best_val_accuracy'] = max(history.history['val_sparse_categorical_accuracy'])
        final_metrics['final/final_val_accuracy'] = history.history['val_sparse_categorical_accuracy'][-1]
    
    final_metrics['final/total_epochs_trained'] = len(history.history.get('loss', []))
    
    wandb.log(final_metrics)
    
    # Save the complete model
    model_dir = save_complete_llm_model(model, word_to_id, id_to_word, history)
    
    # Save for Streamlit interaction
    streamlit_dir = save_model_for_streamlit(model, word_to_id, id_to_word)
    
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view in browser, run: tensorboard --logdir={log_dir}")
    print(f"Complete model saved to: {model_dir}")
    print(f"Streamlit model saved to: {streamlit_dir}")
    
    # Log model save location with proper string handling
    wandb.log({
        "model_save_info/directory": model_dir,
        "model_save_info/tensorboard_logs": log_dir,
        "model_save_info/model_file": "llm_transformer_model.keras"
    })
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 3)
    if 'perplexity' in history.history:
        plt.plot(history.history['perplexity'])
        plt.plot(history.history['val_perplexity'])
        plt.title('Model Perplexity')
        plt.ylabel('Perplexity')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('llm_transformer_training_history.png')
    
    # Log the training plot to wandb
    wandb.log({"training_plots/history": wandb.Image('llm_transformer_training_history.png')})
    
    plt.show()
    
    # Finish wandb run
    wandb.finish()
    
    return model, word_to_id, id_to_word

def load_text_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1", max_samples=50000, split="train"):
    """Load text dataset from Hugging Face or local sources."""
    
    if not DATASETS_AVAILABLE:
        print("Hugging Face datasets not available. Using sample data.")
        return get_sample_texts()
    
    print(f"Loading {dataset_name} dataset from Hugging Face...")
    
    try:
        if dataset_name == "wikitext":
            # WikiText dataset - good for general language modeling
            dataset = load_dataset("wikitext", subset, split=split)
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
            
        elif dataset_name == "openwebtext":
            # OpenWebText - large diverse web text (requires more memory)
            dataset = load_dataset("openwebtext", split=split, streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                if len(item['text'].strip()) > 100:
                    texts.append(item['text'])
            
        elif dataset_name == "bookcorpus":
            # BookCorpus - literature and novels
            dataset = load_dataset("bookcorpus", split=split)
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
            
        elif dataset_name == "imdb":
            # IMDB reviews - sentiment-rich text
            dataset = load_dataset("imdb", split=split)
            texts = [item['text'] for item in dataset]
            
        elif dataset_name == "ag_news":
            # AG News - news articles
            dataset = load_dataset("ag_news", split=split)
            texts = [item['text'] for item in dataset]
            
        elif dataset_name == "wikipedia":
            # Wikipedia articles (English)
            dataset = load_dataset("wikipedia", "20220301.en", split=split, streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                if len(item['text'].strip()) > 200:
                    texts.append(item['text'])
                    
        elif dataset_name == "c4":
            # Common Crawl (C4) - very large web text
            dataset = load_dataset("c4", "en", split=split, streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                if len(item['text'].strip()) > 100:
                    texts.append(item['text'])
                    
        elif dataset_name == "tiny_shakespeare":
            # Tiny Shakespeare - classic literature
            dataset = load_dataset("tiny_shakespeare", split=split)
            # Split into sentences for better training
            full_text = dataset[0]['text']
            sentences = re.split(r'[.!?]+', full_text)
            texts = [sent.strip() for sent in sentences if len(sent.strip()) > 20]
            
        elif dataset_name == "squad":
            # SQuAD - question answering dataset
            dataset = load_dataset("squad", split=split)
            texts = []
            for item in dataset:
                # Combine context and questions for language modeling
                texts.append(item['context'])
                texts.append(item['question'])
                
        elif dataset_name == "cnn_dailymail":
            # CNN DailyMail - news summarization
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                texts.append(item['article'])
                texts.append(item['highlights'])
        
        # CODING DATASETS START HERE - FIXED
        elif dataset_name == "codeparrot":
            # CodeParrot - Python code dataset
            try:
                dataset = load_dataset("codeparrot/codeparrot-clean", split=split, streaming=True, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    if 'content' in item and len(item['content'].strip()) > 100:
                        texts.append(item['content'])
            except Exception as e:
                print(f"Error with codeparrot dataset: {e}")
                # Fallback to a simpler Python code dataset
                texts = get_coding_sample_texts()
        
        elif dataset_name == "github_code":
            # GitHub Code dataset - Fixed data structure handling
            try:
                dataset = load_dataset("codeparrot/github-code", split=split, streaming=True, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    # Handle different possible field names
                    code_content = None
                    if 'code' in item:
                        code_content = item['code']
                    elif 'content' in item:
                        code_content = item['content']
                    elif 'text' in item:
                        code_content = item['text']
                    
                    if code_content and len(str(code_content).strip()) > 50:
                        # Add programming language context if available
                        lang = item.get('language', item.get('programming_language', 'unknown'))
                        code_with_context = f"# Language: {lang}\n{code_content}"
                        texts.append(code_with_context)
            except Exception as e:
                print(f"Error with github_code dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "the_stack":
            # The Stack - large code dataset (requires authentication)
            try:
                dataset = load_dataset("bigcode/the-stack-dedup", data_files="data/python/train-*.parquet", split=split, streaming=True, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    if 'content' in item and len(item['content'].strip()) > 100:
                        texts.append(item['content'])
            except Exception as e:
                print(f"The Stack dataset requires authentication or is unavailable: {e}")
                # Use alternative Python code sources
                texts = get_coding_sample_texts()
        
        elif dataset_name == "code_alpaca":
            # Code Alpaca - instruction-following code dataset
            try:
                dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=split, trust_remote_code=True)
                texts = []
                for item in dataset[:max_samples]:
                    # Handle different field structures
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output = item.get('output', '')
                    
                    if instruction and output:
                        instruction_text = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: {output}"
                        texts.append(instruction_text)
            except Exception as e:
                print(f"Error with code_alpaca dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "python_code_instructions":
            # Python code with instructions - Fixed field handling
            try:
                dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split=split, trust_remote_code=True)
                texts = []
                for item in dataset[:max_samples]:
                    # Handle different possible field names
                    instruction = item.get('instruction', item.get('prompt', ''))
                    output = item.get('output', item.get('completion', item.get('response', '')))
                    
                    if instruction and output:
                        code_text = f"Task: {instruction}\n\nSolution:\n{output}"
                        texts.append(code_text)
            except Exception as e:
                print(f"Error with python_code_instructions dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "code_contests":
            # Code contests dataset
            try:
                dataset = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    # Handle the nested structure
                    description = item.get('description', '')
                    solutions = item.get('solutions', {})
                    
                    if description:
                        problem_text = f"Problem: {description}\n\nSolutions:\n"
                        
                        # Handle different solution formats
                        if isinstance(solutions, dict):
                            solution_list = solutions.get('solution', [])
                        else:
                            solution_list = solutions if isinstance(solutions, list) else []
                        
                        for solution in solution_list[:3]:  # Limit to 3 solutions per problem
                            problem_text += f"\n{solution}\n"
                        
                        if len(problem_text.strip()) > 100:
                            texts.append(problem_text)
            except Exception as e:
                print(f"Error with code_contests dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "code_search_net":
            # CodeSearchNet - code with documentation - Fixed field handling
            try:
                dataset = load_dataset("code_search_net", "python", split=split, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    # Handle different field names
                    func_name = item.get('func_name', item.get('function_name', ''))
                    docstring = item.get('docstring', item.get('documentation', ''))
                    code = item.get('code', item.get('source_code', ''))
                    
                    if func_name and code:
                        code_with_doc = f"Function: {func_name}\nDocstring: {docstring}\nCode:\n{code}"
                        texts.append(code_with_doc)
            except Exception as e:
                print(f"Error with code_search_net dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "apps":
            # APPS dataset - coding problems
            try:
                dataset = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= max_samples:
                        break
                    # Handle the structure
                    problem_id = item.get('problem_id', f'problem_{i}')
                    question = item.get('question', item.get('description', ''))
                    solutions = item.get('solutions', [])
                    
                    if question:
                        problem_text = f"Problem:\n{problem_id}\n{question}\n\nSolutions:\n"
                        for solution in solutions[:2]:  # Limit solutions
                            problem_text += f"{solution}\n"
                        if len(problem_text.strip()) > 100:
                            texts.append(problem_text)
            except Exception as e:
                print(f"Error with apps dataset: {e}")
                texts = get_coding_sample_texts()
        
        elif dataset_name == "conala":
            # CoNaLa - natural language to code - Fixed field handling
            try:
                dataset = load_dataset("neulab/conala", split=split, trust_remote_code=True)
                texts = []
                for item in dataset[:max_samples]:
                    # Handle different field structures
                    intent = item.get('intent', item.get('question', ''))
                    snippet = item.get('snippet', item.get('code', ''))
                    
                    if intent and snippet:
                        nl_to_code = f"Intent: {intent}\nCode: {snippet}"
                        texts.append(nl_to_code)
            except Exception as e:
                print(f"Error with conala dataset: {e}")
                texts = get_coding_sample_texts()
            
        else:
            print(f"Unknown dataset: {dataset_name}. Using sample data.")
            return get_sample_texts()
        
        # Limit to max_samples
        if len(texts) > max_samples:
            texts = texts[:max_samples]
            
        print(f"Loaded {len(texts)} texts from {dataset_name}")
        return texts
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to sample data...")
        if dataset_name in ["codeparrot", "code_alpaca", "python_code_instructions", "code_search_net", 
                           "conala", "github_code", "the_stack", "apps", "code_contests"]:
            return get_coding_sample_texts()
        return get_sample_texts()

def get_coding_sample_texts():
    """Get sample coding texts when coding datasets are not available."""
    return [
        "# Python function to calculate fibonacci numbers\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "# Data structure implementation: Binary Tree\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right",
        "# Web scraping with requests and BeautifulSoup\nimport requests\nfrom bs4 import BeautifulSoup\n\ndef scrape_website(url):\n    response = requests.get(url)\n    soup = BeautifulSoup(response.content, 'html.parser')\n    return soup.get_text()",
        "# Machine learning with scikit-learn\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nmodel = LinearRegression().fit(X_train, y_train)",
        "# Data analysis with pandas\nimport pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('data.csv')\ndf_grouped = df.groupby('category').agg({'value': ['mean', 'sum', 'count']})",
        "# Flask web application\nfrom flask import Flask, render_template, request\n\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return render_template('index.html')\n\nif __name__ == '__main__':\n    app.run(debug=True)",
        "# Algorithm: Quick Sort implementation\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "# Object-oriented programming: Calculator class\nclass Calculator:\n    def __init__(self):\n        self.history = []\n    \n    def add(self, a, b):\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result",
        "# Database operations with SQLAlchemy\nfrom sqlalchemy import create_engine, Column, Integer, String\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker\n\nBase = declarative_base()\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(50))",
        "# API development with FastAPI\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.post('/items/')\nasync def create_item(item: Item):\n    return {'item': item}",
        "# File processing and text analysis\nimport re\nfrom collections import Counter\n\ndef analyze_text(filename):\n    with open(filename, 'r') as file:\n        text = file.read().lower()\n        words = re.findall(r'\\b\\w+\\b', text)\n        return Counter(words)",
        "# Async programming with asyncio\nimport asyncio\nimport aiohttp\n\nasync def fetch_data(session, url):\n    async with session.get(url) as response:\n        return await response.text()\n\nasync def main():\n    async with aiohttp.ClientSession() as session:\n        data = await fetch_data(session, 'https://api.example.com')\n        return data",
        "# Data visualization with matplotlib\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n\nplt.figure(figsize=(10, 6))\nplt.plot(x, y, label='sin(x)')\nplt.xlabel('x')\nplt.ylabel('y')\nplt.legend()\nplt.show()",
        "# Unit testing with pytest\nimport pytest\n\ndef add_numbers(a, b):\n    return a + b\n\ndef test_add_numbers():\n    assert add_numbers(2, 3) == 5\n    assert add_numbers(-1, 1) == 0\n    assert add_numbers(0, 0) == 0",
        "# Regular expressions for text processing\nimport re\n\ndef extract_emails(text):\n    email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'\n    return re.findall(email_pattern, text)"
    ]

def get_sample_texts():
    """Get sample texts when datasets are not available."""
    return [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Deep learning models use neural networks with multiple layers to learn complex patterns in data.",
        "Transformers revolutionized natural language processing by introducing attention mechanisms.",
        "Large language models can generate coherent and contextually relevant text on various topics.",
        "The field of artificial intelligence has grown rapidly in recent years with significant breakthroughs.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Computer vision allows machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning enables agents to learn optimal actions through interaction with environments.",
        # Add more varied sample texts for better training diversity
        "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
        "Renewable energy sources like solar and wind power are becoming increasingly cost-effective.",
        "Space exploration has led to numerous technological innovations that benefit life on Earth.",
        "The internet has fundamentally changed how we communicate, work, and access information.",
        "Biotechnology and genetic engineering hold promise for treating previously incurable diseases.",
        "Quantum computing could revolutionize fields from cryptography to drug discovery.",
        "Virtual and augmented reality technologies are creating new forms of entertainment and education.",
        "The global economy is increasingly interconnected through trade and digital technologies.",
        "Education systems worldwide are adapting to incorporate digital learning tools and methods.",
        "Scientific research relies on peer review and reproducibility to ensure reliable knowledge."
    ]

def preprocess_dataset_text(text):
    """Enhanced preprocessing for dataset texts."""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Remove text within parentheses or brackets that might be metadata
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Basic cleaning
    text = text.strip()
    
    # Skip very short or very long texts
    if len(text) < 20 or len(text) > 10000:
        return ""
    
    return text

def load_text_from_file(file_path, chunk_size=1000, min_chunk_length=50):
    """
    Load text from a .txt file and split it into chunks for training.
    
    Args:
        file_path (str): Path to the .txt file
        chunk_size (int): Approximate number of words per chunk
        min_chunk_length (int): Minimum length of text chunks to keep
        
    Returns:
        list: List of text chunks ready for training
    """
    print(f"Loading text from file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    try:
        # Read the entire file
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"Loaded file with {len(full_text)} characters")
        
        # Basic preprocessing
        # Remove excessive whitespace and normalize
        full_text = re.sub(r'\n+', '\n', full_text)  # Normalize line breaks
        full_text = re.sub(r'\s+', ' ', full_text)    # Normalize spaces
        
        # Split into paragraphs first (by double newlines or similar)
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        texts = []
        current_chunk = ""
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < min_chunk_length:
                continue
                
            words = paragraph.split()
            
            # If adding this paragraph would exceed chunk_size, save current chunk
            if current_word_count + len(words) > chunk_size and current_chunk:
                if len(current_chunk.strip()) >= min_chunk_length:
                    texts.append(current_chunk.strip())
                current_chunk = paragraph
                current_word_count = len(words)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
                current_word_count += len(words)
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk.strip()) >= min_chunk_length:
            texts.append(current_chunk.strip())
        
        print(f"Split into {len(texts)} text chunks")
        print(f"Average chunk length: {sum(len(t) for t in texts) / len(texts):.0f} characters")
        
        return texts
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def load_multiple_text_files(file_paths, chunk_size=1000, min_chunk_length=50):
    """
    Load text from multiple .txt files and combine them.
    
    Args:
        file_paths (list): List of paths to .txt files
        chunk_size (int): Approximate number of words per chunk
        min_chunk_length (int): Minimum length of text chunks to keep
        
    Returns:
        list: Combined list of text chunks from all files
    """
    all_texts = []
    
    for file_path in file_paths:
        texts = load_text_from_file(file_path, chunk_size, min_chunk_length)
        all_texts.extend(texts)
        print(f"Added {len(texts)} chunks from {file_path}")
    
    print(f"Total chunks from all files: {len(all_texts)}")
    return all_texts

def load_text_from_directory(directory_path, file_extension=".txt", chunk_size=1000, min_chunk_length=50):
    """
    Load all text files from a directory.
    
    Args:
        directory_path (str): Path to directory containing text files
        file_extension (str): File extension to look for (default: ".txt")
        chunk_size (int): Approximate number of words per chunk
        min_chunk_length (int): Minimum length of text chunks to keep
        
    Returns:
        list: Combined list of text chunks from all files in directory
    """
    print(f"Loading text files from directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
    
    # Find all text files in directory
    text_files = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(file_extension.lower()):
            text_files.append(os.path.join(directory_path, filename))
    
    print(f"Found {len(text_files)} {file_extension} files")
    
    if not text_files:
        print(f"No {file_extension} files found in {directory_path}")
        return []
    
    # Load all files
    return load_multiple_text_files(text_files, chunk_size, min_chunk_length)

def interactive_conversation(model, word_to_id, id_to_word):
    """Interactive conversation interface with the trained model."""
    print("\n" + "="*60)
    print("🤖 INTERACTIVE CONVERSATION MODE")
    print("="*60)
    print("You can now chat with your trained LLM!")
    print("Commands:")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("  - Type 'temp <number>' to change temperature (e.g., 'temp 0.8')")
    print("  - Type 'length <number>' to change max response length")
    print("  - Type 'reset' to clear conversation history")
    print("-" * 60)
    
    conversation_history = []
    temperature = 0.8
    max_response_length = 100
    
    # Initialize wandb for conversation logging
    wandb.init(
        project="llm-transformer-training",
        job_type="interactive_conversation",
        config={
            "vocab_size": len(word_to_id),
            "model_type": "transformer_llm",
            "initial_temperature": temperature,
            "initial_max_length": max_response_length
        }
    )
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
                
            elif user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"🌡️ Temperature set to {temperature}")
                        wandb.log({"conversation/temperature_changed": temperature})
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except (IndexError, ValueError):
                    print("❌ Usage: temp <number> (e.g., temp 0.8)")
                continue
                
            elif user_input.lower().startswith('length '):
                try:
                    new_length = int(user_input.split()[1])
                    if 10 <= new_length <= 500:
                        max_response_length = new_length
                        print(f"📏 Max response length set to {max_response_length}")
                        wandb.log({"conversation/max_length_changed": max_response_length})
                    else:
                        print("❌ Length must be between 10 and 500")
                except (IndexError, ValueError):
                    print("❌ Usage: length <number> (e.g., length 50)")
                continue
                
            elif user_input.lower() == 'reset':
                conversation_history = []
                print("🔄 Conversation history cleared!")
                wandb.log({"conversation/history_reset": True})
                continue
            
            # Add user input to conversation history
            conversation_history.append(f"Human: {user_input}")
            
            # Create context from recent conversation history (last 3 exchanges)
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            context = " ".join(recent_history) + " AI:"
            
            print("🤖 AI: ", end="", flush=True)
            
            # Generate response with typing effect
            response = generate_text(
                model, 
                context, 
                word_to_id, 
                id_to_word, 
                max_length=max_response_length, 
                temperature=temperature
            )
            
            # Clean up the response (remove context repetition)
            if "AI:" in response:
                response = response.split("AI:")[-1].strip()
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            
            # Simulate typing effect
            import time
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.03)  # Adjust speed as needed
            print()  # New line after response
            
            # Add AI response to conversation history
            conversation_history.append(f"AI: {response}")
            
            # Log conversation to wandb
            conversation_count += 1
            wandb.log({
                "conversation/exchange_count": conversation_count,
                "conversation/user_input_length": len(user_input),
                "conversation/ai_response_length": len(response),
                "conversation/temperature_used": temperature,
                "conversation/max_length_used": max_response_length,
                "conversation/history_length": len(conversation_history)
            })
            
            # Log conversation turn as a table entry every 5 exchanges
            if conversation_count % 5 == 0:
                recent_exchanges = []
                for i in range(max(0, len(conversation_history) - 10), len(conversation_history), 2):
                    if i + 1 < len(conversation_history):
                        recent_exchanges.append([
                            conversation_count - (len(conversation_history) - i) // 2,
                            conversation_history[i].replace("Human: ", ""),
                            conversation_history[i + 1].replace("AI: ", "")
                        ])
                
                if recent_exchanges:
                    wandb.log({
                        f"conversation/recent_exchanges_batch_{conversation_count // 5}": wandb.Table(
                            columns=["turn", "user_input", "ai_response"],
                            data=recent_exchanges
                        )
                    })
            
        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print("🔄 Please try again or type 'quit' to exit.")
    
    # Log final conversation statistics
    wandb.log({
        "conversation/final_exchange_count": conversation_count,
        "conversation/final_history_length": len(conversation_history),
        "conversation/conversation_completed": True
    })
    
    wandb.finish()

def save_model_for_streamlit(model, word_to_id, id_to_word, model_name="streamlit_model"):
    """Save model and tokenizer specifically for Streamlit interaction."""
    
    # Create streamlit model directory
    streamlit_dir = "streamlit_model"
    os.makedirs(streamlit_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(streamlit_dir, "model.keras")
    model.save(model_path)
    
    # Save tokenizer
    with open(os.path.join(streamlit_dir, "word_to_id.pkl"), 'wb') as f:
        pickle.dump(word_to_id, f)
    with open(os.path.join(streamlit_dir, "id_to_word.pkl"), 'wb') as f:
        pickle.dump(id_to_word, f)
    
    # Save model configuration for Streamlit
    config = {
        "vocab_size": len(word_to_id),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "model_parameters": {
            "num_layers": NUM_LAYERS,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "dff": DFF,
            "dropout_rate": DROPOUT_RATE
        },
        "special_tokens": {
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "start_token": START_TOKEN,
            "end_token": END_TOKEN
        },
        "tokenizers_available": TOKENIZERS_AVAILABLE
    }
    
    with open(os.path.join(streamlit_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model saved for Streamlit in: {streamlit_dir}")
    return streamlit_dir

if __name__ == "__main__":
    print("🚀 Starting COMPREHENSIVE LARGE LLM training process")
    print("="*70)
    
    # System status check
    print("\n🔍 SYSTEM STATUS:")
    print(f"   TensorFlow: {tf.__version__}")
    print(f"   GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"   CUDA Support: {tf.test.is_built_with_cuda()}")
    print(f"   Datasets Library: {'✅ Available' if DATASETS_AVAILABLE else '❌ Not installed'}")
    print(f"   Tokenizers Library: {'✅ Available (BPE)' if TOKENIZERS_AVAILABLE else '⚠️  Using word-level fallback'}")
    
    if not TOKENIZERS_AVAILABLE:
        print("\n💡 OPTIMIZATION TIP:")
        print("   Install tokenizers for better performance: pip install tokenizers")
        print("   This enables subword tokenization which is more efficient for large vocabularies")
    
    print("\n" + "="*70)
    
    # Dataset options - updated with better fallbacks for coding datasets
    DATASET_OPTIONS = {
        # General text datasets
        "wikitext": {"subset": "wikitext-2-raw-v1", "description": "Wikipedia articles (small)", "max_samples": 25000},
        "imdb": {"subset": None, "description": "Movie reviews", "max_samples": 30000},
        "ag_news": {"subset": None, "description": "News articles", "max_samples": 30000},
        "tiny_shakespeare": {"subset": None, "description": "Shakespeare texts", "max_samples": 15000},
        "squad": {"subset": None, "description": "Question-answer pairs", "max_samples": 25000},
        
        # Local custom datasets
        "extracted_conversations": {"subset": None, "description": "Extracted conversations dataset", "max_samples": 50000},
        "rag_data": {"subset": None, "description": "RAG data", "max_samples": 50000},
        
        # CODING DATASETS (with better error handling)
        "github_code": {"subset": None, "description": "GitHub code repository", "max_samples": 50000},
        "code_alpaca": {"subset": None, "description": "Code instruction dataset", "max_samples": 50000},
        "python_code_instructions": {"subset": None, "description": "Python coding instructions", "max_samples": 50000},
        "code_search_net": {"subset": "python", "description": "Code with documentation", "max_samples": 50000},
        "conala": {"subset": None, "description": "Natural language to code", "max_samples": 50000},
        
        # Optional coding datasets (may have access issues)
        "codeparrot": {"subset": None, "description": "Python code dataset", "max_samples": 50000},
        "the_stack": {"subset": None, "description": "Large code dataset (Python)", "max_samples": 50000},
        "apps": {"subset": None, "description": "Coding competition problems", "max_samples": 50000},
        "code_contests": {"subset": None, "description": "Programming contests", "max_samples": 50000},
    }
    
    print("🚀 LARGE MULTI-DATASET TRAINING MODE")
    print(f"Will train LARGE model on ALL {len(DATASET_OPTIONS)} datasets:")
    for name, config in DATASET_OPTIONS.items():
        print(f"  📚 {name}: {config['description']} (max {config['max_samples']} samples)")
    
    print(f"\n🏗️  LARGE MODEL CONFIGURATION:")
    print(f"  🔹 Layers: {NUM_LAYERS}")
    print(f"  🔹 Model dimension: {D_MODEL}")
    print(f"  🔹 Attention heads: {NUM_HEADS}")
    print(f"  🔹 Feed-forward dimension: {DFF}")
    print(f"  🔹 Context length: {MAX_SEQUENCE_LENGTH}")
    print(f"  🔹 Vocabulary size: {VOCAB_SIZE}")
    
    # Initialize master wandb run for the entire training process
    wandb.init(
        project="llm-transformer-training",
        config={
            "training_mode": "large_multi_dataset_comprehensive",
            "model_scale": "large",
            "total_datasets": len(DATASET_OPTIONS),
            "datasets": list(DATASET_OPTIONS.keys()),
            "total_max_samples": sum(config["max_samples"] for config in DATASET_OPTIONS.values()),
            "model_config": {
                "num_layers": NUM_LAYERS,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "dff": DFF,
                "dropout_rate": DROPOUT_RATE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "vocab_size": VOCAB_SIZE
            }
        },
        job_type="large_multi_dataset_training"
    )
    
    all_texts = []
    dataset_stats = {}
    
    # Load and combine all datasets
    for dataset_name, dataset_config in DATASET_OPTIONS.items():
        print(f"\n📥 Loading {dataset_name}...")
        
        if dataset_name == "extracted_conversations":
            # Load the extracted_conversations.txt file
            conversation_file_path = "extracted_conversations.txt"
            if os.path.exists(conversation_file_path):
                print(f"Loading conversations from {conversation_file_path}...")
                texts = load_text_from_file(
                    conversation_file_path, 
                    chunk_size=500,  # Smaller chunks for conversation data
                    min_chunk_length=30  # Shorter minimum length for conversations
                )
            else:
                print(f"File {conversation_file_path} not found, skipping...")
                texts = []
        elif dataset_name == "rag_data":
            # Load from local directory for RAG data
            rag_data_file = "rag_data.txt"
            if os.path.exists(rag_data_file):
                print(f"Loading RAG data from file: {rag_data_file}")
                texts = load_text_from_file(
                    rag_data_file,
                    chunk_size=1000,
                    min_chunk_length=30
                )
            else:
                print(f"File {rag_data_file} not found, skipping…")
                texts = []
        elif dataset_name in ["codeparrot", "code_alpaca", "python_code_instructions", "code_search_net", 
                             "conala", "github_code", "the_stack", "apps", "code_contests"]:
            # Load coding datasets from Hugging Face
            if dataset_config["subset"]:
                texts = load_text_dataset(
                    dataset_name=dataset_name,
                    subset=dataset_config["subset"],
                    max_samples=dataset_config["max_samples"]
                )
            else:
                texts = load_text_dataset(
                    dataset_name=dataset_name,
                    max_samples=dataset_config["max_samples"]
                )
        else:
            # Load other datasets from Hugging Face
            if dataset_config["subset"]:
                texts = load_text_dataset(
                    dataset_name=dataset_name,
                    subset=dataset_config["subset"],
                    max_samples=dataset_config["max_samples"]
                )
            else:
                texts = load_text_dataset(
                    dataset_name=dataset_name,
                    max_samples=dataset_config["max_samples"]
                )
        
        # Limit to max_samples if specified
        max_samples = dataset_config["max_samples"]
        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]
        
        dataset_stats[dataset_name] = {
            "raw_count": len(texts),
            "total_chars": sum(len(text) for text in texts),
            "avg_length": np.mean([len(text) for text in texts]) if texts else 0
        }
        
        all_texts.extend(texts)
        print(f"✅ Added {len(texts)} texts from {dataset_name}")
    
    print(f"\n📊 DATASET LOADING SUMMARY:")
    print(f"Total texts loaded: {len(all_texts)}")
    
    # Log dataset loading statistics with proper naming
    for dataset_name, stats in dataset_stats.items():
        wandb.log({
            f"data_loading/{dataset_name}/count": stats["raw_count"],
            f"data_loading/{dataset_name}/total_chars": stats["total_chars"],
            f"data_loading/{dataset_name}/avg_length": stats["avg_length"]
        })
    
    wandb.log({
        "data_loading/summary/total_texts": len(all_texts),
        "data_loading/summary/total_characters": sum(len(text) for text in all_texts),
        "data_loading/summary/average_length": np.mean([len(text) for text in all_texts]) if all_texts else 0
    })
    
    # Enhanced preprocessing for all combined texts
    print("\n🔧 Preprocessing all texts...")
    processed_texts = []
    invalid_texts = 0
    
    for i, text in enumerate(all_texts):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i}/{len(all_texts)} texts")
        
        processed_text = preprocess_dataset_text(text)
        if processed_text:  # Only add non-empty texts
            processed_texts.append(processed_text)
        else:
            invalid_texts += 1
    
    print(f"✅ After preprocessing: {len(processed_texts)} valid texts")
    
    # Log preprocessing results with cleaner naming
    wandb.log({
        "data_preprocessing/valid_texts": len(processed_texts),
        "data_preprocessing/invalid_texts": invalid_texts,
        "data_preprocessing/retention_rate": len(processed_texts) / len(all_texts) if all_texts else 0,
        "data_preprocessing/avg_processed_length": np.mean([len(text) for text in processed_texts]) if processed_texts else 0
    })
    
    if len(processed_texts) == 0:
        print("❌ No valid texts after preprocessing. Using sample data.")
        processed_texts = get_sample_texts()
        wandb.log({"data_preprocessing/fallback_to_samples": True})
    
    wandb.finish()  # Finish the data loading run
    
    # Load existing model if available
    model = None
    if os.path.exists('llm_transformer_model.keras'):
        try:
            print("🔄 Loading existing LLM model...")
            model = tf.keras.models.load_model('llm_transformer_model.keras')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🆕 Will create new model instead.")
            model = None
    else:
        print("🆕 No existing model found, will create new one.")
    
    # Train the model on all combined datasets
    print(f"\n🎯 Starting training on {len(processed_texts)} combined text samples...")
    result = train_llm_model(processed_texts, model=model)
    
    if result is not None:
        model, word_to_id, id_to_word = result
        
        # Test text generation with samples from different datasets
        print("\n🧪 Testing text generation with diverse prompts...")
        
        # Initialize wandb for text generation testing
        wandb.init(
            project="llm-transformer-training",
            config={
                "trained_on_datasets": list(DATASET_OPTIONS.keys()),
                "vocab_size": len(word_to_id),
                "model_type": "transformer_llm_multi_dataset",
                "test_temperature": 0.8,
                "test_max_length": 50
            },
            job_type="final_text_generation_test"
        )
        
        # Comprehensive test prompts covering all training datasets including coding
        test_prompts = [
            # Wikipedia/General knowledge
            "The future of artificial intelligence",
            "Climate change is",
            "In the field of science",
            
            # Movie reviews (IMDB style)
            "This movie was",
            "The acting in this film",
            "I really enjoyed",
            
            # News (AG News style)
            "Breaking news:",
            "According to recent reports",
            "The government announced",
            
            # Literature (Shakespeare style)
            "To be or not to be",
            "In fair Verona where",
            "Love is",
            
            # Q&A (SQuAD style)
            "The answer to this question is",
            "When we consider the facts",
            "It is important to understand",
            
            # Conversation (extracted conversations style)
            "Hello, how are you",
            "What do you think about",
            "I was wondering if",
            "Can you help me with",
            "That's interesting because",
            
            # CODING PROMPTS
            "def fibonacci(n):",
            "# Write a function to",
            "import pandas as pd",
            "class Calculator:",
            "# This function calculates",
            "for i in range(",
            "if __name__ == '__main__':",
            "try:",
            "# Sort a list of",
            "# Create a web scraper"
        ]
        
        generation_results = []
        print("\n🎭 Generated text samples:")
        print("-" * 60)
        
        for i, prompt in enumerate(test_prompts):
            generated = generate_text(model, prompt, word_to_id, id_to_word, max_length=50, temperature=0.8)
            print(f"💭 Prompt: {prompt}")
            print(f"🤖 Generated: {generated}\n")
            
            # Log to wandb
            generation_results.append({
                "prompt": prompt,
                "generated_text": generated,
                "prompt_index": i
            })
        
        # Log all generation results
        wandb.log({
            "text_generation/examples": wandb.Table(
                columns=["prompt_category", "prompt", "generated_text", "prompt_index"],
                data=[
                    [
                        "Wikipedia" if i < 3 else
                        "Movie_Reviews" if i < 6 else  
                        "News" if i < 9 else
                        "Literature" if i < 12 else
                        "QA" if i < 15 else
                        "Conversation" if i < 25 else
                        "Coding",
                        r["prompt"], 
                        r["generated_text"], 
                        r["prompt_index"]
                    ] for i, r in enumerate(generation_results)
                ]
            )
        })
        
        # Log generation statistics
        wandb.log({
            "text_generation/total_prompts_tested": len(generation_results),
            "text_generation/avg_response_length": np.mean([len(r["generated_text"]) for r in generation_results]),
            "text_generation/categories_tested": 7  # Updated to include coding
        })
        
        wandb.finish()
        
        # Start interactive conversation mode
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Model trained on {len(processed_texts)} texts from {len(DATASET_OPTIONS)} datasets (including coding datasets)")
        print(f"📖 Vocabulary size: {len(word_to_id):,} tokens")
        print(f"🧠 Model parameters: {model.count_params():,}")
        print("🖥️ Model includes training on: Python code, algorithms, documentation, and programming concepts")
        
        # Ask user if they want to chat
        print("\n🗣️ Would you like to have a conversation with your trained model?")
        user_choice = input("Enter 'yes' or 'y' to start chatting, anything else to exit: ").strip().lower()
        
        if user_choice in ['yes', 'y']:
            interactive_conversation(model, word_to_id, id_to_word)
        else:
            print("👋 Thanks for training! You can run the script again to chat with your model.")
        
    else:
        print("❌ Training failed!")
        print("Please check the error messages above and try again.")
