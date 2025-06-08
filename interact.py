import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import os
import re
from collections import Counter
import time
import datetime

# Try to import tokenizers
try:
    from tokenizers import ByteLevelBPETokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

# Configure TensorFlow for Streamlit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.warning(f"GPU configuration error: {e}")

class ModelLoader:
    """Class to handle model loading and text generation."""
    
    def __init__(self, model_dir="llm_transformer_complete_20250607_224747"):
        self.model_dir = model_dir
        self.model = None
        self.word_to_id = None
        self.id_to_word = None
        self.config = None
        
    def load_model(self):
        """Load the trained model and tokenizer."""
        if not os.path.exists(self.model_dir):
            return False, "Model directory not found. Please train a model first."
        
        try:
            # Load configuration - try config.json first, then metadata.json
            config_path = os.path.join(self.model_dir, "config.json")
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            elif os.path.exists(metadata_path):
                # Load from metadata.json and convert to config format
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Convert metadata to config format expected by Streamlit
                self.config = {
                    "vocab_size": metadata.get("vocab_size", 50000),
                    "max_sequence_length": metadata.get("max_sequence_length", 1024),
                    "model_parameters": metadata.get("model_parameters", {
                        "num_layers": 4,
                        "d_model": 512,
                        "num_heads": 6,
                        "dff": 1024,
                        "dropout_rate": 0.1
                    }),
                    "special_tokens": metadata.get("special_tokens", {
                        "pad_token": "<PAD>",
                        "unk_token": "<UNK>",
                        "start_token": "<START>",
                        "end_token": "<END>"
                    }),
                    "tokenizers_available": False
                }
            else:
                return False, "Configuration file (config.json or metadata.json) not found."
            
            # Load model
            model_path = os.path.join(self.model_dir, "model.keras")
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            else:
                return False, "Model file not found."
            
            # Load tokenizer
            word_to_id_path = os.path.join(self.model_dir, "word_to_id.pkl")
            id_to_word_path = os.path.join(self.model_dir, "id_to_word.pkl")
            
            if os.path.exists(word_to_id_path) and os.path.exists(id_to_word_path):
                with open(word_to_id_path, 'rb') as f:
                    self.word_to_id = pickle.load(f)
                with open(id_to_word_path, 'rb') as f:
                    self.id_to_word = pickle.load(f)
                
                # Update vocab_size in config to match actual tokenizer
                self.config["vocab_size"] = len(self.word_to_id)
            else:
                return False, "Tokenizer files not found."
            
            return True, "Model loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.!?,:;]', '', text)
        return text.strip()
    
    def tokenize_text(self, text, max_length=None):
        """Convert text to token IDs."""
        if max_length is None:
            max_length = self.config["max_sequence_length"]
        
        words = self.preprocess_text(text).split()
        start_token = self.config["special_tokens"]["start_token"]
        end_token = self.config["special_tokens"]["end_token"]
        unk_token = self.config["special_tokens"]["unk_token"]
        pad_token = self.config["special_tokens"]["pad_token"]
        
        ids = [self.word_to_id.get(start_token)]
        for w in words[:max_length-2]:
            ids.append(self.word_to_id.get(w, self.word_to_id[unk_token]))
        ids.append(self.word_to_id.get(end_token))
        
        # Pad
        pad_id = self.word_to_id[pad_token]
        return ids + [pad_id] * (max_length - len(ids))
    
    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """Generate text using the trained model."""
        if not self.model or not self.word_to_id:
            return "Model not loaded!"
        
        try:
            # Tokenize prompt
            prompt_tokens = self.tokenize_text(prompt)
            
            # Remove padding and get actual prompt length
            pad_token = self.config["special_tokens"]["pad_token"]
            prompt_tokens = [t for t in prompt_tokens if t != self.word_to_id[pad_token]]
            
            # Ensure we don't exceed max sequence length
            max_seq_len = self.config["max_sequence_length"]
            if len(prompt_tokens) >= max_seq_len:
                prompt_tokens = prompt_tokens[-(max_seq_len - 1):]
            
            generated = prompt_tokens.copy()
            
            for _ in range(max_length):
                # Prepare input
                input_seq = generated + [self.word_to_id[pad_token]] * (max_seq_len - len(generated))
                input_seq = input_seq[:max_seq_len]
                
                # Predict next token
                input_array = np.array([input_seq])
                predictions = self.model.predict(input_array, verbose=0)[0]
                
                # Get the last non-padded position
                last_pos = min(len(generated) - 1, max_seq_len - 1)
                next_token_logits = predictions[last_pos]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probabilities = tf.nn.softmax(next_token_logits).numpy()
                    next_token = np.random.choice(len(probabilities), p=probabilities)
                else:
                    next_token = np.argmax(next_token_logits)
                
                # Stop if we hit end token
                end_token = self.config["special_tokens"]["end_token"]
                if next_token == self.word_to_id[end_token]:
                    break
                
                generated.append(next_token)
                
                # Stop if we've reached maximum sequence length
                if len(generated) >= max_seq_len:
                    break
            
            # Convert back to text
            words = []
            special_tokens = [
                self.config["special_tokens"]["pad_token"],
                self.config["special_tokens"]["start_token"],
                self.config["special_tokens"]["end_token"]
            ]
            
            for token_id in generated:
                word = self.id_to_word.get(token_id, self.config["special_tokens"]["unk_token"])
                if word not in special_tokens:
                    # Strip BPE whitespace marker if present
                    word = word.replace('Ġ', '')
                    words.append(word)
            
            return ' '.join(words)
            
        except Exception as e:
            return f"Error generating text: {str(e)}"

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="🤖 LLM Chat Interface",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🤖 Large Language Model Chat Interface")
    st.markdown("### Chat with your trained transformer model!")
    
    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'loading_status' not in st.session_state:
        st.session_state.loading_status = None
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🔧 Model Controls")
        
        # Model loading section
        st.subheader("📥 Load Model")
        
        model_dir = st.text_input(
            "Model Directory", 
            value="llm_transformer_complete_20250607_224747",
            help="Directory containing the trained model files"
        )
        
        if st.button("🔄 Load Model", type="primary"):
            st.session_state.model_loader.model_dir = model_dir
            
            with st.spinner("Loading model..."):
                success, message = st.session_state.model_loader.load_model()
                st.session_state.model_loaded = success
                st.session_state.loading_status = message
        
        # Display loading status
        if st.session_state.loading_status:
            if st.session_state.model_loaded:
                st.success(st.session_state.loading_status)
                
                # Display model info
                if st.session_state.model_loader.config:
                    config = st.session_state.model_loader.config
                    st.info(f"""
                    **Model Info:**
                    - Vocabulary Size: {config['vocab_size']:,}
                    - Layers: {config['model_parameters']['num_layers']}
                    - Model Dimension: {config['model_parameters']['d_model']}
                    - Attention Heads: {config['model_parameters']['num_heads']}
                    """)
            else:
                st.error(st.session_state.loading_status)
        
        st.divider()
        
        # Generation parameters
        st.subheader("⚙️ Generation Settings")
        
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values make output more random, lower values more focused"
        )
        
        max_length = st.slider(
            "📏 Max Response Length",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of tokens to generate"
        )
        
        st.divider()
        
        # Conversation controls
        st.subheader("💬 Conversation")
        
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            if st.session_state.conversation_history:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.txt"
                
                with open(filename, 'w') as f:
                    for entry in st.session_state.conversation_history:
                        f.write(f"{entry['role']}: {entry['content']}\n\n")
                
                st.success(f"Conversation saved as {filename}")
            else:
                st.warning("No conversation to save!")
    
    # Main chat interface
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar.")
        st.info("""
        **Instructions:**
        1. Make sure you have trained a model using the main training script
        2. Enter the model directory path in the sidebar (default: 'streamlit_model')
        3. Click 'Load Model' to load your trained model
        4. Start chatting!
        """)
        return
    
    # Chat container
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        for entry in st.session_state.conversation_history:
            if entry['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.write(entry['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(entry['content'])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Create context from recent conversation
                recent_history = st.session_state.conversation_history[-6:]  # Last 3 exchanges
                context_parts = []
                for entry in recent_history[:-1]:  # Exclude the current prompt
                    if entry['role'] == 'user':
                        context_parts.append(f"Human: {entry['content']}")
                    else:
                        context_parts.append(f"AI: {entry['content']}")
                
                context = " ".join(context_parts) + f" Human: {prompt} AI:"
                
                # Generate response
                response = st.session_state.model_loader.generate_text(
                    context, 
                    max_length=max_length, 
                    temperature=temperature
                )
                
                # Clean up response
                if "AI:" in response:
                    response = response.split("AI:")[-1].strip()
                if "Human:" in response:
                    response = response.split("Human:")[0].strip()
                
                # Display response with typing effect
                response_placeholder = st.empty()
                displayed_response = ""
                
                for char in response:
                    displayed_response += char
                    response_placeholder.write(displayed_response)
                    time.sleep(0.02)  # Typing effect
                
                # Add assistant response to history
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': response
                })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Streamlit | "
        "🔧 Powered by TensorFlow | "
        f"🤖 Model Status: {'✅ Loaded' if st.session_state.model_loaded else '❌ Not Loaded'}"
    )

if __name__ == "__main__":
    # Check if running with streamlit
    try:
        # This will fail if not running with streamlit
        import streamlit.runtime.scriptrunner as sr
        if sr.get_script_run_ctx() is None:
            raise RuntimeError("Not running with streamlit")
        main()
    except (ImportError, RuntimeError):
        print("❌ This script must be run with Streamlit!")
        print("🚀 To run the app, use the following command:")
        print("   streamlit run interact.py")
        print()
        print("📝 Make sure you have streamlit installed:")
        print("   pip install streamlit")
        print()
        print("🔧 If you want to run it in the browser, use:")
        print("   streamlit run interact.py --server.port 8501")
