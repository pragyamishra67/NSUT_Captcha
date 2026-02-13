import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import keras_tuner as kt
import streamlit as st
import pandas as pd
import numpy as np

# --- CALLBACKS FOR VISUALIZATION ---

class StreamlitPlotCallback(Callback):
    """Callback for Manual Training Mode"""
    def __init__(self, plot_container):
        super().__init__()
        self.plot_container = plot_container
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('accuracy'))
        self.val_accs.append(logs.get('val_accuracy'))
        
        with self.plot_container:
            st.markdown("### 📈 Training Progress")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart({
                    "Train Loss": self.losses, 
                    "Val Loss": self.val_losses
                }, height=200, color=["#1A73E8", "#EA4335"])
            with col2:
                st.line_chart({
                    "Train Accuracy": self.accs, 
                    "Val Accuracy": self.val_accs
                }, height=200, color=["#1A73E8", "#34A853"])

class TunerUpdateCallback(Callback):
    """Callback for Bayesian Auto-Tuning Mode"""
    def __init__(self, status_box, chart_box, trial_id):
        self.status_box = status_box
        self.chart_box = chart_box
        self.trial_id = trial_id
        self.epoch_acc = []
        
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('val_accuracy')
        self.epoch_acc.append(acc)
        
        # Update Status Text
        self.status_box.markdown(f"""
        **Running Trial: {self.trial_id}** | Epoch: {epoch + 1}  
        Current Accuracy: `{acc:.4f}`
        """)
        
        # Update Mini-Chart for this trial
        self.chart_box.line_chart(self.epoch_acc, height=100)

# --- CUSTOM TUNER FOR STREAMLIT ---

class StreamlitTuner(kt.BayesianOptimization):
    def __init__(self, st_status_container, st_metrics_container, **kwargs):
        super().__init__(**kwargs)
        self.st_status = st_status_container
        self.st_metrics = st_metrics_container
        self.trial_results = []
        
    def run_trial(self, trial, *args, **kwargs):
        # 1. Setup UI elements for this specific trial
        with self.st_status:
            st.markdown(f"#### 🔄 Optimizing... (Trial {trial.trial_id})")
            status_box = st.empty()
            chart_box = st.empty()
            
        # 2. Inject our custom callback
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(TunerUpdateCallback(status_box, chart_box, trial.trial_id))
        kwargs['callbacks'] = callbacks
        
        # 3. Run the standard Keras Tuner trial logic AND RETURN IT (Fix is here)
        return super().run_trial(trial, *args, **kwargs)
        
    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        # Collect results
        score = trial.score
        hps = trial.hyperparameters.values
        
        # Flatten dictionary for dataframe
        result_entry = {"Trial ID": trial.trial_id, "Val Accuracy": score}
        result_entry.update(hps)
        self.trial_results.append(result_entry)
        
        # Update the Leaderboard DataFrame in real-time
        df = pd.DataFrame(self.trial_results)
        df = df.sort_values(by="Val Accuracy", ascending=False)
        
        with self.st_metrics:
            st.markdown("### 🏆 Optimization Leaderboard")
            st.dataframe(
                df.style.highlight_max(axis=0, subset=["Val Accuracy"], color="#E6F4EA"),
                use_container_width=True
            )

# --- MODEL BUILDERS ---

def build_manual_model(filters1, filters2, dense_units, dropout_rate, lr):
    model = Sequential([
        Conv2D(filters1, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tuner_model(hp):
    model = Sequential()
    
    # Tunable First Block
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=96, step=32),
        kernel_size=3, activation='relu', input_shape=(32, 32, 1)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    # Tunable Second Block
    model.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=3, activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    
    # Tunable Dense
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model