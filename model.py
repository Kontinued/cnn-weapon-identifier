from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(num_classes, input_shape=(336,336,3)):
    model = Sequential([
        InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape),
        # AttentionLayer(),  # Optional attention
        GlobalAveragePooling2D(),
        Dense(2048, activation='relu'),
        BatchNormalization(),
        Dropout(0.7),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
