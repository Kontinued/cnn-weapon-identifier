from data_preprocessing import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

train_dir = "/content/drive/MyDrive/training_images"
test_dir = "/content/drive/MyDrive/test"

train_gen, val_gen, _ = get_data_generators(train_dir, test_dir)
num_classes = train_gen.num_classes

model = build_model(num_classes)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-5)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[reduce_lr]
)
model.save("weapon_classifier.h5")
