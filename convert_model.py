import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("bone_fracture_model.h5")

# Export the model in SavedModel format
model.export("saved_model_bonefracture")
