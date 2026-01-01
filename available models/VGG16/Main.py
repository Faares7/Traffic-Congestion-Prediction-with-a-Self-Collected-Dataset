# main.py
from Model import create_vgg16_model

# Define the model
model = create_vgg16_model()
model.save('vgg16_model.h5')  # Save the trained model as 'vgg16_model.h5'

# Now you can train the model using the training data
# model.fit(X_train, y_train, epochs=10, batch_size=32)
