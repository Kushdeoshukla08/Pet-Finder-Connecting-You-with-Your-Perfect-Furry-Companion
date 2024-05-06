from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(_name_)

# Load the models
cat_dog_model = load_model('dog_cat_model.h5')
cat_breed_model = load_model('cat_classification.h5')
dog_breed_model = load_model('dog_classification.h5')

# Helper function to process image
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize the image as per the input size of your model
    img = np.expand_dims(img, axis=0)
    return img

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded image
            image_path = os.path.join('static', file.filename)
            file.save(image_path)
            
            # Process the image
            processed_image = process_image(image_path)
            
            # Classify if it's a cat or a dog
            prediction = cat_dog_model.predict(processed_image)
            if prediction[0][0] > 0.5:  # Assuming cat: 0, dog: 1
                animal_type = 'Dog'
                # Call dog breed classification model
                breed_prediction = dog_breed_model.predict(processed_image)
                breed = 'Dog Breed: ' + str(breed_prediction.argmax())
            else:
                animal_type = 'Cat'
                # Call cat breed classification model
                breed_prediction = cat_breed_model.predict(processed_image)
                breed = 'Cat Breed: ' + str(breed_prediction.argmax())
            
            # Delete the uploaded image after classification
            os.remove(image_path)
            
            # Render result template
            return render_template('result.html', result=f'{animal_type}, {breed}')
        else:
            return 'No file uploaded'

if _name_ == '_main_':
    app.run(debug=True)