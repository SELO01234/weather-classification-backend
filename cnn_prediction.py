import os
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import utils as image_utils

# Desired size of the images
size = 100

# Batch size for processing images
batch_size = 1000

# Root directory of images
image_root = "./input_folder"

#dictionary to hold names and results of images
result_dictionary = {}

def images_to_matrix(image_root, size, batch_size=8000):
    """Converts images in the test directory to matrices.

    Args:
        image_root (str): The root directory of the images (test directory).
        size (int): The desired size for the images.
        batch_size (int): The number of images to process in each batch.

    Returns:
        dict: A dictionary containing arrays of data and filenames.
    """
    all_data = []
    filenames = []

    print(f"Checking directory: {image_root}")
    if not os.path.isdir(image_root):
        print(f"'{image_root}' is not a directory.")
        return {'data': np.array([]), 'filenames': np.array([])}

    filenames_list = os.listdir(image_root)

    images_processed = 0

    for filename in filenames_list:
        if images_processed >= 15000:
            break

        image_path = os.path.join(image_root, filename)
        if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                img = image_utils.load_img(image_path, target_size=(size, size))
                img = img.convert('RGB')
                img = image_utils.img_to_array(img)
                if img.shape == (size, size, 3):
                    all_data.append(img)
                    filenames.append(filename)
                    images_processed += 1
                else:
                    print(f"Skipping {image_path}, incorrect shape: {img.shape}")
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
        else:
            print(f"'{image_path}' is not a valid image file.")

    return {'data': np.array(all_data), 'filenames': np.array(filenames)}

def save_images_by_prediction(test_data, filenames, predictions, class_labels, output_dir="output_images"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in class_labels:
        class_dir = os.path.join(output_dir, label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for i, prediction in enumerate(predictions):
        predicted_label_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_label_index]

        img_array = test_data[i]
        img = Image.fromarray(np.uint8(img_array)).convert('RGB')
        img.save(os.path.join(output_dir, predicted_label, filenames[i]))

        #saving images to result dictionary
        result_dictionary.update({filenames[i]: predicted_label})

def zip_directory(folder_path, zip_path):

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))


def clearFolders(folder_path = './output_images', zip_path = './classified_images.zip'):
    """
    Deletes output_images folder and zip file corresponding to that folder in order to
    create new output for new request
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    if os.path.exists(zip_path):
        os.remove(zip_path)

def predictImages(image_root, size, batch_size, model_path):

    clearFolders()
    print("Cleared folders")

    result = images_to_matrix(image_root, size, batch_size)

    test_data = result['data']
    filenames = result['filenames']

    inference_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

    # Create a new model using the loaded TFSMLayer
    input_shape = (100, 100, 3)  # Adjust the input shape based on your data
    inputs = tf.keras.Input(shape=input_shape)
    outputs = inference_layer(inputs)
    imageModel = tf.keras.Model(inputs, outputs)

    class_labels = ["Sunny", "Rainy", "Snowy", "Foggy"]

    predictions = imageModel.predict(test_data)

    predictions = predictions['dense_1']  # Accessing the predictions

    save_images_by_prediction(test_data, filenames, predictions, class_labels)

    zip_directory("output_images", "classified_images.zip")
    print(" zipped ")

    for i, prediction in enumerate(predictions):
        predicted_label_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_label_index]
        predicted_probability = float(prediction[predicted_label_index])

        probabilities = {class_labels[j]: float(prediction[j]) for j in range(len(class_labels))}

        print(f"Image {filenames[i]}:")
        print(f"  Predicted Label: {predicted_label} (Probability: {predicted_probability:.2f})")
        print("  Probabilities:")
        for label, prob in probabilities.items():
            print(f"    {label}: {prob:.2f}")
        print()

#Function that is accessed by flask
def predict_images():
    model_path = './cnn_with_dropoutt'
    result_dictionary.clear()
    predictImages(image_root, size, batch_size, model_path)
    print(result_dictionary)
    return result_dictionary
