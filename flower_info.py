import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load the pre-trained model using pickle
MODEL_PATH = 'flower_predictor.pkl'  # Adjust this to your actual model file path
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Flower information dictionary with more details
flower_info = {
    "Rose": {
        "Scientific Name": "Rosa",
        "Species": "Rosaceae",
        "Commonly Found": "Worldwide, especially in temperate regions.",
        "History": "Roses have been revered for thousands of years, symbolizing love and beauty. They were cultivated in ancient civilizations, notably by the Greeks and Romans.",
        "Harvesting Tips": "Cut stems at an angle in the early morning for the best longevity. Remove thorns carefully to avoid injury.",
        "Care Info": "Roses thrive in full sun with at least 6 hours of sunlight. They require regular watering, particularly during dry spells, and should be planted in well-drained soil enriched with organic matter."
    },
    "Daisy": {
        "Scientific Name": "Bellis perennis",
        "Species": "Asteraceae",
        "Commonly Found": "Found across Europe, North America, and temperate regions worldwide.",
        "History": "Daisies have long been associated with innocence and purity, often featured in folklore and traditional medicine as a remedy for various ailments.",
        "Harvesting Tips": "Cut flowers at ground level when they are fully open to maximize bloom life.",
        "Care Info": "Daisies prefer full sun and well-drained soil. They are drought-tolerant once established but benefit from regular watering during dry periods."
    },
    "Dandelion": {
        "Scientific Name": "Taraxacum",
        "Species": "Asteraceae",
        "Commonly Found": "Native to Eurasia but found worldwide.",
        "History": "The dandelion has been used in traditional medicine for centuries, symbolizing resilience and survival. Its leaves and roots have culinary uses as well.",
        "Harvesting Tips": "For the best taste, harvest young leaves before the plant flowers. Roots can be harvested in spring or fall.",
        "Care Info": "Dandelions thrive in a variety of soils and require full sun. They are hardy and adaptable but can spread quickly if not managed."
    },
    "Sunflower": {
        "Scientific Name": "Helianthus annuus",
        "Species": "Asteraceae",
        "Commonly Found": "Primarily found in North America but also cultivated in Europe and Asia.",
        "History": "Sunflowers were cultivated by Native Americans for their seeds and oil. They symbolize adoration and loyalty, often associated with the sun.",
        "Harvesting Tips": "Harvest sunflowers when the seeds are fully developed, usually in late summer or early fall. Hang them upside down to dry.",
        "Care Info": "Sunflowers prefer full sun and well-drained soil. Regular watering is important, especially during dry spells."
    },
    "Tulip": {
        "Scientific Name": "Tulipa",
        "Species": "Liliaceae",
        "Commonly Found": "Originally from Central Asia, widely cultivated in Europe, particularly the Netherlands.",
        "History": "Tulips became famous during the Dutch Golden Age and are symbols of elegance and beauty. They sparked one of the first economic bubbles in history.",
        "Harvesting Tips": "Cut tulips in the early morning when the buds are closed to extend their vase life. Place them in water immediately.",
        "Care Info": "Tulips prefer well-drained soil and full sun. They should not be overwatered, as this can lead to bulb rot."
    }
}

# Class names based on your model
class_names = list(flower_info.keys())

# Streamlit interface
st.title("Flower Classification App")
st.write("Upload an image of a flower, and the model will classify it.")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    image = image.resize((128, 128))  # Resize to your model's input size
    image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Perform the prediction using the loaded model
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    label = class_names[predicted_class_index[0]]

    # Get detailed information about the flower from the dictionary
    flower_details = flower_info[label]

    # Display the result and additional flower information
    st.write(f"Predicted Flower: **{label}**")
    st.write(f"Scientific Name: {flower_details['Scientific Name']}")
    st.write(f"Species: {flower_details['Species']}")
    st.write(f"Commonly Found: {flower_details['Commonly Found']}")
    st.write(f"History: {flower_details['History']}")
    st.write(f"Harvesting Tips: {flower_details['Harvesting Tips']}")
    st.write(f"Care Info: {flower_details['Care Info']}")
