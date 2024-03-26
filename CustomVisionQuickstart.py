from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import requests
import os
import pandas as pd
from pathlib import Path
import time

start_time = time.time()

# name prediction key and endpoint
prediction_key = "5685b89e848146f69d1a9c465db57528"
prediction_ENDPOINT = "https://uksouth.api.cognitive.microsoft.com/"

# create prediction client
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_ENDPOINT, prediction_credentials)

# connect to Azure
base_url = predictor.config.base_url.format(Endpoint=prediction_ENDPOINT[:-1])
prediction_url = "https://uksouth.api.cognitive.microsoft.com/customvision/v3.0/Prediction/3053440a-dd5f-4474-a359-d7eabc55fab0/detect/iterations/Iteration5/image"

# Set the threshold probability for predictions
threshold_probability = 0.5

# Process each image in the folder
image_folder = r"C:\Users\sinea\Documents\Dissertation\tobacco_data_zhugy\Tobacco800_SinglePage\test"
images = os.listdir(image_folder)

# Create an empty list to store the prediction results
results = []

# make loop that sends each image in folder to Azure and returns prbability
for image_name in images:
    # Open the image file
    with open(os.path.join(image_folder, image_name), "rb") as image_file:
        # Send the image for prediction
        headers = {"Prediction-Key": prediction_key,
                   "Content-Type": "application/octet-stream"}
        response = requests.post(
            prediction_url, headers=headers, data=image_file)

        # Process the prediction response
        predictions = response.json()["predictions"]

        # Access and use the prediction list and if its probability is above the threshold set earlier, append result to list

        for prediction in predictions:
            if prediction["probability"] >= threshold_probability:
                tag_name = prediction["tagName"]
                probability = prediction["probability"]
                bbox = prediction["boundingBox"]
                left = bbox["left"]
                top = bbox["top"]
                width = bbox["width"]
                height = bbox["height"]

                results.append({"ImageName": image_name, "TagName": tag_name,
                                    "Probability": probability, "Left": left,
                                    "Top": top, "Width": width, "Height": height})


# Convert results list to a DataFrame and save as a csv file
df = pd.DataFrame(results)

filepath = Path('C:/Users/sinea/Documents/Dissertation/azure_predictions_test10.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath)

print("Process finished --- %s seconds ---" % (time.time() - start_time))
