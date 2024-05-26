import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import time
import json

from pymongo import MongoClient

from dotenv import load_dotenv
load_dotenv()

dbclient = MongoClient()
db = dbclient.dipl

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client for synchronous operations
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)
# [END client]


# [START caption]

folder_dir = os.environ["NAPS_FOLDER"]

# Load image to analyze into a 'bytes' object
for jpg in os.listdir(folder_dir):


    with open(os.path.join(folder_dir, jpg), "rb") as f:
        image_data = f.read()

    # Get a caption for the image. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.OBJECTS, VisualFeatures.TAGS],
        gender_neutral_caption=True,  # Optional (default is False)
        )

    mongo_dict = {
        "title": jpg
    }

    if result.caption is not None:
        mongo_dict["caption"] = result.caption.text


    if result.objects is not None:
        obj_list = []
        for obj in result.objects["values"]:
            # print(obj)
            obj_dict = {
                "name" : obj.tags[0].name,
                "bounding_box": obj["boundingBox"],
                "confidence": obj.tags[0].confidence
            }
            obj_list.append(obj_dict)

    mongo_dict["objects"] = obj_list

    if result.tags is not None:
        tag_list = []
        for t in result.tags["values"]:
            tag_dict = {
                "name": t.name,
                "confidence": t.confidence
            }
            tag_list.append(tag_dict)

    mongo_dict["tags"] = tag_list
    

    result = db.tags.insert_one(mongo_dict)
    print('Inserted post id %s with name %s ' % (result.inserted_id, jpg))

    time.sleep(10)
# [END caption]

