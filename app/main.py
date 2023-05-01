from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

import torch

from util import load_model, pre_process, post_process 



model = load_model()
app = FastAPI()

@app.post("/upload")
async def create_upload_files(files: UploadFile = File()):
    """ Create API endpoint to send image to and specify
     what type of file it'll take

    :param files: Get image files, defaults to File(...)
    :type files: List[UploadFile], optional
    :return: A list of png images
    :rtype: list(bytes)
    """
    # Return preprocessed input batch and loaded image
    image, orig_image = pre_process(files)

    # Run the model and postpocess the output
    with torch.inference_mode():
        prediction = model(image)
    prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

    # # Post process and stitch together the two images to return them
    boxes, pred_classes = post_process(orig_image, prediction)
    result = dict({"data": {"boxes": boxes, "classes": pred_classes}})
    print(result)
    return result


@app.get("/")
async def main():
 
    return {"message" : "success"}
