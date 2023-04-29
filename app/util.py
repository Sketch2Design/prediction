import torch
import torchvision.transforms as T
import cv2
import os
import numpy as np

def load_model():
	path = os.path.join('model','model.pth')
	
	modal = torch.load(path, map_location=torch.device('cpu'))
	modal.to('cpu')
	modal.eval()

	return modal  


def pre_process(image_file):
    # get the image file name for saving output later on
	# Load image
	image = cv2.imdecode(np.frombuffer(image_file.file.read(),
										np.uint8),
						cv2.IMREAD_COLOR)
	# reseze 
	image = cv2.resize(image, (512, 512))
	orig_image = image.copy()
	# BGR to RGB
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
	# make the pixel range between 0 and 1
	image /= 255.0
	# bring color channels to front
	image = np.transpose(image, (2, 0, 1)).astype(np.float)
	# convert to tensor
	image = torch.tensor(image, dtype=torch.float).cpu()
	# add batch dimension
	image = torch.unsqueeze(image, 0)
		
	return image, orig_image



def post_process(image, prediction):
	classes = (
    	"background", "rectangle", "frame_rectangle", "ellipse", "frame_ellipse"
	)
	threshold = 0.85

	if len(prediction[0]['boxes']) != 0:
		boxes = prediction[0]['boxes'].data.numpy()
		scores = prediction[0]['scores'].data.numpy()
		# filter out boxes according to `detection_threshold`
		boxes = boxes[scores >= threshold].astype(np.int32)
		# get all the predicited class names
		pred_classes = [classes[i] for i in prediction[0]['labels'].cpu().numpy()]

			
	return boxes,pred_classes