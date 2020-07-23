import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

parser = argparse.ArgumentParser(description = "Image Classifier Part II")
parser.add_argument('--image_path', default='./test_images/hard-leaved_pocket_orchid.jpg', type=str , help='The Image Path')
parser.add_argument('--model', default='best_midel.h5', type=str , help='The Model Path')
parser.add_argument('--topk', default=5, type=int , help='Top K Results')
parser.add_argument('--classes', default='label_map.json' , help='Class Names')

arguments = parser.parse_args()

image = arguments.image_path
model=arguments.model
topK=arguments.topk
classes= arguments.classes

with open('label_map.json', 'r') as f:
   class_names = json.load(f)

model2 = tf.keras.models.load_model('best_model.h5' ,custom_objects={'KerasLayer':hub.KerasLayer})  


def process_image(images):
    tensor_image = tf.convert_to_tensor(images)
    resized_image = tf.image.resize(tensor_image, (224,224))
    norm_image = resized_image/225
    np_image=norm_image.numpy()

    return np_image

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    test_image=np.asarray(image)
    processimage=process_image(test_image)
    redim=np.expand_dims(processimage, axis=0)
    prob_predict= model2.predict(redim)
    prob_predict=prob_predict.tolist()
    
    probs, classes = tf.math.top_k(prob_predict, k=top_k)
    
    probs= probs.numpy().tolist()[0]
    classes=classes.numpy().tolist()[0]
    return probs, classes



class_new_names = dict() 
for i in class_names:   
    class_new_names[str(int(i)-1)]=class_names[i]



def plot_image(image_path):


    image= Image.open(image_path)
    test_image = np.asarray(image)
    procimage = process_image(test_image)

    probs, classes= predict(image_path , model2, 5)
    print (probs)
    print ("Class Labels", classes)

    names = [class_new_names[str(idd)] for idd in classes]
    print("Flower Names",names)
   

if __name__ == "__main__" :
    predict(image, model, topK)
    plot_image(image)