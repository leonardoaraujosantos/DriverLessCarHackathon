'''
HTTP Rest server for allowing the model to be infered from a json type request
References:
    https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
    https://s3.us-east-2.amazonaws.com/prettyprinted/flask_cheatsheet.pdf
    https://github.com/sugyan/tensorflow-mnist
    https://github.com/benman1/tensorflow_flask
    https://github.com/sofeikov/WebImageCrop/blob/master/main.py
    http://stackoverflow.com/questions/4628529/how-can-i-created-a-pil-image-from-an-in-memory-file
    https://uk.mathworks.com/help/matlab/ref/webread.html
    https://uk.mathworks.com/help/matlab/ref/webwrite.html
    http://uk.mathworks.com/matlabcentral/fileexchange/59673-upload-a-file-to-dropbox-directly-from-matlab
    https://uk.mathworks.com/help/matlab/ref/jsonencode.html
    https://uk.mathworks.com/help/mps/restfuljson/json-representation-of-matlab-data-types.html
'''

# Import Flask stuff
from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import argparse
import scipy.misc
import model
import io
from PIL import Image
import matplotlib.pyplot as plt

# Force to see just the first GPU
# https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
import os

# Parser command arguments
# Reference:
# https://www.youtube.com/watch?v=cdblJqEUDNo
parser = argparse.ArgumentParser(description='HTTP server to infer angles from images')
parser.add_argument('--port', type=int, required=False, default=8090, help='HTTP Port')
parser.add_argument('--model', type=str, required=False, default='save/model-0', help='Trained driver model')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--top_crop', type=int, required=False, default=126, help='Top crop to avoid horizon')
parser.add_argument('--bottom_crop', type=int, required=False, default=226, help='Bottom crop to avoid front of car')
args = parser.parse_args()

def init_tensorflow_model(gpu, model_path):
    # Set enviroment variable to set the GPU to use
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        print('Set tensorflow on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Build model and get references to placeholders
    model_in, model_out, labels_in, model_drop = model.build_graph_placeholder()

    # Load tensorflow model
    print("Loading model: %s" % model_path)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    return sess, model_in, model_out, labels_in, model_drop

def pre_proc_img(image, crop_start=126, crop_end=226):
    image = scipy.misc.imresize(np.array(image)[crop_start:crop_end], [66, 200]) / 255.0
    return image


# Initialize tensorflow
sess, model_in, model_out, labels_in, model_drop = init_tensorflow_model(args.gpu, args.model)

# Add app to use flask
app = Flask(__name__)


# Service that will return an angle given an image
# From matlab this would be
# webwrite('127.0.0.1:8090/angle_from_file);
@app.route('/angle_from_file', methods=['POST'])
def get_angle_from_file():
    # Get image file from json request
    imagefile = request.files['file']
    print (type(imagefile))

    # Convert file to a image (This way we don't need to save the image to disk)
    image_bytes = imagefile.stream.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert('RGB')

    # Convert PIL image to numpy array
    img_array = np.asarray(pil_image)

    # Do some image processing
    img_array = pre_proc_img(img_array)

    # Get steering angle from tensorflow model (Also convert from rad to degree)
    degrees = sess.run(model_out, feed_dict={model_in: [img_array], model_drop: 1.0})[0][0]

    return jsonify(output=float(degrees))



'''
Service that will return an angle given some data
JSON format
{
	"rows": "2",
	"cols": "3",
	"depth": "1",
	"data": [1,2,3,4,5,6]
}

# From matlab this would be
# A = [1,2,3; 4, 5, 6]
# options = weboptions('MediaType','application/json');
# webwrite('http://127.0.0.1:8090/angle_from_data',jsonencode(struct('rows',2,'cols',3,'depth',1,'data',A)), options)

'''
@app.route('/angle_from_data', methods=['POST'])
def get_angle_from_data():
    # Get data from json request
    request_rows = request.json['rows']
    request_cols = request.json['cols']
    request_depth = request.json['depth']
    request_data = request.json['data']

    # Transform data to numpy array
    img_array = np.array(request_data, dtype=np.uint8).reshape([int(request_rows),int(request_cols), int(request_depth)])

    plt.imshow(img_array)
    plt.show()

    # Do some image processing
    img_array = pre_proc_img(img_array)

    # Get steering angle from tensorflow model (Also convert from rad to degree)
    degrees = sess.run(model_out, feed_dict={model_in: [img_array], model_drop: 1.0})[0][0]

    return jsonify(output=float(degrees))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=args.port)