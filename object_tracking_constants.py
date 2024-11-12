import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop spam from tensorflow
import tensorflow as tf

session = 56
trial = 4
is_retro = False

all_camera_names = ['table-side', 'table-top', 'back', 'counter-top', 'ceiling', 'front']
rect_by_cam = ['table', 'table', 'table', 'counter', 'table', 'table'] # the rectangle to perform optimization of camera params on.

xdof = 6 # degrees of freedom per camera

alpha = 750 # focal length in pixels

corners = ['ul', 'ur', 'lr', 'll'] # how the corners of the table/counter are called

groundtruth_w = { # corner positions in world coords: upper left, upper right, lower right, lower left
    'table': tf.constant([[0.38, -1.6, 0.73], [0.38, 0, 0.73], [1.18, 0, 0.73], [1.18, -1.6, 0.73]], dtype=tf.float64),
    'counter': tf.constant([[-1.66, -0.415, 0.74], [-1.66, 1.385, 0.74], [-1.06, 1.385, 0.74], [-1.06, -0.415, 0.74]], dtype=tf.float64),
    'origin': tf.constant([[0, 0, 0]], dtype=tf.float64)
}

# does only one instance of this tableware_class exist?
predifined_classes = {'bowl--salad': True, 'bowl--cooker': True, 'plate--pasta': True, 'bread': True, 'butter': True, 'jam': True, 'nutella': True, 'salt': True, 'shaker--pepper': True, 'sugar': True, 'cereal': True, 'milk': True, 'coffee': True ,'wine--bottle': True, 'water': True, 'bowl--cereal': False, 'plate': False, 'knife-bread': True, 'utensil, pasta': True, 'ladle': True, 'utensil-spoon, salad': True, 'utensil-fork, salad': True, 'teaspoon': False, 'tablespoon': False, 'fork': False, 'knife': False, 'glass-water': False, 'glass--wine': False, 'cup-coffee': False}

img_shape = (720, 1280, 3)