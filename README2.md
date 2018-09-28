# Semantic Segmentation

Things to do:
Try to use a tf.keras.application model to the eager execution and see differences in training with from scratch

    '''
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=input_x,  pooling='avg')
    x = model.outputs[0]
    BUSCAR KERAS GET LAYER Y NAME
    e0 = tf.get_default_graph().get_tensor_by_name("activation/Relu:0")
    e1 = tf.get_default_graph().get_tensor_by_name("add_2/add:0")
    e2 = tf.get_default_graph().get_tensor_by_name("add_6/add:0")
    e3 = tf.get_default_graph().get_tensor_by_name("add_12/add:0")
    e4 = tf.get_default_graph().get_tensor_by_name("add_15/add:0")
    '''
Save model (see https://github.com/Shathe/MNasNet-Keras-Tensorflow)
Poner lo del learning rate descendente con una formula
Implement Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation
Implement the DenseASPP
Implement flip-inference (horizontal)
Implement multiscale inferece
Implement the auxialiary loss function
Implement models pretraiend on imagenet
Implement BiSeNet?

Interesting links: 
https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
