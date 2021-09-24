import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
# from tensorflow.keras.losses import MeanSquaredLogarithmicError
import joblib
import logging
from sklearn.exceptions import NotFittedError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import datetime
from pathlib import Path

# from torch import tensor

from tensorflow.keras import backend as K
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import msle
import tensorflow as tf
from tensorflow.keras import Model
# from AnamolyDetection_AutoEncoder import calculate_similarity_loss
# import numpy as np


plt.rcParams["figure.figsize"] = [16,9]



def generate_save_path(log_dir):
    """
    generates the path to save model checkpoints
    """
    _dir =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(log_dir + r'\tf_' + _dir)



def auto_encoder_1(output_units=140, code_size=8):
    """
    Auto-encoder architecture
    """
    model = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(code_size, activation='relu'),
      
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
    ])
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')
    return model


def auto_encoder_2(output_units=6, code_size=3):
    model = Sequential([
            Dense(5, activation='relu'),
            Dropout(0.1),
            Dense(4, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='relu'),
            Dropout(0.1),
            Dense(code_size, activation='relu'),
      
            Dense(3, activation='relu'),
            Dropout(0.1),
            Dense(4, activation='relu'),
            Dropout(0.1),
            Dense(5, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
    ])
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')

    return model



# class AutoEncoder(Model):
#     def __init__(self,output_units=140, code_size=8):
#         super().__init__()
#         self.encoder = Sequential([
#             Dense(64, activation='relu'),
#             Dropout(0.1),
#             Dense(32, activation='relu'),
#             Dropout(0.1),
#             Dense(16, activation='relu'),
#             Dropout(0.1),
#             Dense(code_size, activation='relu')
#         ])
#         self.decoder = Sequential([
#             Dense(16, activation='relu'),
#             Dropout(0.1),
#             Dense(32, activation='relu'),
#             Dropout(0.1),
#             Dense(64, activation='relu'),
#             Dropout(0.1),
#             Dense(output_units, activation='sigmoid')
#         ])
  


#     def call(self, inputs):
#         decoded = self.encoder_decoder(inputs)
#         # decoded = self.decoder(encoded)
#         return decoded



class AutoEncoder:
    """
    Auto-Encoder wrapper class for preparing data, training, testing, 
    calculating the threshold values, and getting predictions
    """

    def __init__(self,model,log_dir = r'C:\Users\daniy\Documents\vu\mthesis\logs'):
        logging.info(tf.config.list_physical_devices('GPU'))
        self.scaler = None
        # print(model)
        self.model = model
        # self.model.compile(loss='msle', metrics=['mse'], optimizer='adam')

        self.callbacks = []

        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

        # log_dir = Path( log_dir +'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
        self.log_dir = log_dir
        self.save_dir = None

        self.tensorboard_hist_freq = 1
        self.early_stopping = True
        self.tensorboard = True
        # self.weights_biases = self.model.weights
        # self.weights = get_weights_only(self.model)
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # self.tensorboard_callback 

        # self.compile_callbacks()

    
    def load_model(self, _path):
        model_path = Path(Path(_path) / 'model')
        scaler_path = Path(Path(_path)/ 'scaler.save')
        self.model.load_weights(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model.built=True
        # self.weights_biases = self.model.weights
        # self.weights = get_weights_only(self.model)

    def generate_log_callback(self):
        _path = generate_save_path(self.log_dir)
        self.save_dir = _path
        # save_path = Path(_path / r'cp.ckpt')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_path, histogram_freq=1)
        # save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
        #                                          save_weights_only=True,
        #                                          verbose=1)
        logging.info(f'Tensorboard: {_path}')
        # logging.info(f'Checkpoint path: {save_path}')
        return tensorboard_callback


    def train(self,x_train, x_test, y_train=None, y_test=None, callbacks= [], epochs = 70, verbose =True, batch_size=100):

        if y_train is None:
            y_train = x_train
        if y_test is None:
            y_test = x_test

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        callbacks.append(self.early_stopping_callback)
        callbacks.append(self.generate_log_callback())

        history = self.model.fit(
        x_train,
        y_train,
        verbose = verbose,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks= callbacks
        )
        self.save_scaler()
        self.save_model()
        # print(self.model.evaluate(x=x_test, y=x_test, return_dict=True, verbose=True))
        return history

    def train_autoencoder(self, x_train, y_train, x_test, y_test):

        pass

    def save_model(self):
        save_path = Path(self.save_dir/ 'model')
        logging.info(f'Model saved: {save_path}')
        saved = self.model.save_weights(save_path)
        logging.info(saved)

    def save_scaler(self):
        save_path = Path(self.save_dir / 'scaler.save')
        joblib.dump(self.scaler, save_path)
        logging.info(f'Scaler saved: {save_path}')

    def prerpare_data(self,data, split = 0.2, normal_label= 1, target_col = 140, scaler = MinMaxScaler(feature_range=(0, 1))):
        
        if normal_label is None:
            norm_only = data.index
        else:
            norm_only = data[data[target_col] == normal_label].index

        y = data[data.index.isin(norm_only)][target_col]
        x = data[data.index.isin(norm_only)].drop(columns = [target_col])
        x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=split)

        abnormal_data_x = data[~data.index.isin(norm_only)].drop(columns = [target_col])
        abnormal_data_y = data[~data.index.isin(norm_only)][target_col]

        if self.scaler is None:
            self.scaler = scaler
        try:
            x_train_scaled = self.scaler.transform(x_train.copy())
        except NotFittedError:
            x_train_scaled = self.scaler.fit_transform(x_train.copy())

        x_test_scaled = self.scaler.transform(x_test.copy())

        if len(abnormal_data_x) > 0:
            abnormal_data_x = self.scaler.transform(abnormal_data_x.copy())

        return x_train_scaled, x_test_scaled, y_train, y_test, abnormal_data_x, abnormal_data_y




    def get_reconstruction_errors(self, data):
        reconstructions = self.model.predict(data)

        return calculate_similarity_loss(reconstructions, data)
        # reconstruction_errors = tf.keras.losses.msle(reconstructions, data)

        # return(pd.Series(reconstruction_errors))

    def find_threshold(self, data):
        reconstructions = self.model.predict(data)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.msle(reconstructions, data)
        # threshold for anomaly scores
        threshold = np.mean(reconstruction_errors.numpy()) \
            + np.std(reconstruction_errors.numpy())
        return threshold

    def get_predictions(self, data, threshold = 0.0051954877254968695):
        predictions = self.model.predict(data)
        # provides losses of individual instances
        errors = tf.keras.losses.msle(predictions, data)
        # 0 = anomaly, 1 = normal
        anomaly_mask = pd.Series(errors) > threshold
        preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
        return preds

   


    def get_accuracy(self, x ,y, threshold = 0.006):
        predictions = self.get_predictions( x, threshold)
        return(accuracy_score(predictions, y))

    def get_model_layer_outputs(self, input):
        n_layers = len(self.model.layers)


class LambdaModel:
    """
    Wrapper class to take trained trained Auto-encoder and add lambda layer for classification
    """
    def __init__(self, ae_model, threshold = 0.0051954877254968695, input_shape=(None,140)):
        
        self.ae_model = ae_model
        self.ae_model.build(input_shape=input_shape)
        self.threshold = threshold
        self.model = self.build_model(self.ae_model)
    
    

    def get_prediction_activation(self, x):
    
        # print(x.shape)
        x1 = tf.convert_to_tensor(x[0][0:140])
        x2 = tf.convert_to_tensor(x[0][140:])
        # print(x.shape,x1.shape,x2.shape)
        loss = msle(x1,x2)
        return tf.cond(tf.greater_equal(loss, self.threshold, name='activation_condition'),  lambda: tf.constant([0]),lambda:tf.constant([1]))
        #     return tf.constant(1)
    # return tf.constant(0)

    def build_model(self, ae_model):
        concat = tf.keras.layers.Concatenate()([ae_model.input, ae_model.output])
        lambda_layer = Lambda(self.get_prediction_activation)(concat)
        lambda_model = Model(ae_model.input, lambda_layer)

        return lambda_model

    def predict(self, data):
        preds = []
        for i in tqdm(data):
            preds.append(self.model.predict(np.array([i])))

        return preds


def calculate_similarity_loss(x1, x2, as_df = True):
    """
    calculating reconstrotion error/similarity loss given 2 tensors of similar dimensions
    """
    reconstruction_errors = tf.keras.losses.msle(x1, x2)
    if as_df:
        return(pd.Series(reconstruction_errors))
    return reconstruction_errors





from tensorflow.keras import Model
from tqdm.notebook import tqdm
from scipy.stats.stats import pearsonr, spearmanr
def get_layer_activations(model, data, skip_layers = ['dropout','concatenate', 'input']):
    """
    get activations of all nodes in all layers, except for skip layers, of a model for input (data)
    """
    data = np.array(data)
    outputs = np.array([])
    outputs = np.append(outputs, data)
#     print(outputs.shape)
#     outputs.append(data)
    for n,i in enumerate(model.layers):   
        # print(i.name)
        l = [s for s in skip_layers if  s.lower() in i.name.lower() ]
       
        # if 'dropout' not  in i.name and 'concatenate' not in i.name and i.name != 'dense_input':
        if not l:
            # print(i.name)
        # if any(i.name in s for s in exclude):
        # if i.name not in exclude:
            # print(i.name)
            earlyPredictor = Model(model.inputs, model.layers[n].output)
            outputs = np.append(outputs, np.array(earlyPredictor(data)))
#             print(outputs.shape)
#             outputs.append(np.array(earlyPredictor(data)))
        # else:
        #    earlyPredictor = Model(ae.model.inputs, ae.model.layers[n].output)
        #    _outputs.append(np.array(earlyPredictor(np.array([x_train[0]]))))

    # return np.array(outputs)
    return outputs

def get_activation_values(model, data):
    """
    get activations of all nodes in all layers, except for skip layers, of a model for a set of input (data)
    """
    outputs = []
    for i in tqdm(data):
#         output = get_activation_values(model, [i])
        
#         arr = []
#         for j in output:
#             arr += list(j)
#         outputs.append(arr)
        outputs.append(get_layer_activations(model, [i]))
    return outputs




def get_coactivation_values(df):
    """
    Use df of activation value of model to get coactivation values
    """

    mat = np.zeros((df.shape[1],df.shape[1]))

    for i in tqdm(df.columns):
        for j in df.columns:
            corr = spearmanr(df[i], df[j])[0]
        
                
                
            mat[i][j] = corr
    df_corr = pd.DataFrame(mat, index = df.columns, columns = df.columns)
    return df_corr


def create_edge_df(df_corr):
    """
    Create df for agency matrix for edge of coactivation graph using coactivation values
    """
    skip = 0
    values = []
    source = []
    dest = []
    for col in df_corr.columns:
        for _col in df_corr.columns[skip:]:
            if _col !=  col:
                values.append(df_corr.iloc[col, _col])
                source.append(col)
                dest.append(_col)
            # values.append(df_corr.iloc[col, _col])
        skip+=1
    edges = pd.DataFrame()
    edges['Source'] = source
    edges['Target'] = dest
    edges['Type'] = 'Undirected'
    edges['Weight'] = values

    return edges




def mask_weights(node_placement, weights):

    "mask the output of selected neurons"
    count = 0
#     _ = []
    for n, i in enumerate(weights):
        if count == node_placement[0]:
#             print(count, i.shape)
            if len(i.shape) == 2:
                _shape = i.shape[-1]
                weights[n][node_placement[-1]] = np.zeros(shape = (_shape))
        if n == 0:
            pass
        else:
            if len(i) != len(weights[n-1]):
                count+=1
#         print(count, i.shape)
    return weights
#                 print(i.shape)
#                 break
#     print(len(i), count,i.shape)



def get_neuron_structure(model,skip_layers = ['dropout','concatenate'], output_shape = (None,140)):
    """
    get dimensionality of indivdual layers in the network
    """
    shape = []
    for n,i in enumerate(model.layers):   
        
        l = [s for s in skip_layers if  s.lower() in i.name.lower() ]
       
        
        if not l:
            _shape = i.input_shape
#             print(i.name, shape)
            shape.append((_shape, i.name))
    shape.append((output_shape, 'output'))
    return(shape)

def get_node_placement(node_id, model_structure):
    """
    get location of node based on node_id
    """

    prev = 0
    for n,i in enumerate(model_structure):
#         if n == 0:
        shape = i[0][-1]
        
        shape = shape+prev

        if node_id < shape:
            return (n,abs(prev-(node_id % shape)))
#         print(n)
        prev=shape
    return -1

###############################################################################

class NN_Graph:

    def __init__(self, model):
        self.model = model
        self.weights_biases = model.weights
        self.weights = get_weights_only(self.model)
        

    
    def generate_network_graph(self,model_input, **kwargs):
        outputs = get_layer_activations(self.model, np.array([model_input]))
        graph = generate_network_graph(self.weights, outputs, **kwargs)
        return graph


def draw_network(graph, node_attr = 'activation', edge_attr = 'weight'):
    activations = [graph.nodes[i][node_attr] for i in list(graph.nodes)]

    if edge_attr is None:
        pass
    else:
        edges,weights = zip(*nx.get_edge_attributes(graph,edge_attr).items())


    pos = nx.multipartite_layout(graph)

    if edge_attr is None:
        nx.draw(graph,pos,  node_color = activations, nodelist = graph.nodes(), cmap= plt.cm.GnBu_r ,node_size=50)
    else:    
        nx.draw(graph,pos, edge_color = weights, node_color = activations, nodelist = graph.nodes(), cmap= plt.cm.GnBu_r ,node_size=50, edge_cmap=plt.cm.Reds)

    normalize = mcolors.Normalize(vmin=0, vmax=1)

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.GnBu_r)
    scalarmappaple.set_array(activations)
    plt.colorbar(scalarmappaple)


    normalize = mcolors.Normalize(vmin=np.array(weights).min(), vmax=np.array(weights).max())

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.Reds)
    scalarmappaple.set_array(weights)
    plt.colorbar(scalarmappaple)
    plt.show()
        
# def get_layer_activations(model, data):
#    outputs = []
#    outputs.append(data)
#    for n,i in enumerate(model.layers):   
#       # print(i.name)
#       if 'dropout' not  in i.name:
#          earlyPredictor = Model(model.inputs, model.layers[n].output)
#          outputs.append(np.array(earlyPredictor(data)))
#       # else:
#       #    earlyPredictor = Model(ae.model.inputs, ae.model.layers[n].output)
#       #    _outputs.append(np.array(earlyPredictor(np.array([x_train[0]]))))

#    return np.array(outputs)





def generate_network_graph(model_weights, outputs, directed=False, edge_weights = True):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for nk, k in enumerate(model_weights):

        for ni, i in enumerate(np.array(k)):


            node_i = ni*10 + nk
            if not G.has_node(node_i):
                
                G.add_node(node_i)
            for nj,j in enumerate(i):
                node_j = nj*10 + nk+1
                G.add_node(node_j)
                if edge_weights:
                    G.add_edge(node_i, node_j, weight = float(j))
                else:
                    G.add_edge(node_i, node_j)




    for nk, k in enumerate(outputs):
        k = k[0]
        for ni, i in enumerate(k):
            # output = i[0]
            node_i = ni*10 + nk
            G.nodes[node_i]['activation'] = i
            G.nodes[node_i]['subset'] = nk
    
    return G


def get_weights_only(model):
    weights = []
    for i in model.weights:
        if 'bias' not in i.name:
            weights.append(i)

    return weights
