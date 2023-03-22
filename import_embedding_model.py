from keras.models import Model 
from keras.models import model_from_json

def import_embedding_model(model_dict):
    json_f = open(model_dict + 'pan_ligand/model_pan_ligand.json', 'r')
    loaded_model_json = json_f.read()
    json_f.close()
    pan_ligand = model_from_json(loaded_model_json)
    pan_ligand.load_weights(model_dict + 'pan_ligand/model_pan_ligand.h5') 
    layer_model = Model(inputs=pan_ligand.input, outputs=pan_ligand.layers[24].output)
    
    return layer_model
