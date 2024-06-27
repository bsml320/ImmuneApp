from keras.models import Model 
from keras.models import model_from_json

def import_ba_model(model_dict):
    json_f = open(model_dict + 'binding/model_pan_binding.json', 'r')
    loaded_model_json = json_f.read()
    json_f.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dict + 'binding/model_pan_binding.h5')    
     
    return loaded_model

def import_el_model(model_dict):
    json_f = open(model_dict + 'ligands/model_pan_ligands.json', 'r')
    loaded_model_json = json_f.read()
    json_f.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_dict + 'ligands/model_pan_ligands.h5')  
       
    return loaded_model

def import_el_model_m(main_dir, sub_dir, number):
    models = []
    sizes = [2048,4096,8192,16384,32768]
    for i in range(number):
        json_f = open(main_dir + sub_dir + "/model_pan_ligands_size_%s_%s.json" % (str(sizes[i]), str(i+1)), 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(main_dir + sub_dir + "/model_pan_ligands_size_%s_%s.h5" % (str(sizes[i]), str(i+1)))
        models.append(loaded_model)  
    return models

def import_ba_model_m(main_dir, sub_dir, number):
    models = []
    sizes = [512,1024,2048,4096,8192]
    for i in range(number):
        json_f = open(main_dir + sub_dir + "/model_pan_binding_size_%s.json" % str(sizes[i]), 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(main_dir + sub_dir + "/pan_binding_model_size_%s_weights.h5" % str(sizes[i]))
        models.append(loaded_model)  
    return models