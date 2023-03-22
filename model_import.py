from keras.models import model_from_json

def import_model(main_dir, sub_dir, number):
    models = []
    for i in range(number):
        json_f = open(main_dir + sub_dir + "/model_"+str(i)+".json", 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights((main_dir + sub_dir + "/model_"+str(i)+".h5"))
        models.append(loaded_model)  
    return models