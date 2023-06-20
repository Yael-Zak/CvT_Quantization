# for now this script only deals with how to load the base model with all of it's weights for validation
import torch
from transformers import CvtConfig, CvtModel, CvtForImageClassification

# Initializing a Cvt msft/cvt style configuration
configuration = CvtConfig()

# Initializing a model (with random weights) from the msft/cvt style configuration
model = CvtModel(configuration)
base_model_state_dict = model.state_dict()
new_state_dict = {}
for index, layer_name in enumerate(base_model_state_dict.keys()):
    # print("layer name is:" + layer_name + "\nlayer tesnor is of size:" + str(layer_tensor.size()))
    new_state_dict[layer_name] = base_model_state_dict[layer_name].half()
# Accessing the model configuration
model.load_state_dict(new_state_dict)

print("Finished! stop here ahsheli")
