import torch
from transformers import CvtConfig, CvtModel, CvtForImageClassification
import os


def quantize_model(model, layer_type_to_quantize='convolution_projection', save=False, compute=False):
    base_model_state_dict = model.state_dict()
    new_state_dict = {}
    layers_quantized = {}
    dict_for_size = []
    for layer_name, layer_params in base_model_state_dict.items():
        if layer_type_to_quantize in layer_name:
            new_state_dict[layer_name] = layer_params.to(torch.float16)
            layers_quantized[layer_name] = new_state_dict[layer_name].numel()
        else:
            new_state_dict[layer_name] = layer_params
        dict_for_size.append(new_state_dict[layer_name].numel())

    if save:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
        torch.save(new_state_dict, quantized_model_path)
        print("Quantized model state dict saved at:", quantized_model_path)

    if compute:
        original_model_size = sum(dict_for_size)
        num_of_Mb = 2*original_model_size/(1024**2)
        print("original size in Mb: " + str(num_of_Mb))

        quantized_num_parameters = sum(layers_quantized.values())
        num_of_Mb = 2*quantized_num_parameters/(1024**2)
        print("Number of Mb saved: " + str(num_of_Mb))

    return new_state_dict


"""
quantized_num_parameters = sum(layers_quantized.values())
print("Number of quantized parameters: " + str(quantized_num_parameters))

# Save the quantized model state dict
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
torch.save(new_state_dict, quantized_model_path)
print("Quantized model state dict saved at:", quantized_model_path)
    
# Initializing a Cvt msft/cvt style configuration
# configuration = CvtConfig()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# layer_type_to_quantize = 'convolution_projection'

############################### Example on how to apply quantization ###############################
# Initializing a model (with random weights) from the msft/cvt style configuration
# model = CvtModel(configuration)
base_model_state_dict = model.state_dict()

new_state_dict = {}
layers_quantized = {}

for layer_name, layer_params in base_model_state_dict.items():
    if layer_type_to_quantize in layer_name:
        new_state_dict[layer_name] = layer_params.half()
        layers_quantized[layer_name] = new_state_dict[layer_name].numel()
    else:
        new_state_dict[layer_name] = layer_params

quantized_num_parameters = sum(layers_quantized.values())
print("Number of quantized parameters: " + str(quantized_num_parameters))

# Save the quantized model state dict
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
torch.save(new_state_dict, quantized_model_path)
print("Quantized model state dict saved at:", quantized_model_path)

## ignore
# static quantiztion from tutorial 10

# set quantization config for server (x86) deployment
myModel.qconfig = torch.quantization.get_default_config('fbgemm')
# insert observers
torch.quantization.prepare(myModel, inplace=True)
# Calibrate the model and collect statistics
# convert to quantized version
torch.quantization.convert(myModel, inplace=True)


# dynamic quantization usage example
# TODO: need to see more about static quantization - how to apply only to conv proj 
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

"""