# for now this script only deals with how to load the base model with all of it's weights for validation

from transformers import CvtConfig, CvtModel, CvtForImageClassification

# Initializing a Cvt msft/cvt style configuration
configuration = CvtConfig()

# Initializing a model (with random weights) from the msft/cvt style configuration
model1 = CvtModel(configuration)

# Accessing the model configuration
configuration = model1.config

model2 = CvtForImageClassification.from_pretrained("microsoft/cvt-13", output_hidden_states=True, return_dict=True)

print("Finished! stop here ahsheli")
