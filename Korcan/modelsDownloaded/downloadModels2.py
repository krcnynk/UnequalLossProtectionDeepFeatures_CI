# for keras
# from classification_models.keras import Classifiers

# for tensorflow.keras
from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('densenet121')
model = ResNet18((224, 224, 3), weights='imagenet')