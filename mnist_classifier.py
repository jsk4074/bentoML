import bentoml
import tensorflow as tf

from bentoml.artifact import TensorflowSavedModelArtifact
from bentoml.adapters import TfTensorInput


MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class MnistClassifier(bentoml.BentoService):

    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict(self, inputs):
        outputs = self.artifacts.model.predict_image(inputs)
        output_classes = tf.math.argmax(outputs, axis=1)
        return [MNIST_CLASSES[c] for c in output_classes]