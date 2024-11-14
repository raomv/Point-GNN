import tensorflow as tf

# Verifica si hay GPUs disponibles
if tf.test.is_gpu_available(cuda_only=True):
    print("GPU está disponible para TensorFlow.")
    print("Nombre de la GPU:", tf.test.gpu_device_name())
else:
    print("No se encontró GPU disponible para TensorFlow.")
print(tf.__version__)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("version de tensorflow", tf.version.VERSION)