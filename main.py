import os
import warnings
import absl.logging

"""" Desabilita os avisos desnecessários do TensorFlow do terminal, para funcionar
 tem que ser chamado antes de importar o "Keras"."""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho para as pastas onde estão os dados de teste e treinamento.
teste = r"#"
treino = r"#"

# Carrega os dados para testa-los.
carregar_teste = ImageDataGenerator(rescale=1. / 255)
carregar_treino = ImageDataGenerator(rescale=1. / 255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

treinar_modelo = carregar_treino.flow_from_directory(treino, target_size=(150, 150), batch_size=32, class_mode="binary")

testar_modelo = carregar_teste.flow_from_directory(teste, target_size=(150, 150), batch_size=32, class_mode="binary")

# Criação do modelo.
modelo = Sequential([Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)), MaxPooling2D(2, 2),
                     Conv2D(64, (3, 3), activation="relu"), MaxPooling2D(2, 2), Conv2D(128, (3, 3), activation="relu"),
                     MaxPooling2D(2, 2), Flatten(), Dense(512, activation="relu"), Dropout(0.5),
                     Dense(1, activation="sigmoid")])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinar o modelo. Epochs = Quantos ciclos de treino eu quero.
treino = modelo.fit(treinar_modelo, epochs=15, validation_data=testar_modelo)

# Salvar o modelo.
modelo.save("gatos_cachorros_modelo.h5")

print("Modelo salvo!")
