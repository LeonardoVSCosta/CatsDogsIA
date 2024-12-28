import os
import warnings

import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carregar o modelo.
modelo = load_model("gatos_cachorros_modelo.h5")

# Local da imagem de teste.
imagem_caminho = r"#"

# Adequa a imagem escolhida ao modelo.
imagem = image.load_img(imagem_caminho, target_size=(150, 150))
imagem_array = image.img_to_array(imagem) / 255.0
imagem_array = np.expand_dims(imagem_array, axis=0)

# Faz a previsão e exibe o resultado.
previsao = modelo.predict(imagem_array)

if previsao[0][0] < 0.5:
    print("É um cachorro na imagem!")
else:
    print("É um gato na imagem!")
