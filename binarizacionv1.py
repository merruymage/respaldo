import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt # Para visualizar las imágenes

# --- Función para la binarización Niblack ---
# OpenCV no tiene Niblack directamente, así que lo implementamos o usamos una alternativa similar (como Niblack en scikit-image o implementarlo manualmente)
# Para este ejemplo, podemos simular un efecto similar o usar una implementación común.
# Una alternativa muy popular y efectiva en OpenCV es la binarización adaptativa.
# Sin embargo, si realmente quieres Niblack, a menudo se implementa con una ventana deslizante.
# Aquí te muestro cómo puedes usar una implementación de Niblack o binarización adaptativa de OpenCV como alternativa cercana y efectiva.

def apply_niblack(image_gray, window_size=51, k=0.2):
    """
    Aplica una binarización tipo Niblack (usando binarización adaptativa de OpenCV como aproximación).
    Para una implementación estricta de Niblack, podrías necesitar una librería como scikit-image o implementarlo manualmente.
    OpenCV's ADAPTIVE_THRESH_GAUSSIAN_C a menudo da resultados similares y es muy eficiente.

    Args:
        image_gray (np.array): La imagen en escala de grises.
        window_size (int): Tamaño de la ventana para el cálculo adaptativo. Debe ser un número impar.
        k (float): Constante de Niblack.

    Returns:
        np.array: La imagen binarizada.
    """
    if window_size % 2 == 0:
        window_size += 1 # Asegura que el tamaño de la ventana sea impar

    # Binarización adaptativa de OpenCV como una buena aproximación a Niblack
    # ADAPTIVE_THRESH_GAUSSIAN_C es a menudo la mejor opción para texto.
    binary_image = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window_size, k
    )
    return binary_image

# --- Configuración de EasyOCR ---
# Puedes especificar los idiomas que necesitas.
# Para español: ['es'], para inglés: ['en'], etc.
reader = easyocr.Reader(['es'], gpu=True) # Usa GPU si está disponible, si no, pon gpu=False

# --- Ruta a tu imagen ---
image_path = '27084609001.jpg' # ¡Cambia esto por la ruta real de tu imagen!

# --- Cargar y preprocesar la imagen con Niblack (o su aproximación) ---
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar binarización tipo Niblack
    niblack_binary_img = apply_niblack(gray_img, window_size=51, k=0.2)

    # --- Mostrar la imagen binarizada (opcional) ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagen Original (Escala de Grises)')
    plt.imshow(gray_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Imagen Binarizada con Niblack (Aproximación)')
    plt.imshow(niblack_binary_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # --- Pasar la imagen preprocesada a EasyOCR ---
    # EasyOCR espera una imagen en formato NumPy array.
    # Aquí pasamos la imagen binarizada para que EasyOCR la procese.
    result = reader.readtext(niblack_binary_img)

    # --- Imprimir los resultados ---
    print("\n--- Resultados de EasyOCR después de Niblack ---")
    for (bbox, text, prob) in result:
        print(f'Texto: "{text}", Confianza: {prob:.4f}')

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"Ocurrió un error: {e}")