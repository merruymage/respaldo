import cv2
import numpy as np
import matplotlib.pyplot as plt # Para visualizar la imagen binarizada
import os # Para manipular rutas de archivos y guardar la imagen

# --- Función para la binarización Niblack (o su aproximación) ---
def apply_niblack(image_gray, window_size=51, k=0.2):
    """
    Aplica una binarización tipo Niblack usando la binarización adaptativa de OpenCV.
    Esta es una excelente aproximación para texto y caligrafía.

    Args:
        image_gray (np.array): La imagen en escala de grises.
        window_size (int): Tamaño de la ventana para el cálculo adaptativo. Debe ser un número impar.
                           Valores comunes: 15, 21, 31, 41, 51. Experimenta con esto.
        k (float): Constante que se resta del umbral.
                   Valores comunes: de -0.2 a +0.2. Positivos engrosan, negativos adelgazan.
    Returns:
        np.array: La imagen binarizada (fondo blanco, texto negro).
    """
    # Asegura que el tamaño de la ventana sea impar
    if window_size % 2 == 0:
        window_size += 1
    # Asegura que el tamaño de la ventana sea al menos 3
    if window_size < 3:
        window_size = 3

    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C es generalmente la mejor opción para texto.
    # cv2.THRESH_BINARY_INV invierte los colores (texto oscuro sobre fondo claro, que es lo que quieres para OCR).
    binary_image = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window_size, k
    )
    return binary_image

# --- Ruta a tu imagen ---
image_path = '27084609001.jpg' # ¡Asegúrate de que esta ruta sea correcta!
output_dir = 'binarized_images' # Directorio donde se guardará la imagen binarizada

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# --- Cargar y binarizar la imagen ---
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {image_path}")

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Parámetros para la binarización ---
    # AJUSTA ESTOS VALORES PARA MEJORAR LA BINARIZACIÓN DE TU CALIGRAFÍA
    # Puedes probar diferentes combinaciones para ver cuál da el mejor resultado
    binarization_window_size = 41  # Prueba con 21, 31, 41, 51, etc. (siempre impar)
    binarization_k_value = 5.0     # Prueba con 0.2, 5.0, -2.0, etc. (este es muy sensible)

    niblack_binary_img = apply_niblack(gray_img, 
                                       window_size=binarization_window_size, 
                                       k=binarization_k_value)

    # --- Mostrar la imagen binarizada ---
    plt.figure(figsize=(8, 8))
    plt.title(f'Imagen Binarizada (Window Size: {binarization_window_size}, k: {binarization_k_value})')
    plt.imshow(niblack_binary_img, cmap='gray')
    plt.axis('off') # Oculta los ejes para una mejor visualización
    plt.show()

    # --- Guardar la imagen binarizada ---
    # Construir el nombre del archivo de salida
    base_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{file_name_without_ext}_binarized.png" # Usar PNG para mantener la calidad binaria
    output_path = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_path, niblack_binary_img)
    print(f"\nImagen binarizada guardada exitosamente en: {output_path}")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"Ocurrió un error: {e}")