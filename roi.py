import cv2

# Variables globales para almacenar las coordenadas del rectángulo
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image_display # image_display para dibujar en la imagen visible

    # Si se hizo clic izquierdo
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # Si el ratón se mueve mientras se arrastra (clic izquierdo presionado)
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        # Crea una copia temporal para dibujar el rectángulo sin modificar la original
        temp_image = image.copy() # O 'image_display' si ya la has redimensionado
        cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Imagen para ROI", temp_image)

    # Si se soltó el clic izquierdo
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Dibuja el rectángulo final en la imagen original (o la anotada si quieres)
        cv2.rectangle(image_display, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Imagen para ROI", image_display)

        # Imprime las coordenadas
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        # Asegúrate de que las coordenadas sean mínimas y máximas
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        print(f"ROI seleccionada: [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]")
        # Aquí puedes almacenar estas coordenadas en una lista para futuras ROIs
        # o pasarlas directamente a EasyOCR.

# --- Tu código principal ---
image = cv2.imread('27084609001.jpg') # Carga tu imagen
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Prepara la imagen para mostrar (la misma lógica de redimensionado que ya tienes)
(h, w) = image.shape[:2]
max_display_width = 1200
max_display_height = 800

if w > max_display_width or h > max_display_height:
    width_scale = max_display_width / w
    height_scale = max_display_height / h
    scale = min(width_scale, height_scale)
    new_width = int(w * scale)
    new_height = int(h * scale)
    image_display = cv2.resize(image.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)
else:
    image_display = image.copy()


cv2.namedWindow("Imagen para ROI")
cv2.setMouseCallback("Imagen para ROI", click_and_crop)

print("Haz clic y arrastra para seleccionar una ROI. Presiona 'q' para salir.")

while True:
    cv2.imshow("Imagen para ROI", image_display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): # Presiona 'q' para salir
        break

cv2.destroyAllWindows()