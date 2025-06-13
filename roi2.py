import cv2

image = cv2.imread('27084609001.jpg')
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Permite al usuario seleccionar múltiples ROIs.
# Controla: Presiona 'Espacio' o 'Enter' para finalizar la selección actual y empezar una nueva.
# Presiona 'Esc' para terminar el proceso de selección de múltiples ROIs.
rois = cv2.selectROIs("Selecciona Multiples Regiones de Interes", image, showCrosshair=True, fromCenter=False)

# rois será una lista de tuplas (x, y, ancho, alto)
print("Coordenadas de las ROIs seleccionadas:")
all_regions = []
for r in rois:
    x_min = int(r[0])
    y_min = int(r[1])
    x_max = int(r[0] + r[2])
    y_max = int(r[1] + r[3])
    print(f"  [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]")
    all_regions.append([x_min, y_min, x_max, y_max])
    # Opcional: Dibuja los ROIs seleccionados en la imagen
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow("ROIs Seleccionadas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ahora puedes usar 'all_regions' en tu `reader.readtext()`
# Ejemplo: result = reader.readtext(Image_path, region=all_regions)