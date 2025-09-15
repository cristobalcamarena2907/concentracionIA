from PIL import Image
import numpy as np
import requests
from io import BytesIO
from sklearn.cluster import KMeans

# Descargar la imagen desde la URL
url = "https://tec.mx/sites/default/files/repositorio/Campus/Monterrey/rectoria-campus-monterrey-tec.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image = image.convert("RGB")

# Convertir a arreglo numpy y normalizar
img_np = np.array(image)
w, h = image.size
X = img_np.reshape(-1, 3) / 255.0

k = 128  # Puedes cambiar este valor para experimentar
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Sustituir cada pixel por el color de su centroide
compressed_img = kmeans.cluster_centers_[labels]
compressed_img = (compressed_img * 255).astype(np.uint8)
compressed_img = compressed_img.reshape(img_np.shape)

# Mostrar la imagen comprimida
Image.fromarray(compressed_img).show()