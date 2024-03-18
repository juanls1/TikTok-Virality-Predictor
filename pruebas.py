from selenium import webdriver

# Inicializar el navegador
driver = webdriver.Chrome()

# Abrir la página web
driver.get("https://www.tiktok.com/es/")

# Obtener todos los elementos de la página
elements = driver.find_elements("xpath", "//*")

# Crear un conjunto para almacenar los nombres de clase únicos
class_names = set()

# Iterar sobre todos los elementos y obtener sus clases
for element in elements:
    try:
        # Reintenta encontrar el elemento para evitar el error de StaleElementReferenceException
        element = driver.find_element("xpath", "//*")
        classes = element.get_attribute("class")
        if classes:
            class_names.update(classes.split())
    except Exception as e:
        print("Error al obtener clases del elemento:", e)

# Imprimir los nombres de clase únicos
print("Nombres de clase únicos:")
for class_name in class_names:
    print(class_name)

# Cerrar el navegador
driver.quit()
