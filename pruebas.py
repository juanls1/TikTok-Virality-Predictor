from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Inicializar el navegador
driver = webdriver.Chrome()

# Abrir la página web
driver.get("https://www.tiktok.com/es/")
time.sleep(10)

#obtener el elemento button con texto "Rechazar todo"
button = driver.find_elements(By.CLASS_NAME, "button-wrapper")
print("---------")
print(button)
print("---------")

# Obtener todos los elementos de la página
elements = driver.find_elements("xpath", "//*")


# Cerrar el navegador
driver.quit()
