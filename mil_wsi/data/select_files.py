import os
import shutil

# Definir rutas relativas
source_folder = "mil_wsi/results/bracs/pt_files"
extention = ".pt"
destination_folder = os.path.join(source_folder, "no_pasar")


# Asegurar que la carpeta destino existe
os.makedirs(destination_folder, exist_ok=True)

# Lista de archivos permitidos sin la extensión
allowed_files = [
    "BRACS_264", "BRACS_265", "BRACS_286", "BRACS_292", "BRACS_301", "BRACS_305", "BRACS_310", "BRACS_735",
    "BRACS_736", "BRACS_737", "BRACS_738", "BRACS_739", "BRACS_740", "BRACS_1261", "BRACS_1275", "BRACS_1276",
    "BRACS_1283", "BRACS_1284", "BRACS_1286", "BRACS_1295", "BRACS_1296", "BRACS_1330", "BRACS_1334", "BRACS_1366",
    "BRACS_1367", "BRACS_1412", "BRACS_1416", "BRACS_1423", "BRACS_1424", "BRACS_1473", "BRACS_1474", "BRACS_1476",
    "BRACS_1579", "BRACS_1584", "BRACS_1585", "BRACS_1588", "BRACS_1591", "BRACS_1595", "BRACS_1598", "BRACS_1601",
    "BRACS_1608", "BRACS_1609", "BRACS_1614", "BRACS_1619", "BRACS_1632", "BRACS_1820", "BRACS_1821", "BRACS_1824",
    "BRACS_1842", "BRACS_1843"
]

# Buscar y mover archivos no permitidos
for file in os.listdir(source_folder):
    if file.endswith(extention):
        
        file_name = os.path.splitext(file)[0]  # Quitar la extensión
        print(file_name)
        if file_name not in allowed_files:  # Si no está en la lista
            src_path = os.path.join(source_folder, file)
            dst_path = os.path.join(destination_folder, file)
            shutil.move(src_path, dst_path)
            print(f"Movido: {file} -> {destination_folder}")

print("Proceso completado.")
