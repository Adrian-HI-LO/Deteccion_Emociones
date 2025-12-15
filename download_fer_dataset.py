#!/usr/bin/env python3
"""
Script para descargar el dataset FER-2013 de Kaggle
Competencia: Facial Expression Recognition Challenge
"""

import os
import sys
import subprocess
import zipfile
import shutil

def check_kaggle_credentials():
    """Verifica que existan las credenciales de Kaggle"""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("ERROR: No se encontraron credenciales de Kaggle")
        print("\nPara configurar Kaggle:")
        print("1. Ve a https://www.kaggle.com/settings/account")
        print("2. En 'API', haz clic en 'Create New Token'")
        print("3. Se descargará kaggle.json")
        print("4. Mueve el archivo: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("5. Asegura permisos: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Verificar permisos
    current_perms = oct(os.stat(kaggle_json).st_mode)[-3:]
    if current_perms != '600':
        print(f"Ajustando permisos de {kaggle_json}...")
        os.chmod(kaggle_json, 0o600)
    
    return True

def download_dataset():
    """Descarga el dataset de Kaggle"""
    print("Descargando dataset FER-2013 de Kaggle...")
    print("Esto puede tomar varios minutos dependiendo de tu conexión...\n")
    
    # Crear directorio DataSets si no existe
    os.makedirs("DataSets", exist_ok=True)
    
    # Descargar dataset
    competition_name = "challenges-in-representation-learning-facial-expression-recognition-challenge"
    
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_name, "-p", "DataSets"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("✓ Descarga completada")
        return True
        
    except subprocess.CalledProcessError as e:
        print("ERROR al descargar:")
        print(e.stderr)
        if "403" in e.stderr or "access" in e.stderr.lower():
            print("\n⚠️ Necesitas aceptar las reglas de la competencia:")
            print(f"   https://www.kaggle.com/c/{competition_name}/rules")
            print("   1. Visita el enlace")
            print("   2. Haz clic en 'I Understand and Accept'")
            print("   3. Vuelve a ejecutar este script")
        return False
    except Exception as e:
        print(f"ERROR inesperado: {e}")
        return False

def extract_dataset():
    """Extrae el archivo ZIP descargado"""
    zip_file = "DataSets/challenges-in-representation-learning-facial-expression-recognition-challenge.zip"
    
    if not os.path.exists(zip_file):
        print(f"ERROR: No se encontró {zip_file}")
        return False
    
    print(f"\nExtrayendo {zip_file}...")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("DataSets")
        print("✓ Extracción completada")
        
        # Listar archivos extraídos
        print("\nArchivos extraídos:")
        for file in os.listdir("DataSets"):
            if file.endswith('.csv'):
                size = os.path.getsize(f"DataSets/{file}") / (1024*1024)
                print(f"  - {file} ({size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"ERROR al extraer: {e}")
        return False

def cleanup_old_dataset():
    """Elimina el dataset antiguo de archive/"""
    old_dataset = "DataSets/archive"
    
    if os.path.exists(old_dataset):
        print(f"\n¿Eliminar dataset antiguo en {old_dataset}? (s/N): ", end='')
        response = input().strip().lower()
        
        if response == 's':
            shutil.rmtree(old_dataset)
            print("✓ Dataset antiguo eliminado")
        else:
            print("Dataset antiguo conservado")

def main():
    print("=" * 60)
    print("Descargador de FER-2013 Dataset")
    print("=" * 60)
    print()
    
    # Verificar credenciales
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Descargar dataset
    if not download_dataset():
        sys.exit(1)
    
    # Extraer dataset
    if not extract_dataset():
        sys.exit(1)
    
    # Preguntar si eliminar dataset antiguo
    cleanup_old_dataset()
    
    print("\n" + "=" * 60)
    print("✓ Dataset FER-2013 listo para usar")
    print("=" * 60)
    print("\nSiguientes pasos:")
    print("1. Actualiza train_emotions_model.py para usar el nuevo dataset")
    print("2. Ejecuta: python train_emotions_model.py")
    print()

if __name__ == "__main__":
    main()
