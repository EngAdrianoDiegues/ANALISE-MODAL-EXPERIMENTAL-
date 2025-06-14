from cx_Freeze import setup, Executable

setup(
    name="AME",
    version="1.0",
    description="Aplicativo AME",
    executables=[Executable("AME.py")],
    options={
        "build_exe": {
            "include_files": ["Engrenagens.png", "foto_GVA.png"],  # Inclui arquivos de imagem
            "packages": ["kivy", "babel"],  # Inclua pacotes necessários
        }
    }
)
