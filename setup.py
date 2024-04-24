from cx_Freeze import setup, Executable

setup(
    name="3d_reconstruction",
    version="1.0",
    description="Construct a 3D Model from 2D Images",
    executables=[Executable("gui.py")]
)
