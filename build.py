import os
import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()
    return process.returncode

# Create a virtual environment
venv_path = "venv"

if not os.path.exists(venv_path):
    print("Creating virtual environment...")
    return_code = run_command(f"{sys.executable} -m venv {venv_path}")
    if return_code != 0:
        print("Failed to create virtual environment.")
        sys.exit(1)
    print("Virtual environment created successfully.")
else:
    print("Virtual environment already exists.")

# Activate the virtual environment
print("Activating virtual environment...")
activate_script = "activate.bat" if sys.platform == "win32" else "activate"
activate_path = os.path.join(venv_path, "Scripts" if sys.platform == "win32" else "bin", activate_script)
activate_command = f"{activate_path}"

if sys.platform != "win32":
    activate_command = f"source {activate_command}"

# Install requirements
print("Installing requirements...")
with open("requirements.txt", "w") as f:
    f.write("jax[cuda12_pip]\n")
    f.write("jaxlib\n")

install_command = f"{activate_command} && pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_releases.html"
return_code = run_command(install_command)

if return_code != 0:
    print("Failed to install requirements.")
    sys.exit(1)

print("Requirements installed successfully.")
print("Project setup complete.")