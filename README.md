# **FARMBOT**

- The docs folder contains my documentation about the farmbot project.
- The others folder includes various project-related files such as SolidWorks files, images, etc.
- The test folder contains files used for testing the code.
---

## Give admin permission to the USB port for UART communication
### JETSON NANO
```bash
sudo chmod 666 /dev/ttyUSB0 
```
### RasPi
```bash
sudo raspi-config
```

- Select: 3. Interface Options
- Select: I6 Serial Port
- Select: Yess

**Run file main.py**
```bash
python3 main.py
```
---
## Clone my project
```bash
git clone https://github.com/minhthong2514/FreeJobs.git
```

---