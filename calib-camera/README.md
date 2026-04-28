# Calibrate Camera
**This step is very important, it will affect to performant system. Therefore, we need to calibrate carefully.The first thing to do is turn off auto mode of camera.
To do this on ubuntu, we must use v2l2-ctl.**

1. if your system isn't install, you can copy and paste in your command:
```bash
sudo apt install v4l-utils
```
2. Check the list controls of port video:
```bash
v4l2-ctl -d /dev/video0 --list-ctrls
```
3. Check camera resolution:
```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```
After this step, a list of currently available controls on your camera will be displayed.
