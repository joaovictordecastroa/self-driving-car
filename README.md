# self-driving-car

## Install requirements

Create virtual enviroment
```
python -m venv .\venv
```

Install packages
```
.\venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Config virtual joystick

1. Install latest version of [vJoy](https://github.com/shauleiz/vJoy) to create virtual joysticks ([sourceforge](https://sourceforge.net/projects/vjoystick/));
2. Execute latest version of [x360ce](https://www.x360ce.com/) to emulate the virtual joystick as Xbox360 joystick;
3. In x360ce, click in `Add...` at the top right of the screen to add a new controller;
4. Select just the option that `Product Name` is `vJoy Device` and click in `Add Selected Device`;
5. Copy the content in `configs/joystick_preset.xml`;
6. Click in `Paste Preset` at the bottom of the screen;


## Troubleshoot

If some error occurs in libGL.so.1 in OpenCV:
```
apt-get update && apt-get install libgl1
```
 