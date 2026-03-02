import time
from evdev import UInput, ecodes as e, AbsInfo
from pynput import keyboard

# 定义虚拟XBOX手柄的摇杆和按键能力 (与 unitree_mujoco 的 Xbox 布局一致)
cap = {
    e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y, e.BTN_TL, e.BTN_TR, e.BTN_SELECT, e.BTN_START],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0)),  # LX (0)
        (e.ABS_Y, AbsInfo(value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0)),  # LY (1)
        (e.ABS_RX, AbsInfo(value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0)), # RX (3)
        (e.ABS_RY, AbsInfo(value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0)), # RY (4)
    ]
}

# 创建虚拟设备
ui = UInput(cap, name='Virtual-Xbox-Controller', version=0x3)
print("虚拟XBOX手柄已创建。请保持此脚本运行。")
print("控制说明: W/A/S/D 控制左摇杆, 上下左右方向键控制右摇杆")
print("按键说明: J -> A键, K -> B键, U -> X键, I -> Y键")

# 摇杆状态缓存
state = {'LX': 0, 'LY': 0, 'RX': 0, 'RY': 0}
axis_max = 32767

def update_stick():
    ui.write(e.EV_ABS, e.ABS_X, state['LX'])
    ui.write(e.EV_ABS, e.ABS_Y, state['LY'])
    ui.write(e.EV_ABS, e.ABS_RX, state['RX'])
    ui.write(e.EV_ABS, e.ABS_RY, state['RY'])
    ui.syn()

def on_press(key):
    try:
        # 左摇杆 (W/A/S/D)
        if key.char == 'w': state['LY'] = -axis_max
        elif key.char == 's': state['LY'] = axis_max
        elif key.char == 'a': state['LX'] = -axis_max
        elif key.char == 'd': state['LX'] = axis_max
        
        # 动作按键 (A, B, X, Y)
        elif key.char == 'j': ui.write(e.EV_KEY, e.BTN_A, 1)
        elif key.char == 'k': ui.write(e.EV_KEY, e.BTN_B, 1)
        elif key.char == 'u': ui.write(e.EV_KEY, e.BTN_X, 1)
        elif key.char == 'i': ui.write(e.EV_KEY, e.BTN_Y, 1)
        
        ui.syn()
    except AttributeError:
        # 右摇杆 (方向键)
        if key == keyboard.Key.up: state['RY'] = -axis_max
        elif key == keyboard.Key.down: state['RY'] = axis_max
        elif key == keyboard.Key.left: state['RX'] = -axis_max
        elif key == keyboard.Key.right: state['RX'] = axis_max
    
    update_stick()

def on_release(key):
    try:
        # 归零左摇杆
        if key.char in ['w', 's']: state['LY'] = 0
        elif key.char in ['a', 'd']: state['LX'] = 0
        
        # 释放动作按键
        elif key.char == 'j': ui.write(e.EV_KEY, e.BTN_A, 0)
        elif key.char == 'k': ui.write(e.EV_KEY, e.BTN_B, 0)
        elif key.char == 'u': ui.write(e.EV_KEY, e.BTN_X, 0)
        elif key.char == 'i': ui.write(e.EV_KEY, e.BTN_Y, 0)
        
        ui.syn()
    except AttributeError:
        # 归零右摇杆
        if key in [keyboard.Key.up, keyboard.Key.down]: state['RY'] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]: state['RX'] = 0

    update_stick()

# 启动键盘监听
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()