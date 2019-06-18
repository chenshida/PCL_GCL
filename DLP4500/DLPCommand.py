#!/usr/bin/python2
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__Copyright__ = 'Copyrigh 2019, PI'
__version__ = "1.0"
__data__ = "2019.04.09"

import os
import sys
import numpy as np
import hid
import ctypes
'''
CMD_HEAD =       [0x00, 0x40, 0x00, 0x03, 0x00, 0x39, 0x1a, 0x01] # load image from flash by index
LED_RED =        [0x00, 0x40, 0x00, 0x03, 0x00, 0x07, 0x1a, 0x01] # R
LED_GREEN =      [0x00, 0x40, 0x00, 0x03, 0x00, 0x07, 0x1a, 0x02] # G
LED_BLUE =       [0x00, 0x40, 0x00, 0x03, 0x00, 0x07, 0x1a, 0x04] # B
LED_RB =         [0x00, 0x40, 0x00, 0x03, 0x00, 0x07, 0x1a, 0x05] # RB

PWM_INVERT =     [0x00, 0x40, 0x00, 0x03, 0x00, 0x05, 0x1a, 0x00]# invert
PWM_NOT_INVERT = [0x00, 0x40, 0x00, 0x03, 0x00, 0x05, 0x1a, 0x01]# not invert

POWER_STANDBY =  [0x00, 0x40, 0x00, 0x03, 0x00, 0x00, 0x02, 0x01]
VIDEO_MODE =     [0x00, 0x40, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00]
LED_CURRENT =    [0x00, 0x40, 0x00, 0x05, 0x00, 0x01, 0x0b, 0xc8, 0x78, 0x7d]


GET_LED_ENABLE =        [0x00, 0xc0, 0x00, 0x02, 0x00, 0x07, 0x1a]
GET_LED_ENABLE_RES =    [0xc0, 0x00, 0x01, 0x00, 0x03, 0x1a]

GET_CURRENTS_VAL =      [0x00, 0xc0, 0x00, 0x02, 0x00, 0x01, 0x0b]
GET_CURRENTS_VAL_RES =  [0xc0, 0x00, 0x03, 0x00, 0x97, 0x78, 0x7d]

GET_LEDPWM_INVERT =     [0x00, 0xc0, 0x00, 0x02, 0x00, 0x05, 0x1a]
GET_LEDPWM_INVERT_RES = [0xc0, 0x00, 0x01, 0x00, 0x00, 0x1a]
'''


def listHidDevice():
    dev = hid.enumerate()
    for d in dev:
        print(d)
    for d in dev:
        if d["product_string"] == "DLPC350":
            print("Find DLP device, VID: %s, PID: %s" % (d["product_id"], d["vendor_id"]))
            pid = d["product_id"]
            vid = d["vendor_id"]
            return True, vid, pid
    return False, 0, 0


class LEDSelect(ctypes.Structure):
    _fields_ = [
        ("R_EN", ctypes.c_uint8),
        ("G_EN", ctypes.c_uint8),
        ("B_EN", ctypes.c_uint8),
    ]

class LEDCurrent(ctypes.Structure):
    _fields_ = [
        ("R_VALUE", ctypes.c_uint8),
        ("G_VALUE", ctypes.c_uint8),
        ("B_VALUE", ctypes.c_uint8),
    ]
class LEDStatus(ctypes.Structure):
    _fields_ = [
        ("LED_ENABLE", LEDSelect),
        ("LED_CURRENT", LEDCurrent),
        ("LED_CURRENT", LEDCurrent),
        ("INVERT", ctypes.c_uint8),
    ]

class DLP4500API():
    CMD_HEAD = {
        "SET": np.array([0x00, 0x40, 0x00, 0x03, 0x00]),
        "GET": np.array([0x00, 0xc0, 0x00, 0x02, 0x00])
    }
    SET_CMD_FUNC_CODE = {
        "LOAD_IMAGE": np.array([0x39, 0x1a]),
        "LED_SELECT": np.array([0x07, 0x1a]),
        "CURRENT_MODE": np.array([0x05, 0x1a]),
        "CURRENT_VALUE": np.array([0x01, 0x0b]),
        "OPERATION_MODE": np.array([0x00, 0x02]),
    }

    GET_CMD_FUNC_CODE = {
        "LED_ENABLE": np.array([0x07, 0x1a]),
        "CURRENT_VAL": np.array([0x01, 0x0b]),
        "LEDPWM_INVERT": np.array([0x05, 0x1a]),
    }

    USB_READ_MAX_SIZE = 64

    def __init__(self, vid=None, pid=None):
        self.led_status = LEDStatus()
        if vid is not None and pid is not None:
            self.__openDevice(vid, pid)
        else:
            ret, vid, pid = listHidDevice()
            if ret:
                self.__openDevice(vid, pid)
            else:
                raise ValueError("can not find DLP4500 device!")

    def __openDevice(self, vendor_id, product_id):
        self.hid_dev = hid.device()
        try:
            self.hid_dev.open(vendor_id, product_id)
            print("Manufacturer: %s" % self.hid_dev.get_manufacturer_string())
            print("Product: %s" % self.hid_dev.get_product_string())
            print("Serial No: %s" % self.hid_dev.get_serial_number_string())
            print("Write the data")
        except IOError as e:
            print(e)
            print("You probably don't have the hard coded device. Update the hid.device line")
            print("in this script with one from the enumeration list output above and try again.")

    def __DLPWrite(self, cmd):
        self.hid_dev.write(cmd)

    def __DLPRead(self):
        read_data = self.hid_dev.read(self.USB_READ_MAX_SIZE)
        return read_data

    def __preSetCmd(self, cmd1, cmd2):
        pre_command = self.CMD_HEAD["SET"].copy()
        pre_command = np.append(pre_command, cmd1)
        pre_command = np.append(pre_command, cmd2)
        return pre_command

    def __preGetCmd(self, cmd):
        pre_command = self.CMD_HEAD["GET"].copy()
        pre_command = np.append(pre_command, cmd)
        return pre_command

    def loadImageFromFlash(self, imageIndex):
        """
        从flash中读取指定编号的图片并投影
        @param imageIndex: flash中图片的索引值
        @return:
        """
        write_cmd = self.__preSetCmd(self.SET_CMD_FUNC_CODE["LOAD_IMAGE"], imageIndex)
        self.__DLPWrite(write_cmd)
        res = self.__DLPRead()

    def setLEDSelection(self, ledRed=1, ledGreen=0, ledBlue=0):
        """
        设置开启哪个LED灯
        @param ledRed: 0: 关闭, 1开启
        @param ledGreen: 0: 关闭, 1开启
        @param ledBlue: 0: 关闭, 1开启
        @return:
        """
        led_value = ledRed * 1 + ledGreen * 2 + ledBlue * 4
        write_cmd = self.__preSetCmd(self.SET_CMD_FUNC_CODE["LED_SELECT"], led_value)
        self.__DLPWrite(write_cmd)
        res = self.__DLPRead()

    def setLEDCurrentValue(self, redCurrent=40, greenCurrent=0, blueCurrent=0):
        """
        设置当前LED电流值
        @param redCurrent: 红灯电流值
        @param greenCurrent: 绿灯电流值
        @param blueCurrent: 蓝灯电流值
        @return:
        """
        current_3x1 = np.array([redCurrent, greenCurrent, blueCurrent])
        write_cmd = self.__preSetCmd(self.SET_CMD_FUNC_CODE["CURRENT_VALUE"], current_3x1)
        self.__DLPWrite(write_cmd)
        res = self.__DLPRead()

    def setLEDCurrentMode(self, currentMode=1):
        """
        设置LED电流值是否反转
        @param currentMode: 0: invert(255-value) 1: normal
        @return:
        """
        write_cmd = self.__preSetCmd(self.SET_CMD_FUNC_CODE["CURRENT_MODE"], currentMode)
        self.__DLPWrite(write_cmd)
        res = self.__DLPRead()

    def setOperationMode(self, operationMode=1):
        """
        设置DLP的工作模式
        @param operationMode: 1: standby 0: video
        @return:
        """
        write_cmd = self.__preSetCmd(self.SET_CMD_FUNC_CODE["OPERATION_MODE"], operationMode)
        self.__DLPWrite(write_cmd)
        res = self.__DLPRead()

    def getLEDEnable(self):
        """
        获取当前所用的LED颜色
        @return: 三个数值, 分别表示R G B, 1代表开启 ,0代表未开启
        """
        write_cmd = self.__preGetCmd(self.GET_CMD_FUNC_CODE["LED_ENABLE"])
        self.__DLPWrite(write_cmd)
        recv = self.__DLPRead()
        # print("getLEDEnable", recv)
        led_enable_val = recv[4]
        return led_enable_val & 0x01, led_enable_val >> 1 & 0x01, led_enable_val >> 2 & 0x01

    def getLEDCurrent(self):
        """
        获取当前DLP三个灯电流大小
        @return: 三个数值,顺序为RGB, 其值为0-255
        """
        write_cmd = self.__preGetCmd(self.GET_CMD_FUNC_CODE["CURRENT_VAL"])
        self.__DLPWrite(write_cmd)
        recv = self.__DLPRead()
        # print("getLEDCurrent", recv)
        return recv[4], recv[5], recv[6]

    def getLEDCurrentInvert(self):
        """
        当前电流是否翻转
        @return: 0: 为翻转, 1:反转
        """
        write_cmd = self.__preGetCmd(self.GET_CMD_FUNC_CODE["LEDPWM_INVERT"])
        self.__DLPWrite(write_cmd)
        recv = self.__DLPRead()
        # print("getLEDCurrentInvert", recv)
        return recv[4]

    def getLEDStatus(self):
        """
        获取当前LED状态, 包括开启LED类型,电流大小,电流类型
        @return: 数据保存在一个结构体里面
        my_test_api = DLP4500API()
        current_led_state = my_test_api.getLEDStatue()
        print("RED LED SELECT: ", current_led_state.LED_ENABLE.R_EN)
        print("GREEN LED SELECT: ", current_led_state.LED_ENABLE.G_EN)
        print("BLUE LED SELECT: ", current_led_state.LED_ENABLE.B_EN)

        print("RED LED CURRENT VALUE: ", current_led_state.LED_CURRENT.R_VALUE)
        print("GREEN LED CURRENT VALUE: ", current_led_state.LED_CURRENT.G_VALUE)
        print("BLUE LED CURRENT VALUE: ", current_led_state.LED_CURRENT.B_VALUE)

        print("LED CURRENT MODE: ", current_led_state.INVERT)
        """
        r_en, g_en, b_en = self.getLEDEnable()
        r_val, g_val, b_val = self.getLEDCurrent()
        invert = self.getLEDCurrentInvert()
        self.led_status.LED_ENABLE.R_EN = r_en
        self.led_status.LED_ENABLE.G_EN = g_en
        self.led_status.LED_ENABLE.B_EN = b_en

        self.led_status.LED_CURRENT.R_VALUE = r_val
        self.led_status.LED_CURRENT.G_VALUE = g_val
        self.led_status.LED_CURRENT.B_VALUE = b_val

        self.led_status.INVERT = invert
        return self.led_status

    def close(self):
        self.hid_dev.close()

def main():
    import time
    my_test_api = DLP4500API()
    # exit()
    my_test_api.setOperationMode(1)
    time.sleep(1)
    my_test_api.setOperationMode(0)
    time.sleep(1)
    my_test_api.setLEDCurrentMode(0)
    time.sleep(1)
    my_test_api.setLEDCurrentValue()
    time.sleep(1)
    my_test_api.setLEDCurrentMode(1)
    time.sleep(1)
    my_test_api.setLEDCurrentValue()
    time.sleep(1)
    my_test_api.setLEDSelection()

    current_led_state = my_test_api.getLEDStatus()
    print("RED LED SELECT: ", current_led_state.LED_ENABLE.R_EN)
    print("GREEN LED SELECT: ", current_led_state.LED_ENABLE.G_EN)
    print("BLUE LED SELECT: ", current_led_state.LED_ENABLE.B_EN)

    print("RED LED CURRENT VALUE: ", current_led_state.LED_CURRENT.R_VALUE)
    print("GREEN LED CURRENT VALUE: ", current_led_state.LED_CURRENT.G_VALUE)
    print("BLUE LED CURRENT VALUE: ", current_led_state.LED_CURRENT.B_VALUE)

    print("LED CURRENT MODE: ", current_led_state.INVERT)

if __name__ == "__main__":
    main()
    # listHidDevice()