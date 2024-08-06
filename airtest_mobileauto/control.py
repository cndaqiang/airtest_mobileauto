#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################
# Author : cndaqiang             #
# Update : 2023-11-10            #
# Build  : 2023-11-10            #
# What   : IOS/Android 自动化任务  #
#################################
# .......
from datetime import datetime, timezone, timedelta
import time
from airtest.core.settings import Settings as ST
import logging
import sys
import os
import numpy as np
import random
import traceback
import subprocess
import shlex
import configparser
import multiprocessing
# 重写函数#
from airtest.core.api import connect_device, sleep
from airtest.core.api import exists as exists_o
from airtest.core.api import touch as touch_o
from airtest.core.api import swipe as swipe_o
from airtest.core.api import start_app as start_app_o
from airtest.core.api import stop_app as stop_app_o
from airtest.core.api import Template as Template_o

# ........................
# python -m pip install --upgrade --no-deps --force-reinstall airtest
# vscode设置image preview的解析目录为assets,就可以预览了
ST.OPDELAY = 1
# 全局阈值的范围为[0, 1]
ST.THRESHOLD_STRICT = 0.8
ST.THRESHOLD = 0.8  # 其他语句的默认阈值
# ST.FIND_TIMEOUT=10 #*2 #获取截图的时间限制
# ST.FIND_TIMEOUT_TMP=1#匹配图形的时间限制, 也许可以再改小些加速

# 控制屏幕输出
# 这个设置可以极低的降低airtest输出到屏幕的信息
logger = logging.getLogger("airtest")
logger.setLevel(logging.WARNING)
#


class Settings(object):
    #
    # 特色修改
    # python解释器是AirtestIDE还是终端的python
    AirtestIDE = "AirtestIDE" in sys.executable
    start_app_syskeys = False
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    testpng = Template_o(os.path.join(current_dir, "tpl_target_pos.png"), record_pos=(-0.28, 0.153), resolution=(960, 540), target_pos=6)
    #
    # 控制端
    platform = sys.platform.lower()
    # 避免和windows名字接近
    platform = "macos" if "darwin" in platform else platform
    #
    # control, 运行控制
    prefix = ""
    figdir = "assets"
    # 时间参数
    # 防止服务器时区不同, 影响对游戏的执行时间判断
    # 设置为8则为东八区
    mobiletime = 8
    eastern_eight_offset = timedelta(hours=mobiletime)
    eastern_eight_tz = timezone(eastern_eight_offset)
    # 日志参数
    logger_dict = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    logger_level = 1  # 设置为0输出详细模式
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("airtest_mobileauto")
    logger.setLevel(logging.DEBUG)
    outputnode = -1
    #

    # client, 客户端情况
    mynode = 0
    totalnode = 1
    multiprocessing = False
    # 客户端
    # 虚拟机,android docker, iphone, etc,主要进行设备的连接和重启
    dockercontain = {}  # "androidcontain"
    BlueStackdir = ""  # "C:\Program Files\BlueStacks_nxt"
    LDPlayerdir = ""  # "D:\GreenSoft\LDPlayer"
    # ADB地址
    LINK_dict = {0: "Android:///127.0.0.1:5555"}
    #

    @classmethod
    def Config(cls, config_file="config.in"):
        if not os.path.exists(config_file):
            return
        config = configparser.ConfigParser()
        with open(config_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
        #
        # control
        cls.prefix = config.get('control', 'prefix', fallback=cls.prefix)
        cls.figdir = config.get('control', 'figdir', fallback=cls.figdir)
        #
        mobiletime = config.getint('control', 'mobiletime', fallback=cls.mobiletime)
        eastern_eight_offset = timedelta(hours=mobiletime)
        cls.eastern_eight_tz = timezone(eastern_eight_offset)
        #
        cls.logger_level = config.getint('control', 'logger_level', fallback=cls.logger_level)
        level = Settings.logger_dict[Settings.logger_level]
        logging.basicConfig(level=level, format='%(message)s')
        logger = logging.getLogger("airtest_mobileauto")
        logger.setLevel(level)
        cls.outputnode = config.getint('control', 'outputnode', fallback=cls.outputnode)
        #
        # client
        cls.mynode = config.getint('client', 'mynode', fallback=cls.mynode)
        cls.totalnode = config.getint('client', 'totalnode', fallback=cls.totalnode)
        cls.multiprocessing = config.getboolean('client', 'multiprocessing', fallback=cls.multiprocessing) and cls.totalnode > 1
        #
        dockercontain_str = config.get('client', 'dockercontain', fallback=str(cls.dockercontain), raw=True)
        cls.dockercontain = eval(dockercontain_str)
        #
        cls.BlueStackdir = config.get('client', 'BlueStackdir', fallback=cls.BlueStackdir)
        cls.LDPlayerdir = config.get('client', 'LDPlayerdir', fallback=cls.LDPlayerdir)
        emulator = ""
        if len(cls.BlueStackdir) > 0:
            cls.win_Instance = {0: "Nougat32"}
            cls.win_InstanceName = {0: "BlueStacks App Player"}
            for i in range(1, 5):
                cls.win_Instance[i] = f"{cls.win_Instance[0]}_{i}"
                cls.win_InstanceName[i] = f"{cls.win_InstanceName[0]} {i}"
            emulator = 'BlueStack'
        elif len(cls.LDPlayerdir) > 0:
            cls.win_Instance = {0: "index=0"}
            cls.win_InstanceName = {0: "雷电模拟器"}
            for i in range(1, 5):
                cls.win_Instance[i] = f"index={i}"
                cls.win_InstanceName[i] = f"{cls.win_InstanceName[0]}-{i}"
            emulator = 'LDPlayer'
        if len(emulator) > 0:
            Instance_str = config.get('client', emulator+'_Instance', fallback=str(cls.win_Instance), raw=True)
            cls.win_Instance = eval(Instance_str)
            Windows_str = config.get('client', emulator+'_Windows', fallback=str(cls.win_InstanceName), raw=True)
            cls.win_InstanceName = eval(Windows_str)
        #
        # 读取LINK_dict，假设配置文件中存储的是字符串形式的字典
        # 本地docker容器
        cls.LINK_dict[0] = "Android:///127.0.0.1:5555"
        cls.LINK_dict[1] = "Android:///127.0.0.1:5565"
        cls.LINK_dict[2] = "Android:///127.0.0.1:5575"
        cls.LINK_dict[3] = "Android:///127.0.0.1:5585"
        cls.LINK_dict[4] = "Android:///127.0.0.1:5595"
        cls.LINK_dict[5] = "Android:///192.168.192.10:5555"  # 服务器上的docker容器
        cls.LINK_dict[6] = "Android:///192.168.192.10:5565"  # 服务器上的docker容器
        cls.LINK_dict[7] = "ios:///http://127.0.0.1:8200"  # Iphone SE映射到本地
        cls.LINK_dict[8] = "ios:///http://169.254.83.56:8100"  # Iphone 11支持无线连接
        cls.LINK_dict[9] = "Android:///emulator-5554"  # 本地的安卓模拟器
        cls.LINK_dict[10] = "Android:///4e86ac13"  # usb连接的安卓手机
        link_dict_str = config.get('client', 'LINK_dict', fallback=str(cls.LINK_dict), raw=True)
        cls.LINK_dict = eval(link_dict_str)

    #

    @classmethod
    def info(cls, prefix=""):
        TimeDebug(prefix+":mynode="+str(cls.mynode))
        TimeDebug(prefix+":totalnode="+str(cls.totalnode))
        TimeDebug(prefix+":LINK_dict="+str(cls.LINK_dict))


# 替代基础的print函数

def loggerhead():
    # 获取当前日期和时间
    current_datetime = datetime.now(Settings.eastern_eight_tz)
    # 格式化为字符串（月、日、小时、分钟、秒）
    formatted_string = current_datetime.strftime("[%m-%d %H:%M:%S]")
    return formatted_string+f"({Settings.mynode})"


def TimeECHO(info, *args, **kwargs):
    if Settings.outputnode >= 0 and Settings.outputnode != Settings.mynode:
        return
    modified_args = loggerhead()+info
    Settings.logger.info(modified_args)
    return


def TimeErr(info, *args, **kwargs):
    if Settings.outputnode >= 0 and Settings.outputnode != Settings.mynode:
        return
    Settings.logger.warning(loggerhead()+info, *args, **kwargs)


def TimeDebug(info, *args, **kwargs):
    if Settings.outputnode >= 0 and Settings.outputnode != Settings.mynode:
        return
    Settings.logger.debug(loggerhead()+info, *args, **kwargs)


def fun_name(level=1):
    """
    def b()
        fun_name(1) == "b"
    """
    import inspect
    fun = inspect.currentframe()
    ilevel = 0
    for i in range(level):
        try:
            fun = fun.f_back
            ilevel = ilevel+1
        except:
            break
    try:
        return str(fun.f_code.co_name)
    except:
        return ""


def funs_name(level=2):
    i = level
    content = fun_name(i)
    while i < 10:
        i = i+1
        i_name = fun_name(i)
        if len(i_name) > 0:
            content = content + "." + i_name
        else:
            break
    return content


# 如果命令需要等待打开的程序关闭, 这个命令很容易卡住
def getstatusoutput(*args, **kwargs):
    try:
        return subprocess.getstatusoutput(*args, **kwargs)
    except:
        return [1, traceback.format_exc()]


def run_command(command=[], sleeptime=20,  quiet=False, must_ok=False):
    """
     执行命令
     统一采用subprocess.Popen(["exec","para","para2","..."])
    """
    exit_code_o = 0
    command_step = 0
    # 获得运行的结果
    if not quiet:
        TimeECHO(funs_name())
    for i_command in command:
        if len(i_command) < 1:
            continue
        # 去掉所有的空白符号看是否还有剩余命令
        trim_insert = shlex.join(i_command).strip()
        if len(trim_insert) < 1:
            continue
        if not quiet:
            TimeECHO("  sysrun:"+trim_insert)
        #
        try:
            # os.system的容易卡，各种命令兼容性也不好，subprocess.Popen可以直接填windows快捷方式里的内容
            # shell用于支持$(cat )等命令, 并且只能用一个字符串
            if len(i_command) == 1:
                process = subprocess.Popen(i_command[0], shell=True)
            else:
                process = subprocess.Popen(i_command)
            result = [0, str(process)]
            # 运行成功的结果会直接输出的
        except:
            result = [1, traceback.format_exc()]
        command_step = command_step + 1
        exit_code = result[0]
        if not quiet:
            if exit_code != 0:
                TimeECHO("result:"+">"*20)
                TimeECHO(result[1])
                TimeECHO("result:"+"<"*20)
        exit_code_o += exit_code
        if must_ok and exit_code_o != 0:
            break
        sleep(sleeptime)
    # 没有执行任何命令
    if command_step == 0:
        exit_code_o = -100
    return exit_code_o


def run_class_command(self=None, command=[], quiet=False, must_ok=False):
    """
 # 执行模块内的文件
 # 以为文件中的命令可能包含self,所以把self作为输入参数
    """
    # 获得运行的结果
    exit_code_o = 0
    command_step = 0
    if not quiet:
        TimeECHO(funs_name())
    for i_command in command:
        # 去掉所有的空白符号看是否还有剩余命令
        trim_insert = i_command.strip()
        if len(trim_insert) < 1:
            continue
        if '#' == trim_insert[0]:
            continue
        if not quiet:
            TimeECHO("  python: "+i_command.rstrip())
        try:
            exec(i_command)
            exit_code = 0
            command_step = command_step + 1
        except:
            traceback.print_exc()
            exit_code = 1
        exit_code_o += exit_code
        if must_ok and exit_code_o != 0:
            break
    # 没有执行任何命令
    if command_step == 0:
        exit_code_o = -100
    return exit_code_o


def getpid_win(IMAGENAME="HD-Player.exe", key="BlueStacks App Player 0"):
    if sys.platform.lower() != "win32":
        return 0
    try:
        command = ["tasklist", "-FI", f"IMAGENAME eq {IMAGENAME}", "/V"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, _ = process.communicate()
        # 中文的windows系统默认返回gbk的编码
        # 尝试使用不同编码解码输出
        encodings = ['gbk', 'utf-8']  # 可以添加其他编码
        decoded_output = None
        for encoding in encodings:
            try:
                decoded_output = output.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not decoded_output:
            TimeECHO(f"getpid_win({IMAGENAME}) error"+":无法解析输出")
            return 0
        cont = decoded_output.splitlines()
        cont = output.decode('gbk', errors='ignore').splitlines()
    except:
        TimeECHO(f"getpid_win({IMAGENAME}) error"+"-"*10)
        traceback.print_exc()
        TimeECHO(f"getpid_win({IMAGENAME}) error"+"-"*10)
        return 0
    PID = 0
    for task in cont:
        taskterm = task.split()
        if len(taskterm) < 3:
            continue
        # IMAGENAME如果太长了会显示不全，因此不能直接IMAGENAME in task
        lenname = len(taskterm[0])
        if lenname == 0:
            continue
        if lenname < len(IMAGENAME):
            if not taskterm[0] == IMAGENAME[:lenname]:
                continue
        # key还是可以显示全的
        if key in task:
            PID = task.split()[1]
            try:
                TimeECHO(f"getpid_win:{task}")
                PID = int(PID)
            except:
                TimeECHO(f"getpid_win({IMAGENAME},{key}) error"+"-"*10)
                traceback.print_exc()
                TimeECHO(f"getpid_win({IMAGENAME},{key}) error"+"-"*10)
                PID = 0
            break
    return PID


def connect_status(times=10):
    for i in np.arange(times):
        try:
            exists_o(Settings.testpng)
            return True
        except:
            if i == times - 1:
                traceback.print_exc()
            TimeECHO(f"{funs_name()}无法连接设备,重试中{i}")
            sleep(1)
            continue
    TimeECHO(f"设备失去联系")
    return False
# ........................


def exists(*args, **kwargs):
    try:
        result = exists_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f" {fun_name(1)}  失败")
        if not connect_status():
            TimeErr(f"{fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = exists_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def touch(*args, **kwargs):
    try:
        result = touch_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f" {fun_name(1)}  失败")
        if not connect_status():
            TimeErr(f"{fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = touch_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def swipe(*args, **kwargs):
    result = False
    try:
        result = swipe_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f" {fun_name(1)}  失败")
        if not connect_status():
            TimeErr(f"{fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = swipe_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def start_app(*args, **kwargs):
    if Settings.start_app_syskeys:
        args_list = list(args)
        args_list[0] = str(args_list[0])+" --pct-syskeys 0"
        args = args_list
        TimeECHO(f"{fun_name(1)} with {args_list[0]}")
        Settings.start_app_syskeys = True
    try:
        result = True
        start_app_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{fun_name(1)} 失败")
        if not connect_status():
            TimeErr(f"{fun_name(1)}连接不上设备")
            return result
        sleep(1)
        # ......
        # 安卓系统的报错, 尝试进行修复
        errormessgae = traceback.format_exc()
        if "AdbError" in errormessgae or True:
            """
            使用start_app启动安卓软件的各种坑（有的安卓系统使用monkey需要添加参数，否则报错）
            方式1(monkey). start_app(package_name), 需要修改Airtest的代码添加`--pct-syskeys 0`(https://cndaqiang.github.io/2023/11/10/MobileAuto/)
            adb -s 127.0.0.1:5555 shell monkey -p com.tencent.tmgp.sgame
            方式2(am start). start_app(package_name, activity)
            获得Activity的方法`adb -s 127.0.0.1:5565 shell dumpsys package com.tencent.tmgp.sgame`有一个Activity Resolver Table
            Airtest代码中是 adb -s 127.0.0.1:5565  shell am start -n package_name/package_name.activity
            可并不是所有的app的启动都遵循这一原则,如
            "com.tencent.tmgp.sgame/SGameActivity",
            "com.tencent.gamehelper.smoba/com.tencent.gamehelper.biz.launcher.ui.SplashActivit
            所以如果相同方式2，还是要修改Airtest的代码，变为package_name/activity
            综合上述原因，还是采取方式1, 添加`--pct-syskeys 0`
            虽然start_app(self.APPID)也能启动, 但是要修改代码airtest/core/android/adb.py,
            即使用start_app(self.APPID,Activity)就不用修改代码了
            """
            args_list = list(args)
            if args_list and "SYS_KEYS has no physical keys but with factor" in errormessgae:
                args_list = list(args)
                args_list[0] = str(args_list[0])+" --pct-syskeys 0"
                args = args_list
                TimeECHO(f"{fun_name(1)} with {args_list[0]}")
                Settings.start_app_syskeys = True
            if "device offline" in errormessgae:
                TimeECHO("ADB device offline")
                return result
        # ......
        try:
            result = True
            start_app_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"再次尝试{fun_name(1)}仍失败，检测是否没有开启ADB,或者重新启动ADB")
            result = False
    return result


def stop_app(*args, **kwargs):
    try:
        result = True
        stop_app_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{fun_name(1)} 失败")
        if not connect_status():
            TimeErr(f"{fun_name(1)}连接不上设备")
            return result
        sleep(1)
        # 下面仍会输出信息，所以这里少报错，让屏幕更干净
        # traceback.print_exc()
        #
        try:
            result = True
            stop_app_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def Template(*args, **kwargs):
    # 在这里修改args和kwargs，例如针对kwargs中的key进行添加内容
    dirname = Settings.figdir
    if "dirname" in kwargs:
        dirname = kwargs["dirname"]
        del kwargs["dirname"]
    args_list = list(args)
    if args_list and "png" in args_list[0]:
        filename = os.path.join(dirname, args_list[0].lstrip('/'))
        if os.path.exists(filename):
            args_list[0] = os.path.join(dirname, args_list[0].lstrip('/'))
        else:
            TimeErr(f"不存在{filename}")
            filename = args_list[0]
            if not os.path.exists(filename):
                TimeErr(f"不存在{filename}")
        args = args_list
    # 调用Template_o函数，传入修改后的参数
    return Template_o(*args, **kwargs)


class DQWheel:
    def __init__(self, var_dict_file='var_dict_file.txt', mynode=-10, totalnode=-10, 容器优化=False):
        self.timedict = {}
        self.容器优化 = 容器优化
        self.辅助同步文件 = "NeedRebarrier.txt"
        self.mynode = mynode
        self.totalnode = totalnode
        self.totalnode_bak = totalnode
        #
        self.barrierlimit = 60*20  # 同步最大时长
        self.filelist = []  # 建立的所有文件，用于后期clear
        self.var_dict_file = var_dict_file
        self.var_dict = self.read_dict(self.var_dict_file)
        self.savepos = True
        # 子程序运行次数
        self.calltimes_dict = {}
        #
        self.stopnow = False
        self.stopfile = ".tmp.barrier.EXIT.txt"
        self.stopinfo = ""
        self.connecttimes = 0
        self.connecttimesMAX = 20
        self.独立同步文件 = f"{self.mynode}.{self.totalnode}.NeedRebarrier.txt"
        self.removefile(self.独立同步文件)

    def list_files(self, path):
        files = []
        with os.scandir(path) as entries:
            for entry in entries:
                files.append(entry.name)
        return files
    #

    def init_clean(self):
        # 不要删除这个文件,开局采用同步的方式进行统一删除,不然时间差会导致很多问题
        # self.removefile(self.辅助同步文件)
        # os.listdir(".")不显示隐藏文件
        for name in self.list_files("."):
            text = ".tmp.barrier."
            if text == name[:len(text)]:
                TimeECHO(f"清理旧文件:{name}")
                self.removefile(name)
    #

    def timelimit(self, timekey="", limit=0, init=True):
        if len(timekey) == 0:
            timekey = "none"
        if not timekey in self.timedict.keys():
            init = True
        if init:
            self.timedict[timekey] = time.time()
            return False
        else:
            if time.time()-self.timedict[timekey] > limit:
                self.timedict[timekey] = time.time()
                return True
            else:
                return False

    def removefile(self, filename):
        TimeECHO(f"remove[{filename}]")
        if os.path.exists(filename):
            try:
                os.remove(filename)
                TimeECHO("删除["+filename+"]成功")
            except:
                traceback.print_exc()
                TimeECHO("删除["+filename+"]失败")
                return False
            if os.path.exists(filename):
                TimeErr("["+filename+"]还存在")
                return False
            else:
                return True
        else:
            TimeECHO("不存在["+filename+"]")
            return False

    def removefiles(self, dir=".", head="", body="", foot=""):
        l_head = len(head)
        l_body = len(body)
        l_foot = len(foot)
        if l_head+l_body+l_foot == 0:
            return True
        for name in os.listdir(dir):
            isname = True
            if len(name) < max(l_head, l_body, l_foot):
                continue
            # 必须三个条件都满足才能删除
            if l_head > 0:
                if not head == name[:l_head]:
                    continue
            if l_body > 0:
                if not body in name:
                    continue
            if l_foot > 0:
                if not foot == name[-l_foot:]:
                    continue
            #
            if isname:
                self.removefile(os.path.join(dir, name))
        return True

    def touchfile(self, filename, content=""):
        TimeECHO(f"touchfile[{filename}]")
        content = str(content)
        if len(content) > 0:
            self.removefile(filename)
        f = open(filename, 'w', encoding='utf-8')
        f.write(content)
        f.close()
        end = ""
        if len(content) > 0:
            end = f"with ({content})"
        TimeECHO(f"创建[{filename}] {end} 成功")

    def touchstopfile(self, content="stop"):
        self.touchfile(self.stopfile, content=content)
        self.stopnow = True
        self.stopinfo = content

    def readstopfile(self):
        if os.path.exists(self.stopfile):
            self.stopinfo = self.readfile(self.stopfile)[0]
            self.stopnow = True
        else:
            self.stopnow = False
        return self.stopnow

    def readfile(self, filename):
        if not os.path.exists(filename):
            TimeECHO("不存在["+filename+"]")
            return [""]
        try:
            f = open(filename, 'r', encoding='utf-8')
            content = f.readlines()
            f.close()
            TimeECHO("Read["+filename+"]成功")
            return content
        except:
            traceback.print_exc()
            TimeECHO("Read["+filename+"]失败")
            return [""]

    #
    def touch同步文件(self, 同步文件="", content=""):
        if len(同步文件) > 1:
            同步文件 = 同步文件
        else:
            同步文件 = self.辅助同步文件 if self.totalnode_bak > 1 else self.独立同步文件
        if self.存在同步文件(同步文件):
            TimeECHO(f"不再创建[{同步文件}]")
            return True
        content = loggerhead()+funs_name() if len(content) == 0 else content
        TimeECHO(f"**** TOUCH **** 创建同步文件[{同步文件}]")
        self.touchfile(同步文件, content)
        return True

    def 存在同步文件(self, 同步文件=""):
        if len(同步文件) > 1:
            if os.path.exists(同步文件):
                TimeECHO(f"存在同步文件[{同步文件}]")
                return True
            else:
                return False
        # 只要是总结点数大于1,无论当前是否组队都判断辅助同步文件
        if self.totalnode_bak > 1 and os.path.exists(self.辅助同步文件):
            TimeECHO(f"存在辅助同步文件[{self.辅助同步文件}]")
            return True
        # 每个进程的独立文件不同,不同节点不会误判
        if os.path.exists(self.独立同步文件):
            TimeECHO(f"存在独立同步文件[{self.独立同步文件}]")
            return True
        return False

    def clean文件(self):
        for i in self.filelist:
            if os.path.exists(i):
                self.removefile(i)
        self.filelist = []
    #

    def barriernode(self, mynode, totalnode, name="barrierFile"):
        if totalnode < 2:
            return True
        if self.存在同步文件():
            TimeErr(f"同步{name}.检测到同步文件")
            return True
        filelist = []
        ionode = mynode == 0 or totalnode == 1
        #
        if ionode:
            TimeECHO(f"BARRIERNODE [{name}]")
        #
        for i in np.arange(1, totalnode):
            filename = f".tmp.barrier.{i}.{name}.txt"
            if ionode:
                if os.path.exists(filename):
                    TimeErr("完蛋,barriernode之前就存在同步文件")
                self.touchfile(filename)
            filelist.append(filename)
            self.filelist.append(filename)
        #
        self.timelimit(timekey=name, limit=self.barrierlimit, init=True)
        times = 0
        while not self.timelimit(timekey=name, limit=self.barrierlimit, init=False):
            times = times+1
            if self.存在同步文件():
                return True
            if ionode:
                barrieryes = True
                for i in filelist:
                    barrieryes = barrieryes and not os.path.exists(i)
                    if not barrieryes:
                        break
                if barrieryes:
                    TimeECHO("."*10)
                    TimeECHO(f"BARRIERNODE 同步完成[{name}]")
                    return True
                if times % 3 == 0:
                    TimeECHO(f"BARRIERNODE ...同步检测[{name}]")
            else:
                if self.removefile(filelist[mynode-1]):
                    return True
            sleep(10)
        if ionode:
            for i in filelist:
                self.removefile(i)
            # 不清除也没事,start时会自动清除
        TimeErr(f"barriernode>{name}<同步失败,创建同步文件")
        self.touch同步文件()
        return False
    # 读取变量
    # read_dict 不仅适合保存字典,而且适合任意的变量类型

    def read_dict(self, var_dict_file="position_dict.txt"):
        global 辅助
        # if 辅助: return {}
        import pickle
        var_dict = {}
        if os.path.exists(var_dict_file):
            TimeECHO("读取"+var_dict_file)
            with open(var_dict_file, 'rb') as f:
                var_dict = pickle.load(f)
        return var_dict
        # 保存变量
    # save_dict 不仅适合保存字典,而且适合任意的变量类型

    def save_dict(self, var_dict, var_dict_file="position_dict.txt"):
        global 辅助
        # if 辅助: return True
        import pickle
        f = open(var_dict_file, "wb")
        pickle.dump(var_dict, f)
        f.close()
    # bcastvar 不仅适合保存字典,而且适合任意的变量类型

    def bcastvar(self, mynode, totalnode, var, name="bcastvar"):
        if totalnode < 2:
            return var
        dict_file = ".tmp."+name+".txt"
        if mynode == 0:
            self.save_dict(var, dict_file)
        self.barriernode(mynode, totalnode, "bcastvar."+name)
        if self.存在同步文件():
            return var
        #
        var_new = self.read_dict(dict_file)
        #
        return var_new

    def uniq_Template_array(self, arr):
        if not arr:  # 如果输入的列表为空
            return []
        #
        seen = set()
        unique_elements = []
        for item in arr:
            if item.filepath not in seen:
                unique_elements.append(item)
                seen.add(item.filepath)
        return unique_elements

    def 存在任一张图(self, array, strinfo="", savepos=False):
        array = self.uniq_Template_array(array)
        判断元素集合 = array
        strinfo = strinfo if len(strinfo) > 0 else "图片"
        if strinfo in self.calltimes_dict.keys():
            self.calltimes_dict[strinfo] = self.calltimes_dict[strinfo]+1
        else:
            self.calltimes_dict[strinfo] = 1
        strinfo = f"第[{self.calltimes_dict[strinfo]}]次寻找{strinfo}"
        length = len(判断元素集合)
        for idx, i in enumerate(判断元素集合):
            TimeECHO(f"{strinfo}({idx+1}/{length}):{i}")
            pos = exists(i)
            if pos:
                TimeECHO(f"{strinfo}成功:{i}")
                # 交换元素位置
                判断元素集合[0], 判断元素集合[idx] = 判断元素集合[idx], 判断元素集合[0]
                if savepos:
                    self.var_dict[strinfo] = pos
                return True, 判断元素集合
        return False, 判断元素集合

    def existsTHENtouch(self, png=Settings.testpng, keystr="", savepos=False):
        savepos = savepos and len(keystr) > 0 and self.savepos
        #
        if self.connecttimes > self.connecttimesMAX:  # 大概率连接失败了,判断一下
            if connect_status(times=max(2, self.connecttimesMAX-self.connecttimes+10)):  # 出错后降低判断的次数
                self.connecttimes = 0
            else:
                self.connecttimes = self.connecttimes+1
                self.touch同步文件(self.独立同步文件)
                return False
        #
        if savepos:
            if keystr in self.var_dict.keys():
                touch(self.var_dict[keystr])
                TimeECHO("touch (saved) "+keystr)
                sleep(0.1)
                return True
        pos = exists(png)
        if pos:
            self.connecttimes = 0
            touch(pos)
            if len(keystr) > 0:
                TimeECHO("touch "+keystr)
            if savepos:
                self.var_dict[keystr] = pos
                self.save_dict(self.var_dict, self.var_dict_file)
            return True
        else:
            self.connecttimes = self.connecttimes+1
            if len(keystr) > 0:
                TimeECHO("NotFound "+keystr)
            return False

    #
    # touch的总时长timelimit s, 或者总循环次数<10
    def LoopTouch(self, png=Settings.testpng, keystr="", limit=0, loop=10, savepos=False):
        timekey = "LOOPTOUCH"+keystr+str(random.randint(1, 500))
        if limit + loop < 0.5:
            limit = 0
            loop = 1
        self.timelimit(timekey=timekey, limit=limit, init=True)
        runloop = 1
        while self.existsTHENtouch(png=png, keystr=keystr+f".{runloop}", savepos=savepos):
            if limit > 0:
                if self.timelimit(timekey=timekey, limit=limit, init=False):
                    TimeErr("TOUCH"+keystr+"超时.....")
                    break
            if runloop > loop:
                TimeErr("TOUCH"+keystr+"超LOOP.....")
                break
            sleep(10)
            runloop = runloop+1
        #
        if exists(png):
            TimeErr(keystr+"图片仍存在")
            return True
        else:
            return False
    # 这仅针对辅助模式,因此同步文件取self.辅助同步文件

    def 必须同步等待成功(self, mynode, totalnode, 同步文件="", sleeptime=60*5):
        同步文件 = 同步文件 if len(同步文件) > 1 else self.辅助同步文件
        if totalnode < 2:
            self.removefile(同步文件)
            return True
        if self.存在同步文件(同步文件):  # 单进程各种原因出错时,多进程无法同步时
            if self.readstopfile():
                return
            TimeECHO(f"---{mynode}---"*5)
            TimeECHO(f"存在同步文件({同步文件}),第一次尝试同步程序")
            start_timestamp = int(time.time())
            # 第一次尝试同步
            self.同步等待(mynode, totalnode, 同步文件, sleeptime)
            # 如果还存在说明同步等待失败,那么改成hh:waitminu*N时刻进行同步
            while self.存在同步文件(同步文件):
                if self.readstopfile():
                    return
                waitminu = int(min(59, 5*totalnode))
                TimeErr(f"仍然存在同步文件,进行{waitminu}分钟一次的循环")
                hour, minu, sec = self.time_getHMS()
                minu = minu % waitminu
                if minu > totalnode:
                    sleepsec = (waitminu-minu)*60-sec
                    TimeECHO(f"等待{sleepsec}s")
                    sleep(sleepsec)
                    continue
                end_timestamp = int(time.time())
                sleepNtime = max(10, sleeptime-(end_timestamp-start_timestamp))+mynode*5
                self.同步等待(mynode, totalnode, 同步文件, sleepNtime)
            TimeECHO(f"+++{mynode}+++"*5)
        else:
            return True
        return not self.存在同步文件(同步文件)

    # 这仅针对辅助模式,因此同步文件取self.辅助同步文件
    def 同步等待(self, mynode, totalnode, 同步文件="", sleeptime=60*5):
        # 同步等待是为了处理,程序因为各种原因无法同步,程序出粗.
        # 重新校验各个进程
        # Step1. 检测到主文件{同步文件} 进入同步状态
        # Step2. 确定所有进程均检测到主文件状态
        # Step3. 检测其余进程是否都结束休息状态
        # 一个节点、一个节点的check
        #
        同步文件 = 同步文件 if len(同步文件) > 1 else self.辅助同步文件
        ionode = mynode == 0 or totalnode == 1
        if totalnode < 2:
            self.removefile(同步文件)
            return True
        if not os.path.exists(同步文件):
            return True
        #
        TimeECHO("进入同步等待")
        同步成功 = True
        name = 同步文件
        全部通信成功文件 = 同步文件+".同步完成.txt"
        全部通信失败文件 = 同步文件+".同步失败.txt"
        self.filelist.append(全部通信成功文件)
        # 前两个节点首先进行判定,因此先进行删除
        if mynode < 2:
            self.removefile(全部通信失败文件)
        # 最后一个通过才会删除成功文件,避免残留文件干扰
        self.removefile(全部通信成功文件)
        for i in np.arange(1, totalnode):
            if mynode > 0 and mynode != i:
                continue
            TimeECHO(f"进行同步循环{i}")
            sleep(mynode*5)
            #
            主辅通信成功 = False
            filename = f".tmp.barrier.{i}.{name}.in.txt"
            if ionode:
                hour, minu, sec = self.time_getHMS()
                myrandom = f"{i}{totalnode}{hour}{minu}{sec}"
                self.touchfile(filename, content=myrandom)
                lockfile = f".tmp.barrier.{myrandom}.{i}.{name}.in.txt"
                self.touchfile(lockfile)
                sleep(5)
                self.filelist.append(filename)
                self.filelist.append(lockfile)
                # 开始通信循环
                主辅通信成功 = False
                for sleeploop in np.arange(60*5):
                    if not os.path.exists(lockfile):
                        主辅通信成功 = True
                        self.removefile(filename)
                        break
                    sleep(1)
                # 判断通信成功与否
                同步成功 = 同步成功 and 主辅通信成功
                if 同步成功:
                    TimeECHO(f"同步{i}成功")
                else:
                    TimeECHO(f"同步{i}失败")
                    self.touchfile(全部通信失败文件)
                    break
                continue
            else:
                同步成功 = False
                # 辅助节点,找到特定,就循环5分钟
                myrandom = ""
                lockfile = f".tmp.barrier.{myrandom}.{i}.{name}.in.txt"
                TimeECHO(f"进行同步判定{i}")
                sleeploop = 0
                for sleeploop in np.arange(60*5*(totalnode-1)):
                    # 主辅通信循环
                    if os.path.exists(filename) and not 主辅通信成功:
                        myrandom = self.readfile(filename)[0].strip()
                    if len(myrandom) > 0 and not 主辅通信成功:
                        lockfile = f".tmp.barrier.{myrandom}.{i}.{name}.in.txt"
                        TimeECHO(f"同步文件更新 lockfile={lockfile}")
                        sleep(10)
                        主辅通信成功 = self.removefile(lockfile)
                    #
                    # 本节点通信成功，开始等待其他节点
                    if 主辅通信成功:
                        hour, minu, sec = self.time_getHMS()
                        if sleeploop % 10 == 0:
                            TimeECHO(f"本节点通信成功{sleeploop}，正在寻找>{全部通信成功文件}<")
                        if os.path.exists(全部通信成功文件):
                            TimeECHO(f"检测到全部通信成功文件{全部通信成功文件}")
                            同步成功 = True
                            break
                    if os.path.exists(全部通信失败文件):
                        TimeErr(f"监测到全部通信失败文件{全部通信失败文件}")
                        同步成功 = False
                        break
                    sleep(1)
        # 到此处完成
        # 因为是逐一进行同步的,所以全部通信成功文件只能由最后一个node负责删除
        if not 同步成功:
            self.touch同步文件(全部通信失败文件)
        同步成功 = 同步成功 and not os.path.exists(全部通信失败文件)
        if 同步成功:
            TimeECHO("同步等待成功")
            file_sleeptime = ".tmp.barrier.sleeptime.txt"
            if ionode:
                TimeECHO(f"存储sleeptime到[{file_sleeptime}]")
                self.touchfile(filename=file_sleeptime, content=str(sleeptime))
                TimeECHO("开始删建文件")
                self.clean文件()
                self.touchfile(全部通信成功文件)
                self.removefile(同步文件)
                self.removefile(全部通信失败文件)
            else:
                TimeECHO("开始读取sleeptime")
                sleeptime_read = self.readfile(file_sleeptime)[0].strip()
                if len(sleeptime_read) > 0:
                    sleeptime = int(sleeptime_read)
        else:
            TimeErr("同步等待失败")
            return False

        #
        self.barriernode(mynode, totalnode, "同步等待结束")
        TimeECHO(f"需要sleep{sleeptime}")
        sleep(sleeptime)
        return not os.path.exists(同步文件)

    def time_getHM(self):
        current_time = datetime.now(Settings.eastern_eight_tz)
        hour = current_time.hour
        minu = current_time.minute
        return hour, minu

    def time_getHMS(self):
        current_time = datetime.now(Settings.eastern_eight_tz)
        hour = current_time.hour
        minu = current_time.minute
        sec = current_time.second
        return hour, minu, sec

    def time_getYHMS(self):
        current_time = datetime.now(Settings.eastern_eight_tz)
        year = current_time.hour
        hour = current_time.hour
        minu = current_time.minute
        sec = current_time.second
        return year, hour, minu, sec

    def time_getweek(self):
        return datetime.now(Settings.eastern_eight_tz).weekday()
    # return 0 - 6

    def hour_in_span(self, startclock=0, endclock=24, hour=None):
        if not hour:
            hour, minu, sec = self.time_getHMS()
            hour = hour + minu/60.0+sec/60.0/60.0
        startclock = (startclock+24) % 24
        endclock = (endclock+24) % 24

        # 不跨越午夜的情况[6,23]
        if startclock <= endclock:
            left = 0 if startclock <= hour <= endclock else self.left_hour(startclock, hour)
        # 跨越午夜的情况[23,6], 即[6,23]不对战
        else:
            left = self.left_hour(startclock, hour) if endclock < hour < startclock else 0
        return left

    def left_hour(self, endtime=24, hour=None):
        if not hour:
            hour, minu, sec = self.time_getHMS()
            hour = hour + minu/60.0+sec/60.0/60.0
        left = (endtime+24-hour) % 24
        return left

    def stoptask(self):
        TimeErr(f"停止Airtest控制,停止信息"+self.stopinfo)
        return
        # 该命令无法结束,直接return吧
        # sys.exit()

    # 旧脚本,适合几个程序,自动商量node编号

    def autonode(self, totalnode):
        if totalnode < 2:
            return 0
        node = -10
        PID = os.getpid()
        filename = "init_node."+str(totalnode)+"."+str(PID)+".txt"
        self.touchfile(filename)
        TimeECHO("自动生成node中:"+filename)
        PID_dict = {}
        for i in np.arange(60):
            for name in os.listdir("."):
                if "init_node."+str(totalnode)+"." in name:
                    PID_dict[name] = name
            if len(PID_dict) == totalnode:
                break
            sleep(5)
        if len(PID_dict) != totalnode:
            self.removefile(filename)
            TimeECHO("文件数目不匹配")
            return node
        #
        strname = np.array(list(PID_dict.keys()))
        PIDarr = np.zeros(strname.size)
        for i in np.arange(PIDarr.size):
            PIDarr[i] = int(strname[i].split(".")[2])
        PIDarr = np.sort(PIDarr)
        for i in np.arange(PIDarr.size):
            TimeECHO("i="+str(i)+". PID="+str(PID)+". PIDarr[i]="+str(PIDarr[i]))
            if PID == PIDarr[i]:
                node = i

        if node < 0:
            TimeECHO("node < 0")
            self.removefile(filename)
            return node
        #
        TimeECHO("mynode:"+str(node))
        if self.barriernode(node, totalnode, "audfonode"):
            self.removefile(filename)
            return node


class deviceOB:
    def __init__(self, 设备类型=None, mynode=0, totalnode=1, LINK="Android:///"+"127.0.0.1:"+str(5555)):
        # 控制端
        self.控制端 = sys.platform.lower()
        # 避免和windows名字接近
        self.控制端 = "macos" if "darwin" in self.控制端 else self.控制端
        #
        # 客户端
        self.device = None
        #
        self.mynode = mynode
        self.totalnode = totalnode
        self.LINK = LINK
        self.LINKport = self.LINK.split(":")[-1]  # port
        # (USB连接时"Android:///id",没有端口
        self.LINKport = "" if "/" in self.LINKport else self.LINKport
        self.LINKtype = self.LINK.split(":")[0].lower()  # android, ios
        self.LINKhead = self.LINK[:-len(self.LINKport)-1] if len(self.LINKport) > 0 else self.LINK  # ios:///ip
        self.LINKURL = self.LINK.split("/")[-1]  # ip:port
        self.设备类型 = 设备类型.lower() if 设备类型 else self.LINKtype
        #
        self.adb_path = "adb"
        if "android" in self.设备类型:
            from airtest.core.android import adb
            self.ADB = adb.ADB()
            self.adb_path = self.ADB.adb_path
        # 不同客户端对重启的适配能力不同
        if "ios" in self.设备类型:
            self.客户端 = "ios"
        elif "win" in self.控制端:
            if os.path.exists(Settings.BlueStackdir) and self.mynode in Settings.win_Instance.keys():
                self.客户端 = "win_BlueStacks"
            elif os.path.exists(Settings.LDPlayerdir) and self.mynode in Settings.win_Instance.keys():
                self.客户端 = "win_LD"
            else:
                self.客户端 = "RemoteAndroid"
        elif "linux" in self.控制端 and self.mynode in Settings.dockercontain.keys():  # Linux + docker
            self.客户端 = "lin_docker"
        elif len(self.LINKport) > 0:  # 通过网络访问的安卓设备
            self.客户端 = "RemoteAndroid"
        else:
            self.客户端 = "USBAndroid"
        #
        self.实体终端 = False
        self.实体终端 = "mac" in self.控制端 or "ios" in self.设备类型
        self.容器优化 = "linux" in self.控制端 and "android" in self.设备类型
        #
        TimeECHO(f"控制端({self.控制端})")
        TimeECHO(f"客户端({self.客户端})")
        TimeECHO(f"ADB =({self.adb_path})")
        TimeECHO(f"LINK({self.LINK})")
        TimeECHO(f"LINKhead({self.LINKhead})")
        TimeECHO(f"LINKtype({self.LINKtype})")
        TimeECHO(f"LINKURL({self.LINKURL})")
        TimeECHO(f"LINKport({self.LINKport})")
        #
        self.连接设备()

    def 连接设备(self, times=1, timesMax=2):
        """
        # 尝试连接timesMax+1次,当前是times次
        """
        self.device = False
        TimeECHO(f"{self.LINK}:开始第{times}/{timesMax+1}次连接")
        try:
            self.device = connect_device(self.LINK)
            if self.device:
                TimeECHO(f"{self.LINK}:链接成功")
                return True
        except:
            if times == timesMax+1:
                traceback.print_exc()
            TimeErr(f"{self.LINK}:链接失败")
            if "ios" in self.设备类型:
                TimeECHO("重新插拔数据线")
        #
        if times <= timesMax:
            TimeECHO(f"{self.LINK}:链接失败,重启设备再次连接")
            self.启动设备()
            return self.连接设备(times+1, timesMax)
        else:
            TimeErr(f"{self.LINK}:链接失败,无法继续")
            return False

    def 启动设备(self):
        command = []
        TimeECHO(f"尝试启动设备中...")
        if self.客户端 == "ios":
            if "mac" in self.控制端:
                TimeECHO(f"测试本地IOS打开中")
            else:
                TimeECHO(f"当前模式无法打开IOS")
                return False
            # 获得运行的结果
            result = getstatusoutput("tidevice list")
            if 'ConnectionType.USB' in result[1]:
                # wdaproxy这个命令会同时调用xctest和relay，另外当wda退出时，会自动重新启动xctest
                # tidevice不支持企业签名的WDA
                self.LINKport = str(int(self.LINKport)+1)
                self.LINK = self.LINKhead+":"+self.LINKport
                # @todo, 此命令没有经过测试
                command.append([f"tidevice $(cat para.txt) wdaproxy -B  com.facebook.WebDriverAgentRunner.cndaqiang.xctrunner --port {self.LINKport} > tidevice.result.txt 2 > &1 &"])
                sleep(20)
            else:
                TimeErr(" tidevice list 无法找到IOS设备重启失败")
                return False
        # android
        elif self.客户端 == "win_BlueStacks":
            instance = Settings.win_Instance[self.mynode]
            command.append([os.path.join(Settings.BlueStackdir, "HD-Player.exe"), "--instance", instance])
        elif self.客户端 == "win_LD":
            instance = Settings.win_Instance[self.mynode]
            command.append([os.path.join(Settings.LDPlayerdir, "dnplayer.exe"), instance])
        elif self.客户端 == "FULL_ADB":
            # 通过reboot的方式可以实现重启和解决资源的效果
            command.append([self.adb_path, "connect", self.LINKURL])
            command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "lin_docker":
            containID = f"{Settings.dockercontain[self.mynode]}"
            command.append(["docker", "restart", containID])
        elif self.客户端 == "RemoteAndroid":
            # 热重启系统
            command.append([self.adb_path, "connect", self.LINKURL])
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "stop"])
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "start"])
        elif self.客户端 == "USBAndroid":
            result = getstatusoutput("adb devices")
            if self.LINKURL in result[1]:
                command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
            else:
                TimeECHO(f"没有找到USB设备{self.LINKURL}\n"+result[1])
                return False
        else:
            TimeECHO(f"未知设备类型")
            return False
        # 开始运行
        exit_code = run_command(command=command)
        if exit_code == 0:
            TimeECHO(f"启动成功")
            return True
        else:
            TimeErr(f"启动失败")
            return False

    def 关闭设备(self):
        command = []
        TimeECHO(f"尝试关闭设备中...")
        if self.客户端 == "ios":
            if "mac" in self.控制端:
                TimeECHO(f"测试本地IOS关闭中")
                command.append(["tidevice", "reboot"])
            else:
                TimeECHO(f"当前模式无法关闭IOS")
                return False
        # android
        elif self.客户端 == "win_BlueStacks":
            # 尝试获取PID
            PID = getpid_win(IMAGENAME="HD-Player.exe", key=Settings.win_InstanceName[self.mynode])
            # BlueStacks App Player 3
            if PID > 0:
                command.append(["taskkill", "/F", "/FI", f"PID eq {str(PID)}"])
            else:
                # 关闭所有虚拟机，暂时不采用
                # command.append(["taskkill", "/F", "/IM", "HD-Player.exe"])
                command = []
        elif self.客户端 == "win_LD":
            # 尝试获取PID
            PID = getpid_win(IMAGENAME="dnplayer.exe", key=Settings.win_InstanceName[self.mynode])
            if PID > 0:
                command.append(["taskkill", "/F", "/FI", f"PID eq {str(PID)}"])
            else:
                # LDPlayer支持adb reboot,👍
                command.append([self.adb_path, "connect", self.LINKURL])
                command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "FULL_ADB":
            # 通过reboot的方式可以实现重启和解决资源的效果
            command.append([self.adb_path, "connect", self.LINKURL])
            command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "lin_docker":
            containID = f"{Settings.dockercontain[self.mynode]}"
            command.append(["docker", "stop", containID])
        elif self.客户端 == "USBAndroid":
            result = getstatusoutput("adb devices")
            if self.LINKURL in result[1]:
                command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
            else:
                TimeECHO(f"没有找到USB设备{self.LINKURL}\n"+result[1])
                command = []
        # 保底的热重启系统
        if self.客户端 == "RemoteAndroid" or len(command) == 0:
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "stop"])
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "start"])
            command.append([self.adb_path, "disconnect", self.LINKURL])
        #
        # 开始运行
        exit_code = run_command(command=command, sleeptime=60)
        if exit_code == 0:
            TimeECHO(f"关闭成功")
            return True
        else:
            TimeECHO(f"关闭失败")
            return False

    def 重启设备(self, sleeptime=0):
        TimeECHO(f"重新启动({self.LINK})")
        self.关闭设备()
        sleeptime = max(10, sleeptime-60)
        printtime = max(30, sleeptime/10)
        TimeECHO("sleep %d min" % (sleeptime/60))
        for i in np.arange(int(sleeptime/printtime)):
            TimeECHO(f"...taskkill_sleep: {i}", end='\r')
            sleep(printtime)
        self.启动设备()
        self.连接设备()


class appOB:
    def __init__(self, APPID="", big=False, device=None):
        self.APPID = APPID
        self.Activity = None if "/" not in self.APPID else self.APPID.split("/")[1]
        self.APPID = self.APPID.split("/")[0]
        self.device = device
        self.big = big  # 是不是大型的程序, 容易卡顿，要多等待一会
    #

    def 打开APP(self):
        if self.Activity:
            TimeECHO(f"打开APP[{self.APPID}/{self.Activity}]中")
            启动成功 = start_app(self.APPID, self.Activity)
        else:
            TimeECHO(f"打开APP[{self.APPID}]中")
            启动成功 = start_app(self.APPID)
        if not 启动成功:
            TimeErr("打开失败,可能失联")
            return False
        else:
            sleep(20)
        return True

    def 重启APP(self, sleeptime=0):
        TimeECHO(f"重启APP中")
        self.关闭APP()
        sleep(10)
        sleeptime = max(10, sleeptime)  # 这里的单位是s
        printtime = max(30, sleeptime/10)
        if sleeptime > 60*60 and self.device:  # >1h
            self.device.重启设备(sleeptime)
        else:
            TimeECHO("sleep %d min" % (sleeptime/60))
            nstep = int(sleeptime/printtime)
            for i in np.arange(nstep):
                TimeECHO(f"...taskkill_sleep: {i}/{nstep}", end='\r')
                sleep(printtime)
        TimeECHO(f"打开程序")
        if self.打开APP():
            if self.big:
                TimeECHO(f"打开程序成功,sleep60*2")
                sleep(60*2)
            return True
        else:
            TimeECHO(f"打开程序失败")
            return False
    #

    def 关闭APP(self):
        TimeECHO(f"关闭APP[{self.APPID}]中")
        if not stop_app(self.APPID):
            TimeErr("关闭失败,可能失联")
            return False
        else:
            sleep(5)
            return True


class TaskManager:
    def __init__(self, config_file, task_class, method_name, *args, **kwargs):
        self.task_class = task_class
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.config_file = config_file
        # 初次读入参数，用于确定单/多进程执行
        Settings.Config(self.config_file)
        Settings.info("Task_init")

    def execute(self):
        TimeDebug("task execute"+str(Settings.multiprocessing))
        if Settings.multiprocessing:
            self.multi_execute()
        else:
            self.single_execute()

    def single_execute(self, mynode=None):
        # 子进程重新配置参数
        Settings.Config(self.config_file)
        #
        if mynode != None:
            TimeDebug("input mynode=%s", mynode)
            Settings.mynode = mynode
        else:
            mynode = Settings.mynode
            TimeDebug("reD mynode=%s", mynode)
        Settings.info("single_exe")
        #
        TimeDebug("single_execute starting with mynode=%s", mynode)
        try:
            task_instance = self.task_class(*self.args, **self.kwargs)
            method = getattr(task_instance, self.method_name)
            method()
        except:
            TimeErr("Error in single_execute with mynode=%s", mynode)
            TimeDebug("Exception traceback: %s", traceback.format_exc())
        #
        TimeDebug("single_execute finished with mynode=%s", mynode)

    def multi_execute(self):
        TimeDebug("Multiprocessing with %s total nodes.", Settings.totalnode)
        m_process = Settings.totalnode
        with multiprocessing.Pool(processes=m_process) as pool:
            results = pool.map(self.single_execute, range(m_process))
            TimeDebug("Mapping started, waiting for results...")
            pool.close()
            pool.join()
