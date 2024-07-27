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
# 时间参数
# 防止服务器时区不同,设定时间为东八区
# 创建一个表示东八区时区的 timedelta 对象
eastern_eight_offset = timedelta(hours=8)
# 创建一个时区对象
eastern_eight_tz = timezone(eastern_eight_offset)
# ? 设置,虚拟机,android docker, iphone, etc,主要进行设备的连接和重启
BlueStackdir = "C:\Program Files\BlueStacks_nxt"
LDPlayerdir = "D:\GreenSoft\LDPlayer"

# 获取当前的运行信息, 有的客户端有bug
AirtestIDE = "AirtestIDE" in sys.executable


# 控制屏幕输出
# 这个设置可以极低的降低airtest输出到屏幕的信息
logger = logging.getLogger("airtest")
logger.setLevel(logging.WARNING)

# 替代基础的print函数


def TimeECHO(info="None", end=""):
    # 由于AirTest客户端的解释器不会输出print的命令
    if AirtestIDE:
        logger.warning(info)
        return
    # 获取当前日期和时间
    current_datetime = datetime.now(eastern_eight_tz)
    # 格式化为字符串（月、日、小时、分钟、秒）
    formatted_string = current_datetime.strftime("[%m-%d %H:%M:%S]")
    modified_args = formatted_string+info
    if len(end) > 0:
        print(modified_args, end=end)
    else:
        print(modified_args)


def TimeErr(info="None"):
    TimeECHO("NNNN:"+info)


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
        return f"not found fun_name({ilevel})"


def funs_name(level=2):
    i = level
    content = fun_name(i)
    while i < 10:
        i = i+1
        try:
            content = content+"."+fun_name(i)
        except:
            break
    return content


# 如果命令需要等待打开的程序关闭, 这个命令很容易卡住
def getstatusoutput(*args, **kwargs):
    try:
        return subprocess.getstatusoutput(*args, **kwargs)
    except:
        return [1, traceback.format_exc()]


def run_command(command=[], sleeptime=20,  prefix="", quiet=False, must_ok=False):
    """
     执行命令
     统一采用subprocess.Popen(["exec","para","para2","..."])
    """
    exit_code_o = 0
    command_step = 0
    # 获得运行的结果
    for i_command in command:
        if len(i_command) < 1:
            continue
        # 去掉所有的空白符号看是否还有剩余命令
        trim_insert = shlex.join(i_command).strip()
        if len(trim_insert) < 1:
            continue
        if not quiet:
            TimeECHO(prefix+"sysrun:"+trim_insert)
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
                TimeECHO(prefix+"result:"+">"*20)
                TimeECHO(result[1])
                TimeECHO(prefix+"result:"+"<"*20)
        exit_code_o += exit_code
        if must_ok and exit_code_o != 0:
            break
        sleep(sleeptime)
    # 没有执行任何命令
    if command_step == 0:
        exit_code_o = -100
    return exit_code_o


def run_class_command(self=None, command=[], prefix="", quiet=False, must_ok=False):
    """
 # 执行模块内的文件
 # 以为文件中的命令可能包含self,所以把self作为输入参数
    """
    # 获得运行的结果
    exit_code_o = 0
    command_step = 0
    for i_command in command:
        # 去掉所有的空白符号看是否还有剩余命令
        trim_insert = i_command.strip()
        if len(trim_insert) < 1:
            continue
        if '#' == trim_insert[0]:
            continue
        if not quiet:
            TimeECHO(prefix+'python: '+i_command.rstrip())
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
        # 使用 'utf-8' 编码解析输出
        cont = output.decode('utf-8', errors='ignore').splitlines()
        # cont = os.popen(f'tasklist -FI "IMAGENAME eq {IMAGENAME}" /V').readlines()
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


def connect_status(times=10, prefix=""):
    # png = Template_o(r"assets/tpl_target_pos.png", record_pos=(-0.28, 0.153), resolution=(960, 540), target_pos=6)
    # 同一个py文件, 只要在调用之前定义过了就可以
    png = Template(r"tpl_target_pos.png", record_pos=(-0.28, 0.153), resolution=(960, 540), target_pos=6)
    prefix = f"{prefix} [{fun_name(2)}][{fun_name(1)}]"
    #
    for i in np.arange(times):
        try:
            exists_o(png)
            return True
        except:
            if i == times - 1:
                traceback.print_exc()
            TimeECHO(f"{prefix}无法连接设备,重试中{i}")
            sleep(1)
            continue
    TimeECHO(f"{prefix}设备失去联系")
    return False
# ........................


def exists(*args, **kwargs):
    prefix = ""
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
        del kwargs["prefix"]
    try:
        result = exists_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{prefix}  {fun_name(1)}  失败")
        if not connect_status(prefix=prefix):
            TimeErr(f"{prefix} {fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = exists_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"{prefix} 再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def touch(*args, **kwargs):
    prefix = ""
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
        del kwargs["prefix"]
    try:
        result = touch_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{prefix}  {fun_name(1)}  失败")
        if not connect_status(prefix=prefix):
            TimeErr(f"{prefix} {fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = touch_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"{prefix} 再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def swipe(*args, **kwargs):
    prefix = ""
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
        del kwargs["prefix"]
    result = False
    try:
        result = swipe_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{prefix}  {fun_name(1)}  失败")
        if not connect_status(prefix=prefix):
            TimeErr(f"{prefix} {fun_name(1)}连接不上设备")
            return result
        sleep(1)
        try:
            result = swipe_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"{prefix} 再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def start_app(*args, **kwargs):
    prefix = ""
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
        del kwargs["prefix"]
    try:
        result = True
        start_app_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{prefix} {fun_name(1)} 失败")
        if not connect_status(prefix=prefix):
            TimeErr(f"{prefix} {fun_name(1)}连接不上设备")
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
                TimeECHO(prefix+f"{fun_name(1)} with {args_list[0]}")
            if "device offline" in errormessgae:
                TimeECHO(prefix+"ADB device offline")
                return result
        # ......
        try:
            result = True
            start_app_o(*args, **kwargs)
        except:
            traceback.print_exc()
            TimeECHO(f"{prefix} 再次尝试{fun_name(1)}仍失败，检测是否没有开启ADB,或者重新启动ADB")
            result = False
    return result


def stop_app(*args, **kwargs):
    prefix = ""
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
        del kwargs["prefix"]
    try:
        result = True
        stop_app_o(*args, **kwargs)
    except:
        result = False
        TimeECHO(f"{prefix} {fun_name(1)} 失败")
        if not connect_status(prefix=prefix):
            TimeErr(f"{prefix} {fun_name(1)}连接不上设备")
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
            TimeECHO(f"{prefix} 再次尝试{fun_name(1)}仍失败")
            result = False
    return result


def Template(*args, **kwargs):
    # 在这里修改args和kwargs，例如针对kwargs中的key进行添加内容
    dirname = "assets"
    if "dirname" in kwargs:
        dirname = kwargs["dirname"]
        del kwargs["dirname"]
    # 将args转换为列表以进行修改
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
    def __init__(self, var_dict_file='var_dict_file.txt', prefix="", mynode=-10, totalnode=-10, 容器优化=False):
        self.timedict = {}
        self.容器优化 = 容器优化
        self.辅助同步文件 = "NeedRebarrier.txt"
        self.mynode = mynode
        self.totalnode = totalnode
        self.totalnode_bak = totalnode
        self.prefix = (f"({mynode})" if mynode >= 0 else "")+prefix
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
        self.独立同步文件 = self.prefix+"NeedRebarrier.txt"
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
                TimeECHO(self.prefix+f"清理旧文件:{name}")
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
        TimeECHO(self.prefix+f"remove[{filename}]")
        if os.path.exists(filename):
            try:
                os.remove(filename)
                TimeECHO(self.prefix+"删除["+filename+"]成功")
            except:
                traceback.print_exc()
                TimeECHO(self.prefix+"删除["+filename+"]失败")
                return False
            if os.path.exists(filename):
                TimeErr(self.prefix+"["+filename+"]还存在")
                return False
            else:
                return True
        else:
            TimeECHO(self.prefix+"不存在["+filename+"]")
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
        TimeECHO(self.prefix+f"touchfile[{filename}]")
        content = str(content)
        if len(content) > 0:
            self.removefile(filename)
        f = open(filename, 'w', encoding='utf-8')
        f.write(content)
        f.close()
        end = ""
        if len(content) > 0:
            end = f"with ({content})"
        TimeECHO(self.prefix+f"创建[{filename}] {end} 成功")

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
            TimeECHO(self.prefix+"不存在["+filename+"]")
            return [""]
        try:
            f = open(filename, 'r', encoding='utf-8')
            content = f.readlines()
            f.close()
            TimeECHO(self.prefix+"Read["+filename+"]成功")
            return content
        except:
            traceback.print_exc()
            TimeECHO(self.prefix+"Read["+filename+"]失败")
            return [""]

    #
    def touch同步文件(self, 同步文件="", content=""):
        if len(同步文件) > 1:
            同步文件 = 同步文件
        else:
            同步文件 = self.辅助同步文件 if self.totalnode_bak > 1 else self.独立同步文件
        if self.存在同步文件(同步文件):
            TimeECHO(f"{self.prefix}不再创建[{同步文件}]")
            return True
        content = self.prefix+":"+funs_name() if len(content) == 0 else content
        TimeECHO(f">{self.prefix}"*10)
        TimeECHO(self.prefix+f"创建同步文件[{同步文件}]")
        self.touchfile(同步文件, content)
        TimeECHO(f"<{self.prefix}"*10)
        # 该文件不添加到列表,仅在成功同步后才删除
        # self.filelist.append(self.辅助同步文件)
        return True

    def 存在同步文件(self, 同步文件=""):
        if len(同步文件) > 1:
            if os.path.exists(同步文件):
                TimeECHO(self.prefix+f"存在同步文件[{同步文件}]")
                return True
            else:
                return False
        # 只要是总结点数大于1,无论当前是否组队都判断辅助同步文件
        if self.totalnode_bak > 1 and os.path.exists(self.辅助同步文件):
            TimeECHO(self.prefix+f"存在辅助同步文件[{self.辅助同步文件}]")
            return True
        # 每个进程的独立文件不同,不同节点不会误判
        if os.path.exists(self.独立同步文件):
            TimeECHO(self.prefix+f"存在独立同步文件[{self.独立同步文件}]")
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
            TimeErr(self.prefix+f"同步{name}.检测到同步文件")
            return True
        filelist = []
        ionode = mynode == 0 or totalnode == 1
        #
        if ionode:
            TimeECHO(self.prefix+"."*10)
            TimeECHO(self.prefix+f">>>>>同步开始>{name}")
        #
        for i in np.arange(1, totalnode):
            filename = f".tmp.barrier.{i}.{name}.txt"
            if ionode:
                if os.path.exists(filename):
                    TimeErr(self.prefix+"完蛋,barriernode之前就存在同步文件")
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
                    TimeECHO(self.prefix+"."*10)
                    TimeECHO(self.prefix+f"<<<<<同步完成>{name}")
                    return True
                if times % 3 == 0:
                    TimeECHO(self.prefix+f"同步{name}检测中")
            else:
                if self.removefile(filelist[mynode-1]):
                    return True
            sleep(10)
        if ionode:
            for i in filelist:
                self.removefile(i)
            # 不清除也没事,start时会自动清除
        TimeErr(self.prefix+f":barriernode>{name}<同步失败,创建同步文件")
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
            TimeECHO(self.prefix+"读取"+var_dict_file)
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
            TimeECHO(self.prefix+f"{strinfo}({idx+1}/{length}):{i}")
            pos = exists(i, prefix=self.prefix)
            if pos:
                TimeECHO(self.prefix+f"{strinfo}成功:{i}")
                # 交换元素位置
                判断元素集合[0], 判断元素集合[idx] = 判断元素集合[idx], 判断元素集合[0]
                if savepos:
                    self.var_dict[strinfo] = pos
                return True, 判断元素集合
        return False, 判断元素集合

    def existsTHENtouch(self, png=Template(r"tpl_target_pos.png"), keystr="", savepos=False):
        savepos = savepos and len(keystr) > 0 and self.savepos
        #
        if self.connecttimes > self.connecttimesMAX:  # 大概率连接失败了,判断一下
            if connect_status(times=max(2, self.connecttimesMAX-self.connecttimes+10), prefix=self.prefix):  # 出错后降低判断的次数
                self.connecttimes = 0
            else:
                self.connecttimes = self.connecttimes+1
                self.touch同步文件(self.独立同步文件)
                return False
        #
        if savepos:
            if keystr in self.var_dict.keys():
                touch(self.var_dict[keystr])
                TimeECHO(self.prefix+"touch (saved) "+keystr)
                sleep(0.1)
                return True
        pos = exists(png, prefix=self.prefix)
        if pos:
            self.connecttimes = 0
            touch(pos)
            if len(keystr) > 0:
                TimeECHO(self.prefix+"touch "+keystr)
            if savepos:
                self.var_dict[keystr] = pos
                self.save_dict(self.var_dict, self.var_dict_file)
            return True
        else:
            self.connecttimes = self.connecttimes+1
            if len(keystr) > 0:
                TimeECHO(self.prefix+"NotFound "+keystr)
            return False

    #
    # touch的总时长timelimit s, 或者总循环次数<10
    def LoopTouch(self, png=Template(r"tpl_target_pos.png"), keystr="", limit=0, loop=10, savepos=False):
        timekey = "LOOPTOUCH"+keystr+str(random.randint(1, 500))
        if limit + loop < 0.5:
            limit = 0
            loop = 1
        self.timelimit(timekey=timekey, limit=limit, init=True)
        runloop = 1
        while self.existsTHENtouch(png=png, keystr=keystr+f".{runloop}", savepos=savepos):
            if limit > 0:
                if self.timelimit(timekey=timekey, limit=limit, init=False):
                    TimeErr(self.prefix+"TOUCH"+keystr+"超时.....")
                    break
            if runloop > loop:
                TimeErr(self.prefix+"TOUCH"+keystr+"超LOOP.....")
                break
            sleep(10)
            runloop = runloop+1
        #
        if exists(png, prefix=self.prefix):
            TimeErr(self.prefix+keystr+"图片仍存在")
            return True
        else:
            return False
    # 这仅针对辅助模式,因此同步文件取self.辅助同步文件

    def 必须同步等待成功(self, mynode, totalnode, 同步文件="", sleeptime=60*5):
        同步文件 = 同步文件 if len(同步文件) > 1 else self.辅助同步文件+funs_name()+".txt"
        if totalnode < 2:
            self.removefile(同步文件)
            return True
        if self.存在同步文件(同步文件):  # 单进程各种原因出错时,多进程无法同步时
            if self.readstopfile():
                return
            TimeECHO(self.prefix+"-."*20)
            TimeECHO(self.prefix+f"存在同步文件({同步文件}),第一次尝试同步同步程序")
            start_timestamp = int(time.time())
            # 第一次尝试同步
            self.同步等待(mynode, totalnode, 同步文件, sleeptime)
            # 如果还存在说明同步等待失败,那么改成hh:waitminu*N时刻进行同步
            while self.存在同步文件(同步文件):
                if self.readstopfile():
                    return
                waitminu = int(min(59, 5*totalnode))
                TimeErr(self.prefix+f"仍然存在同步文件,进行{waitminu}分钟一次的循环")
                hour, minu, sec = self.time_getHMS()
                minu = minu % waitminu
                if minu > totalnode:
                    sleepsec = (waitminu-minu)*60-sec
                    TimeECHO(self.prefix+f"等待{sleepsec}s")
                    sleep(sleepsec)
                    continue
                end_timestamp = int(time.time())
                sleepNtime = max(10, sleeptime-(end_timestamp-start_timestamp))+mynode*5
                self.同步等待(mynode, totalnode, 同步文件, sleepNtime)
            TimeECHO(self.prefix+"-+"*20)
        else:
            return True
        return not self.存在同步文件(同步文件)

    # 这仅针对辅助模式,因此同步文件取self.辅助同步文件
    def 同步等待(self, mynode, totalnode, 同步文件="", sleeptime=60*5):
        同步文件 = 同步文件 if len(同步文件) > 1 else self.辅助同步文件
        if totalnode < 2:
            self.removefile(同步文件)
            return True
        ionode = mynode == 0 or totalnode == 1
        # 同步等待是为了处理,程序因为各种原因无法同步,程序出粗.
        # 重新校验各个进程
        # Step1. 检测到主文件{同步文件} 进入同步状态
        # Step2. 确定所有进程均检测到主文件状态
        # Step3. 检测其余进程是否都结束休息状态
        prefix = f"({mynode})"
        主辅节点通信完成 = False
        发送信标 = True
        # 一个节点、一个节点的check
        if not os.path.exists(同步文件):
            return True
        TimeECHO(self.prefix+":进入同步等待")
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
            TimeECHO(self.prefix+f":进行同步循环{i}")
            sleep(mynode*5)
            if not os.path.exists(同步文件):
                TimeECHO(self.prefix+f"不存在同步文件{同步文件},退出")
                return True
            if self.readstopfile():
                return
            #
            主辅通信成功 = False
            filename = f".tmp.barrier.{i}.{name}.in.txt"
            if ionode:
                hour, minu, sec = self.time_getHMS()
                # myrandom=str(random.randint(totalnode+100, 500))+f"{hour}{minu}{sec}"
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
                    if not os.path.exists(同步文件):
                        TimeECHO(self.prefix+f"不存在同步文件{同步文件},退出")
                        return True
                    if self.readstopfile():
                        return
                    if not os.path.exists(lockfile):
                        主辅通信成功 = True
                        self.removefile(filename)
                        break
                    sleep(1)
                # 判断通信成功与否
                同步成功 = 同步成功 and 主辅通信成功
                if 同步成功:
                    TimeECHO(prefix+f"同步{i}成功")
                else:
                    TimeECHO(prefix+f"同步{i}失败")
                    self.touchfile(全部通信失败文件)
                    return False
                continue
            else:
                同步成功 = False
                # 辅助节点,找到特定,就循环5分钟
                myrandom = "initial"
                myrandom_new = myrandom
                lockfile = f".tmp.barrier.{myrandom}.{i}.{name}.in.txt"
                TimeECHO(self.prefix+f":进行同步判定{i}")
                sleeploop = 0
                for sleeploop in np.arange(60*5*(totalnode-1)):
                    if not os.path.exists(同步文件):
                        TimeECHO(self.prefix+f"不存在同步文件{同步文件},退出")
                        return True
                    if self.readstopfile():
                        return
                    # 主辅通信循环
                    if os.path.exists(filename):
                        if sleeploop % 5 == 0:
                            myrandom_new = self.readfile(filename)[0].strip()
                    if len(myrandom_new) > 0 and myrandom_new != myrandom:
                        myrandom = myrandom_new
                        TimeECHO(prefix+f"同步文件更新myrandom={myrandom}")
                        lockfile = f".tmp.barrier.{myrandom}.{i}.{name}.in.txt"
                        sleep(10)
                        主辅通信成功 = self.removefile(lockfile)
                    if not 主辅通信成功 and myrandom != "initial":
                        TimeECHO(prefix+f"还存在{lockfile}")
                        主辅通信成功 = self.removefile(lockfile)
                    # 避免存在旧文件没有删除的情况,这里不断读取å
                    if 主辅通信成功:
                        hour, minu, sec = self.time_getHMS()
                        if sleeploop % 10 == 0:
                            TimeECHO(prefix+f"正在寻找全部通信成功文件>{全部通信成功文件}<")
                        if os.path.exists(全部通信成功文件):
                            TimeECHO(prefix+f"监测到全部通信成功文件{全部通信成功文件}")
                            同步成功 = True
                            break
                        if os.path.exists(全部通信失败文件):
                            TimeErr(prefix+f"监测到全部通信失败文件{全部通信失败文件}")
                            return False
                    sleep(1)
        # 到此处完成
        # 因为是逐一进行同步的,所以全部通信成功文件只能由最后一个node负责删除
        同步成功 = 同步成功 and not os.path.exists(全部通信失败文件)
        if 同步成功:
            TimeECHO(prefix+"同步等待成功")
            file_sleeptime = ".tmp.barrier.sleeptime.txt"
            if ionode:
                TimeECHO(prefix+f"存储sleeptime到[{file_sleeptime}]")
                self.touchfile(filename=file_sleeptime, content=str(sleeptime))
                TimeECHO(prefix+"开始删建文件")
                self.clean文件()
                self.touchfile(全部通信成功文件)
                self.removefile(同步文件)
                self.removefile(全部通信失败文件)
            else:
                TimeECHO(prefix+"开始读取sleeptime")
                sleeptime_read = self.readfile(file_sleeptime)[0].strip()
                if len(sleeptime_read) > 0:
                    sleeptime = int(sleeptime_read)
        else:
            TimeErr(prefix+"同步等待失败")
            return False

        #
        self.barriernode(mynode, totalnode, "同步等待结束")
        TimeECHO(self.prefix+f"需要sleep{sleeptime}")
        sleep(sleeptime)
        return not os.path.exists(同步文件)

    def time_getHM(self):
        current_time = datetime.now(eastern_eight_tz)
        hour = current_time.hour
        minu = current_time.minute
        return hour, minu

    def time_getHMS(self):
        current_time = datetime.now(eastern_eight_tz)
        hour = current_time.hour
        minu = current_time.minute
        sec = current_time.second
        return hour, minu, sec

    def time_getYHMS(self):
        current_time = datetime.now(eastern_eight_tz)
        year = current_time.hour
        hour = current_time.hour
        minu = current_time.minute
        sec = current_time.second
        return year, hour, minu, sec

    def time_getweek(self):
        return datetime.now(eastern_eight_tz).weekday()
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
        TimeErr(self.prefix+f"停止Airtest控制,停止信息"+self.stopinfo)
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
        TimeECHO(self.prefix+"自动生成node中:"+filename)
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
            TimeECHO(self.prefix+"文件数目不匹配")
            return node
        #
        strname = np.array(list(PID_dict.keys()))
        PIDarr = np.zeros(strname.size)
        for i in np.arange(PIDarr.size):
            PIDarr[i] = int(strname[i].split(".")[2])
        PIDarr = np.sort(PIDarr)
        for i in np.arange(PIDarr.size):
            TimeECHO(self.prefix+"i="+str(i)+". PID="+str(PID)+". PIDarr[i]="+str(PIDarr[i]))
            if PID == PIDarr[i]:
                node = i

        if node < 0:
            TimeECHO(self.prefix+"node < 0")
            self.removefile(filename)
            return node
        #
        TimeECHO(self.prefix+"mynode:"+str(node))
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
        elif "win" in self.控制端 and "127.0.0.1" in self.LINK:
            # 可以通过cmd控制模拟器: f"start /MIN C:\Progra~1\BlueStacks_nxt\HD-Player.exe --instance {instance}" (windows通用，不运行期间可彻底关闭模拟器，省电)
            # 也可以adb reboot控制模拟器(安卓通用，但是BlueStack模拟器不支持)
            # 通过是否运行多开管理，来判断是否使用模拟器
            # LD模拟器支持adb reboot重启模拟器
            BluePID = 0
            LdPID = 0
            # 模拟器启动后的窗口的名字
            self.win_WindowsName = []
            # 模拟器内部的名字(快捷方式中可以查看到)
            self.win_InstanceName = []
            if os.path.exists(os.path.join(BlueStackdir, "HD-MultiInstanceManager.exe")):
                BluePID = getpid_win(IMAGENAME="HD-MultiInstanceManager.exe", key="BlueStacks")
            if os.path.exists(os.path.join(LDPlayerdir, "dnmultiplayer.exe")):
                LdPID = getpid_win(IMAGENAME="dnmultiplayer.exe", key="dnmultiplayer")
            if BluePID > 0:
                self.客户端 = "win_BlueStacks"
                Instance = ["", "1", "2", "3", "4", "5"]
                for i in Instance:
                    if len(i) == 0:
                        self.win_WindowsName.append(f"BlueStacks App Player")
                        # 引擎, Nougat64,Nougat32,Pi64
                        self.win_InstanceName.append(f"Nougat32")
                    else:
                        self.win_WindowsName.append(f"BlueStacks App Player {i}")
                        self.win_InstanceName.append(f"Nougat32_{i}")
                #
            elif LdPID > 0:
                self.客户端 = "win_LD"
                # LD多开模拟器的ID, 通过添加桌面快捷方式可以获取
                Instance = ["0", "1", "2", "3", "4", "5"]
                for i in Instance:
                    self.win_InstanceName.append(f"index={i}")
                    if i == "0":
                        self.win_WindowsName.append(f"雷电模拟器")
                    else:
                        self.win_WindowsName.append(f"雷电模拟器-{i}")
                # LDPlayer 也支持 self.客户端="FULL_ADB" 的模式
                # 但是需要提前开启模拟器
            else:
                self.客户端 = "RemoteAndroid"
        elif "linux" in self.控制端 and "127.0.0.1" in self.LINK:  # Linux + docker
            if os.path.exists("/home/cndaqiang/builddocker/redroid/8arm0"):
                self.客户端 = "lin_docker"
        elif len(self.LINKport) > 0:  # 通过网络访问的安卓设备
            self.客户端 = "RemoteAndroid"
        else:
            self.客户端 = "USBAndroid"
        #
        self.mynode = mynode
        self.prefix = f"({self.mynode})"
        self.totalnode = totalnode
        #
        self.实体终端 = False
        self.实体终端 = "mac" in self.控制端 or "ios" in self.设备类型
        self.容器优化 = "linux" in self.控制端 and "android" in self.设备类型
        #
        TimeECHO(self.prefix+f"控制端({self.控制端})")
        TimeECHO(self.prefix+f"客户端({self.客户端})")
        TimeECHO(self.prefix+f"ADB =({self.adb_path})")
        TimeECHO(self.prefix+f"LINK({self.LINK})")
        TimeECHO(self.prefix+f"LINKhead({self.LINKhead})")
        TimeECHO(self.prefix+f"LINKtype({self.LINKtype})")
        TimeECHO(self.prefix+f"LINKURL({self.LINKURL})")
        TimeECHO(self.prefix+f"LINKport({self.LINKport})")
        #
        self.连接设备()

    def 连接设备(self, times=1, timesMax=2):
        """
        # 尝试连接timesMax+1次,当前是times次
        """
        self.device = False
        TimeECHO(self.prefix+f"{self.LINK}:开始第{times}/{timesMax+1}次连接")
        try:
            self.device = connect_device(self.LINK)
            if self.device:
                TimeECHO(self.prefix+f"{self.LINK}:链接成功")
                return True
        except:
            if times == timesMax+1:
                traceback.print_exc()
            TimeErr(self.prefix+f"{self.LINK}:链接失败")
            if "ios" in self.设备类型:
                TimeECHO(self.prefix+"重新插拔数据线")
        #
        if times <= timesMax:
            TimeECHO(self.prefix+f"{self.LINK}:链接失败,重启设备再次连接")
            self.启动设备()
            return self.连接设备(times+1, timesMax)
        else:
            TimeErr(self.prefix+f"{self.LINK}:链接失败,无法继续")
            return False

    def 启动设备(self):
        command = []
        TimeECHO(self.prefix+f"尝试启动设备中...")
        if self.客户端 == "ios":
            if "mac" in self.控制端:
                TimeECHO(self.prefix+f"测试本地IOS打开中")
            else:
                TimeECHO(self.prefix+f"当前模式无法打开IOS")
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
                TimeErr(self.prefix+": tidevice list 无法找到IOS设备重启失败")
                return False
        # android
        elif self.客户端 == "win_BlueStacks":
            instance = self.win_InstanceName[self.mynode]
            command.append([os.path.join(BlueStackdir, "HD-Player.exe"), "--instance", instance])
        elif self.客户端 == "win_LD":
            instance = self.win_InstanceName[self.mynode]
            command.append([os.path.join(LDPlayerdir, "dnplayer.exe"), instance])
        elif self.客户端 == "FULL_ADB":
            # 通过reboot的方式可以实现重启和解决资源的效果
            command.append([self.adb_path, "connect", self.LINKURL])
            command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "lin_docker":
            虚拟机ID = f"androidcontain{self.mynode}"
            command.append(["docker", "restart", 虚拟机ID])
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
                TimeECHO(self.prefix+f"没有找到USB设备{self.LINKURL}\n"+result[1])
                return False
        else:
            TimeECHO(self.prefix+f"未知设备类型")
            return False
        # 开始运行
        exit_code = run_command(command=command, prefix=self.prefix)
        if exit_code == 0:
            TimeECHO(self.prefix+f"启动成功")
            return True
        else:
            TimeErr(self.prefix+f"启动失败")
            return False

    def 关闭设备(self):
        command = []
        TimeECHO(self.prefix+f"尝试关闭设备中...")
        if self.客户端 == "ios":
            if "mac" in self.控制端:
                TimeECHO(self.prefix+f"测试本地IOS关闭中")
                command.append(["tidevice", "reboot"])
            else:
                TimeECHO(self.prefix+f"当前模式无法关闭IOS")
                return False
        # android
        elif self.客户端 == "win_BlueStacks":
            # 尝试获取PID
            PID = getpid_win(IMAGENAME="HD-Player.exe", key=self.win_WindowsName[self.mynode])
            # BlueStacks App Player 3
            if PID > 0:
                command.append(["taskkill", "/F", "/FI", f"PID eq {str(PID)}"])
            else:  # 关闭所有虚拟机，暂时用不到
                command.append(["taskkill", "/F", "/IM", "HD-Player.exe"])
        elif self.客户端 == "win_LD":
            # 尝试获取PID
            PID = getpid_win(IMAGENAME="dnplayer.exe", key=self.win_WindowsName[self.mynode])
            if PID > 0:
                command.append(["taskkill", "/F", "/FI", f"PID eq {str(PID)}"])
            else:
                # 关闭所有虚拟机，暂时用不到
                # command.append('taskkill /f /im dnplayer.exe')
                # 通过reboot的方式可以实现重启和解决资源的效果
                # LDPlayer支持adb reboot,👍
                command.append([self.adb_path, "connect", self.LINKURL])
                command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "FULL_ADB":
            # 通过reboot的方式可以实现重启和解决资源的效果
            command.append([self.adb_path, "connect", self.LINKURL])
            command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
        elif self.客户端 == "lin_docker":
            虚拟机ID = f"androidcontain{self.mynode}"
            command.append(["docker", "stop", 虚拟机ID])
        elif self.客户端 == "RemoteAndroid":
            # 热重启系统
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "stop"])
            command.append([self.adb_path, "-s", self.LINKURL, "shell", "start"])
            command.append([self.adb_path, "disconnect", self.LINKURL])
        elif self.客户端 == "USBAndroid":
            result = getstatusoutput("adb devices")
            if self.LINKURL in result[1]:
                command.append([self.adb_path, "-s", self.LINKURL, "reboot"])
            else:
                TimeECHO(self.prefix+f"没有找到USB设备{self.LINKURL}\n"+result[1])
                return False
        else:
            TimeECHO(self.prefix+f"未知设备类型")
            return False
        # 开始运行
        exit_code = run_command(command=command, prefix=self.prefix, sleeptime=60)
        if exit_code == 0:
            TimeECHO(self.prefix+f"关闭成功")
            return True
        else:
            TimeECHO(self.prefix+f"关闭失败")
            return False

    def 重启设备(self, sleeptime=0):
        TimeECHO(self.prefix+f"重新启动({self.LINK})")
        self.关闭设备()
        sleeptime = max(10, sleeptime-60)
        printtime = max(30, sleeptime/10)
        TimeECHO(self.prefix+"sleep %d min" % (sleeptime/60))
        for i in np.arange(int(sleeptime/printtime)):
            TimeECHO(self.prefix+f"...taskkill_sleep: {i}", end='\r')
            sleep(printtime)
        self.启动设备()
        self.连接设备()


class appOB:
    def __init__(self, prefix="", APPID="", big=False, device=None):
        self.prefix = prefix
        self.APPID = APPID
        self.Activity = None if "/" not in self.APPID else self.APPID.split("/")[1]
        self.APPID = self.APPID.split("/")[0]
        self.device = device
        self.big = big  # 是不是大型的程序, 容易卡顿，要多等待一会
    #

    def 打开APP(self):
        if self.Activity:
            TimeECHO(self.prefix+f"打开APP[{self.APPID}/{self.Activity}]中")
            启动成功 = start_app(self.APPID, self.Activity)
        else:
            TimeECHO(self.prefix+f"打开APP[{self.APPID}]中")
            启动成功 = start_app(self.APPID, prefix=self.prefix)
        if not 启动成功:
            TimeErr(self.prefix+"打开失败,可能失联")
            return False
        else:
            sleep(20)
        return True

    def 重启APP(self, sleeptime=0):
        TimeECHO(self.prefix+f"重启APP中")
        self.关闭APP()
        sleep(10)
        sleeptime = max(10, sleeptime)  # 这里的单位是s
        printtime = max(30, sleeptime/10)
        if sleeptime > 60*60 and self.device:  # >1h
            self.device.重启设备(sleeptime)
        else:
            TimeECHO(self.prefix+"sleep %d min" % (sleeptime/60))
            nstep = int(sleeptime/printtime)
            for i in np.arange(nstep):
                TimeECHO(self.prefix+f"...taskkill_sleep: {i}/{nstep}", end='\r')
                sleep(printtime)
        TimeECHO(self.prefix+f"打开程序")
        if self.打开APP():
            if self.big:
                TimeECHO(self.prefix+f"打开程序成功,sleep60*2")
                sleep(60*2)
            return True
        else:
            TimeECHO(self.prefix+f"打开程序失败")
            return False
    #

    def 关闭APP(self):
        TimeECHO(self.prefix+f"关闭APP[{self.APPID}]中")
        if not stop_app(self.APPID, prefix=self.prefix):
            TimeErr(self.prefix+"关闭失败,可能失联")
            return False
        else:
            sleep(5)
            return True
