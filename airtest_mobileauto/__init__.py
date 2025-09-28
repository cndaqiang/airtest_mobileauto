#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This package exposes the core functionalities from the control.py module.
By importing this package, you get direct access to all major classes, functions,
and variables needed for the automation tasks.
"""

from .control import (
    # =================================================================
    # 主要的类 (Major Classes)
    # =================================================================
    Settings,          # 全局设置类
    readyaml,          # YAML文件读取类
    DQWheel,           # 核心控制和同步工具类
    deviceOB,          # 设备对象，用于连接、重启等操作
    appOB,             # 应用对象，用于启动、关闭APP等操作
    TaskManager,       # 多进程任务管理器

    # =================================================================
    # 重写的Airtest核心函数 (Rewritten Airtest Core Functions)
    # =================================================================
    exists,            # 检查图片是否存在
    connect_device,    # 连接设备
    touch,             # 点击操作
    swipe,             # 滑动操作
    start_app,         # 启动应用
    stop_app,          # 关闭应用
    Template,          # 图片模板对象

    # =================================================================
    # 日志和调试函数 (Logging and Debugging Functions)
    # =================================================================
    logger,            # 全局日志记录器
    loggerhead,        # 生成日志前缀
    TimeECHO,          # 普通信息输出
    TimeErr,           # 错误信息输出
    TimeDebug,         # 调试信息输出

    # =================================================================
    # 辅助工具函数 (Utility Functions)
    # =================================================================
    save_yaml,         # 保存字典到YAML文件
    fun_name,          # 获取当前函数名
    funs_name,         # 获取函数调用栈名称
    getstatusoutput,   # (已弃用) 获取命令执行结果
    getPopen,          # 以非阻塞方式执行外部命令
    run_command,       # 运行外部命令
    run_class_command, # 在类实例上下文中执行Python代码字符串
    getpid_win,        # (Windows) 根据窗口名获取进程ID
    touchkey_win,      # (Windows) 模拟按键
    connect_status,    # 检查设备连接状态
)

# The __all__ variable defines the public API of the module.
# When a user does 'from airtest_mobileauto import *', only these names will be imported.
__all__ = [
    # Classes
    "Settings",
    "readyaml",
    "DQWheel",
    "deviceOB",
    "appOB",
    "TaskManager",

    # Rewritten Airtest Functions
    "sleep",
    "exists",
    "touch",
    "swipe",
    "start_app",
    "stop_app",
    "Template",

    # Logging Functions
    "logger",
    "loggerhead",
    "TimeECHO",
    "TimeErr",
    "TimeDebug",

    # Utility Functions
    "save_yaml",
    "fun_name",
    "funs_name",
    "getstatusoutput",
    "getPopen",
    "run_command",
    "run_class_command",
    "getpid_win",
    "touchkey_win",
    "connect_status",
]