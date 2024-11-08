#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################
# Author : cndaqiang             #
# Update : 2024-08-18            #
# Build  : 2024-08-18            #
# What   : 更新登录体验服         #
##################################
from airtest_mobileauto.control import *
Tool = DQWheel()
txt=["字典.中路.android.var_dict_N","字典.发育.android.var_dict_N","字典.对抗.android.var_dict_N","字典.打野.android.var_dict_N","字典.游走.android.var_dict_N"]
for i in txt:
    print(i)
    dictfile=Tool.read_dict(i+".txt")
    #
    for key, value in dictfile.items():
        if " " in key:
            print(f"- '{key}'")
    # 替换key中的空格
    new_dict = {key.replace(" ", ""): value for key, value in dictfile.items()}
    #Tool.save_dict(new_dict,i+".yaml")
    #只记录特定位置
    select_dict={}
    for key in ["参战英雄线路","参战英雄头像"]: select_dict[key]=dictfile[key]
    Tool.save_dict(select_dict,i+".yaml")
txt=["android.var_dict_0","android.var_dict_1","android.var_dict_2.ce"]
for i in txt:
    print(i)
    dictfile=Tool.read_dict(i+".txt")
    #
    for key, value in dictfile.items():
        if " " in key:
            print(f"- '{key}'")
    # 替换key中的空格
    new_dict = {key.replace(" ", ""): value for key, value in dictfile.items()}
    Tool.save_dict(new_dict,i+".yaml")
