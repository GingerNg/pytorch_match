#!/bin/bash

ps aux|grep cmdd_bert.py|grep -v grep|awk '{print $2}'|xargs kill -9

export PYTHONUNBUFFERED=1  # 设置缓存大小

nohup python3 cmdd_bert.py &

tail -f nohup.out
