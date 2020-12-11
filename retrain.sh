#!/bin/bash

ps aux|grep cn_punctor_bert.py|grep -v grep|awk '{print $2}'|xargs kill -9

export PYTHONUNBUFFERED=1  # 设置缓存大小

nohup python3 cn_punctor_bert.py &

tail -f nohup.out
