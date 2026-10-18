#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把带 numpy 的字典变成 JSON 能序列化的纯 Python 对象。

★★ 为什么值得一个单独的文件 —— 这是一个【静默到极点的】故障：
     `json.dumps(np.float32(0.9))` / `json.dumps(np.array([1,2,3]))`
     都会抛 TypeError。
   而串口/ROS 的编码是在【后台线程】里做的，那个线程的异常没人接：

        send() 一直成功返回  →  队列一直在涨  →  主循环一无所知
        但一个字都没发出去，而且【永远】不会再发。

    也就是说：**电控那边断了，板子上一切正常，日志全绿。**
    实测触发路径：`--mm` 会把 face/cube/range 转成 list（所以看着没问题），
    但 `normal` 不在长度白名单里，原样是 ndarray → 第一帧就炸后台线程。
    服务单元恰恰就带 `--mm`。

  所以：所有跨进程发出的东西，都必须先过 `jsonable()`。
"""
import numpy as np


def jsonable(o):
    """递归地把 ndarray / numpy 标量换成 list / 原生标量。

    只处理"JSON 认识"的容器（dict / list / tuple），其余原样交给
    json.dumps 去报错 —— 不吞异常，不猜。
    """
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        # ★ .tolist() 会把 float32 降级成 python float —— 正是我们要的。
        #   标量 0-d 数组也走这条，tolist() 直接给标量。
        return o.tolist()
    if isinstance(o, np.generic):           # np.float32 / np.int64 / np.bool_
        return o.item()
    return o


def dumps(rec, **kw):
    """json.dumps 的 jsonable 版本。串口和 ROS 都走这个。"""
    import json
    return json.dumps(jsonable(rec), **kw)
