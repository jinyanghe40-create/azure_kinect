#!/bin/bash
# =====================================================================
#  在【板子上】执行：分层网络体检
#  由电脑上的 board.sh net 通过 ssh 调用
#
#  为什么要"分层"：不同层用不同工具，走不走代理也不一样。
#      ① DNS    解析域名 → IP        （会被污染，不走代理）
#      ② ICMP   ping                 （★ 永远不走代理）
#      ③ TCP    端口通不通           （不走代理）
#      ④ 代理端口本身                 （能不能连上电脑）
#      ⑤ HTTP   真正的网页请求        （★ 只有这层走代理）
#  用错层测，就会得出错误结论（比如"ping 不通 = 上不了网"）。
# =====================================================================
set -u
PROXY=${1:-10.42.0.1:7890}

echo "════════ ① DNS 解析 ════════"
echo "  ⚠️ 国内 DNS 会被污染：同一个域名两次解析结果都可能不一样"
for d in mirrors.aliyun.com baidu.com google.com www.google.com; do
    ip=$(python3 -c "import socket;print(socket.gethostbyname('$d'))" 2>/dev/null)
    printf "  %-24s → %s\n" "$d" "${ip:-解析失败}"
done
echo "  参考: 谷歌真实 IP 是 142.250.x.x / 172.217.x.x；"
echo "        如果解析出别的（比如 104.244.x.x 是 Twitter），就是被污染了"

echo
echo "════════ ② ICMP ping（★ 永远不走 HTTP 代理）════════"
for ip in 223.5.5.5 8.8.8.8 142.250.73.78; do
    if ping -c1 -W2 "$ip" >/dev/null 2>&1; then
        printf "  ping %-16s ✅ 通\n" "$ip"
    else
        printf "  ping %-16s ❌ 不通\n" "$ip"
    fi
done
echo "  注: 223.5.5.5=阿里DNS 8.8.8.8=谷歌DNS 142.250.73.78=谷歌服务器"
echo "      谷歌【不回 ICMP】，所以 ping 它通不通都不能说明问题"

echo
echo "════════ ③ TCP 端口（不走代理）════════"
for hp in mirrors.aliyun.com:443 www.google.com:443; do
    h=${hp%:*}; p=${hp#*:}
    if timeout 4 bash -c "cat < /dev/null > /dev/tcp/$h/$p" 2>/dev/null; then
        printf "  %-30s ✅ 能连\n" "$hp"
    else
        printf "  %-30s ❌ 连不上\n" "$hp"
    fi
done

echo
echo "════════ ④ 电脑的代理端口 ════════"
h=${PROXY%:*}; p=${PROXY#*:}
if timeout 4 bash -c "cat < /dev/null > /dev/tcp/$h/$p" 2>/dev/null; then
    echo "  $PROXY ✅ 能连"
else
    echo "  $PROXY ❌ 连不上"
    echo "     → 电脑上: ①clash 开「允许局域网连接」 ②sudo ufw allow in on enp5s0 to any port $p proto tcp"
fi

echo
echo "════════ ⑤ 真·HTTP 请求（★ 只有这层走代理）════════"
python3 - "$PROXY" <<'PY'
import sys
import urllib.request
proxy = sys.argv[1]
ph = urllib.request.ProxyHandler({"http": "http://" + proxy,
                                  "https": "http://" + proxy})
op = urllib.request.build_opener(ph)
for u in ("https://mirrors.aliyun.com/", "https://www.baidu.com",
          "https://www.google.com"):
    try:
        r = op.open(u, timeout=15)
        print("  %-32s ✅ HTTP %s" % (u, r.status))
    except urllib.error.HTTPError as e:
        # ★ 服务器【回话了】，只是拒绝了我们 —— 这不是网络问题！
        print("  %-32s ⚠️ 服务器返回 HTTP %s（网络是通的，是站点拒绝）"
              % (u, e.code))
    except Exception as e:
        print("  %-32s ❌ %s（这才是真的连不上）" % (u, type(e).__name__))
PY

echo
echo "════════ 结论怎么看 ════════"
echo "  ① 解析正常 + ② ping 国内通 + ⑤ 谷歌 200  →  一切正常，ping 谷歌不通是【正常的】"
echo "  最有用的一条：把 ⑤ 当成'板子能不能上外网'的唯一判据"
