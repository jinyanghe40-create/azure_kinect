#!/usr/bin/env bash
# 修掉"板子 /tmp 被定时整个清空"的坑
#
# 症状：
#   日志、脚本、X socket 会莫名其妙消失，看起来像"进程自己没了"。
#   实测：systemd-tmpfiles-clean.timer 跑过一次之后，
#         /tmp/rs.log、/tmp/.X11-unix/X1 全没了。
#
# 根因：
#   Jetson L4T 的 /usr/lib/tmpfiles.d/tmp.conf 里写的是
#       D /tmp 1777 root root -
#   ★ D = 每次清理【清空整个目录】，而且 `-` 表示不限年龄
#   而标准 Ubuntu 是：
#       q /tmp 1777 root root 10d        （只删 10 天前的）
#
# 修法：
#   在 /etc/tmpfiles.d/ 放一份同名覆盖文件。
#   （/etc 的优先级高于 /usr/lib，同名会覆盖）
#
# 用法（板子上，一次性）:
#   sudo bash ~/work/azure_kinect/fix_tmp_cleanup.sh
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root：sudo bash $0"
    exit 1
fi

echo "=== 现在的规则 ==="
grep -v '^#' /usr/lib/tmpfiles.d/tmp.conf | grep -v '^$' | sed 's/^/  /'

cat > /etc/tmpfiles.d/tmp.conf <<'EOF'
# 覆盖 /usr/lib/tmpfiles.d/tmp.conf
# ★ Jetson L4T 默认是 `D /tmp ... -`（每次清空整个 /tmp），
#   会把日志、socket 全删掉，看起来像进程莫名消失。
#   这里改成 Ubuntu 的标准行为：只删 10 天前的文件。
q /tmp 1777 root root 10d
q /var/tmp 1777 root root 30d

# systemd 私有目录照原样保留
x /tmp/systemd-private-%b-*
X /tmp/systemd-private-%b-*/tmp
x /var/tmp/systemd-private-%b-*
X /var/tmp/systemd-private-%b-*/tmp
R! /tmp/systemd-private-*
R! /var/tmp/systemd-private-*
EOF

echo
echo "=== 覆盖后 ==="
grep -v '^#' /etc/tmpfiles.d/tmp.conf | grep -v '^$' | sed 's/^/  /'

echo
echo "✅ 搞定。以后 /tmp 只清 10 天前的文件。"
echo
echo "参考：如果 /tmp 里的东西你都不用（我们日志已放 ~/work/logs/），"
echo "      不改也行 —— 这个脚本只是让行为符合直觉。"
