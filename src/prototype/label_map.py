# -*- coding: utf-8 -*-
"""标签映射与数据统计原型。

用途：
  1) 把 COLD 测试集的细粒度标签统一映射为二分类标签，规则只写一处，三条线共用；
  2) 复算数据规模与标签分布，生成 data_report.txt，报告中的数字一律以此为准。

标签对应关系（依据官方仓库 README）：
  train.csv / dev.csv 的 label        0 = safe, 1 = offensive
  test.csv 的 fine-grained-label      0 = safe(other-non-offen) → 0
                                      1 = attack individual     → 1
                                      2 = attack group          → 1
                                      3 = safe(anti-bias)       → 0

运行：
  python label_map.py                     # 无参数时跑内置桩样例自测
  python label_map.py train.csv dev.csv test.csv   # 复算真实数据并写 data_report.txt
"""

import csv
import os
import sys

FG_TO_BINARY = {0: 0, 1: 1, 2: 1, 3: 0}
FG_NAMES = {0: "其他非冒犯", 1: "攻击个人", 2: "攻击群体", 3: "反偏见"}

# 只用于自测的桩样例，不是 COLD 原文，不得写入报告
TOY_ROWS = [
    ("这家店的服务态度很好，下次还会再来。", 0, 0),
    ("你这种人就该被开除，脑子有问题。", 1, 1),
    ("某地的人都是骗子，别再跟他们做生意。", 1, 2),
    ("不要因为地域就歧视别人，这不公平。", 0, 3),
    ("呵呵，你们女生就是不适合学理科。", 1, 2),
    ("周末去爬山，天气不错。", 0, 0),
]


def map_fine_grained(value):
    """细粒度标签 → 二分类标签。无法识别时抛错，避免静默出错。"""
    if value is None or value == "":
        raise ValueError("empty fine-grained label")
    v = int(value)
    if v not in FG_TO_BINARY:
        raise ValueError("unknown fine-grained label: %r" % value)
    return FG_TO_BINARY[v]


def avg_chars(texts):
    return round(sum(len(t) for t in texts) / len(texts), 2) if texts else 0.0


def summarize(rows):
    """rows: [(text, binary_label, fine_grained_or_None)] → 统计字典。"""
    total = len(rows)
    offensive = sum(1 for _, y, _ in rows if y == 1)
    fg = {}
    for _, _, f in rows:
        if f is not None:
            fg[f] = fg.get(f, 0) + 1
    return {
        "total": total,
        "offensive": offensive,
        "safe": total - offensive,
        "offensive_ratio": round(offensive / total, 4) if total else 0.0,
        "fg_counts": {FG_NAMES[k]: v for k, v in sorted(fg.items())},
        "avg_chars": avg_chars([t for t, _, _ in rows]),
    }


def _pick(header, candidates):
    for c in candidates:
        if c in header:
            return c
    return None


def read_csv(path, with_fine_grained=None):
    """读取 COLD 的 csv。列名自动探测，避免不同版本表头差异导致失败。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        text_col = _pick(header, ["content", "text", "comment", "sentence"])
        label_col = _pick(header, ["label", "binary_label", "offensive"])
        fg_col = _pick(header, ["fine-grained-label", "fine_grained_label", "fg_label"])
        if text_col is None or label_col is None:
            raise KeyError("无法识别表头: %s" % header)
        if with_fine_grained is None:
            with_fine_grained = fg_col is not None
        rows = []
        for r in reader:
            text = (r.get(text_col) or "").strip()
            if not text:
                continue
            y = int(r[label_col])
            f = map_fine_grained(r.get(fg_col)) if (with_fine_grained and fg_col) else None
            # 注意：test.csv 用细粒度标签推导二分类，与 label 列应一致，此处以细粒度为准
            rows.append((text, f if f is not None else y, f))
        return rows


def write_report(sections, out_path="data_report.txt"):
    lines = ["COLD 数据校验报告", "=" * 40]
    for name, s in sections:
        lines.append("[%s]" % name)
        lines.append("  总条数: %d（冒犯 %d / 正常 %d，冒犯占比 %.1f%%）"
                     % (s["total"], s["offensive"], s["safe"], s["offensive_ratio"] * 100))
        lines.append("  平均字符长度: %.2f" % s["avg_chars"])
        if s["fg_counts"]:
            for k, v in s["fg_counts"].items():
                lines.append("  细粒度 %s: %d" % (k, v))
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _selftest():
    rows = [(t, map_fine_grained(f) if f is not None else y, f) for t, y, f in TOY_ROWS]
    s = summarize(rows)
    assert s["total"] == 6 and s["offensive"] == 3, s
    # 断言：4 条细粒度映射后的二分类结果与人工判断一致
    assert [y for _, y, _ in rows] == [0, 1, 1, 0, 1, 0], rows
    assert map_fine_grained(3) == 0 and map_fine_grained(2) == 1
    try:
        map_fine_grained(9)
        raise AssertionError("应当抛出异常")
    except ValueError:
        pass
    print("selftest passed（桩样例 6 条，非 COLD 原文）")
    print(s)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3:
        sections = []
        for name, path in zip(["train", "dev", "test"], args[:3]):
            if not os.path.exists(path):
                sys.exit("文件不存在: %s" % path)
            rows = read_csv(path, with_fine_grained=(name == "test"))
            sections.append((name, summarize(rows)))
        out = write_report(sections)
        print("written:", out)
        for name, s in sections:
            print(name, s)
    else:
        _selftest()
        print("\n用法: python label_map.py train.csv dev.csv test.csv")
