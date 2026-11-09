# -*- coding: utf-8 -*-
"""智能体四步循环骨架（初判 → 取证 → 复核 → 终止），工具为桩函数。

用途：把选题报告「方案设计·Agent」一节的流程落成可运行的骨架，先验证
      工具契约、状态保存、循环终止与轨迹落盘，再替换为真实模型与真实检索。

注意：本文件中的三个工具都是桩函数（用规则模拟返回），
      只用于验证流程，不代表任何实验结论，也不是最终实现。
运行：python agent_loop.py
"""

import time

MAX_ROUNDS = 3              # 最多工具调用轮数
LOW_CONF = 0.75             # 初判置信度低于此值即取证（阈值偏保守，真实系统按验证集调）
TOOL_TIMEOUT = 3.0          # 单工具超时（秒）
MAX_SECONDS = 20.0          # 单条样本总时长上限
MIN_SIM = 0.10              # 相似案例的最低相似度，低于此值视为没检索到

# 工具名 → 证据槽位
TOOL_KEY = {"check_sensitive": "sensitive", "classify_comment": "model", "retrieve_similar": "cases"}

STRONG_SLURS = ("傻子", "废物", "脑子有问题", "去死")
GROUP_PATTERNS = ("都是骗子", "就是不适合", "就是不行", "就该被")
SENSITIVE = ("傻子", "废物", "脑子有问题", "地域", "女生", "不适合")

# 桩案例库（真实实现应替换为对训练集的向量检索，绝不包含测试集）
CASES = [
    ("这家店的服务态度很好，下次还会再来。", 0),
    ("不要因为地域就歧视别人，这不公平。", 0),
    ("呵呵，你们女生就是不适合学理科。", 1),
    ("某地的人都是骗子，别再跟他们做生意。", 1),
]


# ----------------------------- 工具（桩） -----------------------------
def classify_comment(text):
    """工具一：调用微调后的专用模型，返回 (标签, 概率)。真实实现为 BERT 类模型推理。"""
    if any(w in text for w in STRONG_SLURS):
        return 1, 0.92
    if "歧视" in text and any(w in text for w in ("不要", "别", "反对")):
        return 0, 0.88                      # 反偏见表述：含敏感词但并非冒犯
    if any(p in text for p in GROUP_PATTERNS):
        return 1, 0.66
    return 0, 0.80


def retrieve_similar(text, k=3):
    """工具二：在训练集（此处为桩案例库）中检索语义最近的已标注案例。"""
    def sim(a, b):
        sa = {a[i:i + 2] for i in range(len(a) - 1)}
        sb = {b[i:i + 2] for i in range(len(b) - 1)}
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

    scored = sorted(((sim(text, c), c, y) for c, y in CASES), reverse=True)
    return [{"case": c, "label": y, "score": round(s, 3)}
            for s, c, y in scored[:k] if s >= MIN_SIM]


def check_sensitive(text):
    """工具三：返回命中的敏感表达及位置（可选工具）。"""
    return [{"word": w, "pos": text.find(w)} for w in SENSITIVE if w in text]


# ----------------------------- 初判（桩） -----------------------------
def initial_judge(text):
    """模拟大模型的第一次独立判断。置信度刻意留低，用来触发取证分支。"""
    hits = check_sensitive(text)
    if len(hits) >= 2:
        return 1, 0.72, "命中多个敏感表达"
    if len(hits) == 1:
        return 1, 0.52, "命中敏感表达「%s」，但语境未定" % hits[0]["word"]
    return 0, 0.58, "未命中敏感表达，初判为正常，把握不大"


# ----------------------------- 主循环 -----------------------------
def run_agent(text, max_rounds=MAX_ROUNDS):
    t0 = time.time()
    trace = []
    state = {
        "text": text,
        "initial": None,
        "tools_called": [],
        "round": 0,
        "elapsed": 0.0,
    }

    # 第 1 步：初判
    label0, conf0, why = initial_judge(text)
    state["initial"] = (label0, conf0)
    trace.append({"step": 1, "action": "初判", "output": {"label": label0, "conf": conf0, "reason": why}})

    # 第 2 步：取证（仅在需要时）
    need_evidence = conf0 < LOW_CONF
    evidence = {"model": None, "cases": None, "sensitive": None}
    if need_evidence:
        trace.append({"step": 2, "action": "取证", "output": {"trigger": "初判置信度 %.2f < %.2f" % (conf0, LOW_CONF)}})
        for name, fn, arg in (
            ("check_sensitive", check_sensitive, text),
            ("classify_comment", classify_comment, text),
            ("retrieve_similar", retrieve_similar, text),
        ):
            if state["round"] >= max_rounds or time.time() - t0 > MAX_SECONDS:
                trace.append({"step": 2, "action": "终止取证", "output": {"reason": "达到轮数或时长上限"}})
                break
            state["round"] += 1
            started = time.time()
            try:
                result = fn(arg)
            except Exception as exc:                     # 工具异常：跳过并降级
                trace.append({"step": 2, "action": name, "output": {"error": str(exc), "degraded": True}})
                continue
            if time.time() - started > TOOL_TIMEOUT:     # 超时：跳过并降级
                trace.append({"step": 2, "action": name, "output": {"timeout": True, "degraded": True}})
                continue
            state["tools_called"].append(name)
            evidence[TOOL_KEY[name]] = result
            trace.append({"step": 2, "action": name, "output": result})

    # 第 3 步：复核（没有任何工具证据时维持初判，不做无依据改判）
    got_evidence = any(v for v in evidence.values())
    score = 0.4 * label0
    parts = ["初判 %d（%.2f）" % (label0, conf0)]
    if evidence["model"] is not None:
        m_label, m_prob = evidence["model"]
        score += 0.4 * m_label * (0.5 + m_prob / 2)
        parts.append("专用模型 %d（%.2f）" % (m_label, m_prob))
    if evidence["cases"]:
        votes = [c["label"] for c in evidence["cases"][:3]]
        maj = 1 if sum(votes) * 2 > len(votes) else 0
        score += 0.2 * maj
        parts.append("相似案例 {}/{} 判为冒犯".format(sum(votes), len(votes)))
    if evidence["sensitive"]:
        parts.append("命中敏感表达 %d 个" % len(evidence["sensitive"]))
    final = (1 if score >= 0.5 else 0) if got_evidence else label0
    final_reason = "；".join(parts) if got_evidence else why + "（无工具证据，维持初判）"
    trace.append({
        "step": 3, "action": "复核",
        "output": {"score": round(score, 3), "label": final, "reason": final_reason},
    })

    # 第 4 步：终止
    state["elapsed"] = round(time.time() - t0, 4)
    trace.append({
        "step": 4, "action": "终止",
        "output": {"rounds": state["round"], "tools": state["tools_called"],
                   "elapsed_s": state["elapsed"]},
    })
    return {"text": text, "label": final, "changed": final != label0,
            "reason": final_reason, "trace": trace, "state": state}


DEMO = [
    "这家店的服务态度很好，下次还会再来。",
    "不要因为地域就歧视别人，这不公平。",
    "呵呵，你们女生就是不适合学理科。",
]


def _show(result):
    print("文本:", result["text"])
    for step in result["trace"]:
        print("  [%d] %-14s %s" % (step["step"], step["action"], step["output"]))
    print("  => 最终标签 %d%s\n" % (result["label"], "（初判被改判）" if result["changed"] else ""))


if __name__ == "__main__":
    for text in DEMO:
        _show(run_agent(text))
