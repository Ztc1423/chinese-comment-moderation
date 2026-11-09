# -*- coding: utf-8 -*-
"""评测指标原型：二类宏平均 F1、冒犯类召回率、误杀率、输出有效率。

用途：固化选题报告「判定方式」一节的计分口径，供三条线共用，避免各自另写一套。
依赖：仅标准库。运行：python metrics.py（会执行自测并打印示例）。

标签约定：0 = 正常（safe），1 = 冒犯（offensive）；
          None 表示无法解析的无效输出，一律计错并保留在分母中。
"""


def confusion(y_true, y_pred, positive=1):
    """返回 (TP, FP, FN, TN)。无效输出（None）按判错处理。"""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p is None:                      # 无效输出：一律记错，不从分母剔除
            if t == positive:
                fn += 1
            else:
                fp += 1
            continue
        if t == positive and p == positive:
            tp += 1
        elif t == positive:
            fn += 1
        elif p == positive:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def f1_score(tp, fp, fn):
    """单类 F1：2PR/(P+R)，分母为 0 时返回 0。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true, y_pred):
    """二类宏平均 F1 = (正常类 F1 + 冒犯类 F1) / 2。"""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    f1_offensive = f1_score(tp, fp, fn)
    f1_normal = f1_score(tn, fn, fp)       # 以「正常」为正类时，TN 充当 TP、FN 充当 FP
    return (f1_offensive + f1_normal) / 2


def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def offensive_recall(y_true, y_pred):
    """冒犯类召回率 = TP / (TP + FN)，衡量漏检程度。"""
    tp, _, fn, _ = confusion(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def false_positive_rate(y_true, y_pred):
    """误杀率 = FP / (FP + TN)，衡量正常言论被误判为冒犯的比例。"""
    _, fp, _, tn = confusion(y_true, y_pred)
    return fp / (fp + tn) if fp + tn else 0.0


def valid_rate(y_pred):
    """输出有效率 = 可解析输出数 / 总样本数。"""
    return sum(1 for p in y_pred if p is not None) / len(y_pred) if y_pred else 0.0


def report(y_true, y_pred):
    """汇总为一份可直接写进 csv 的结果字典。"""
    return {
        "n": len(y_true),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "offensive_recall": round(offensive_recall(y_true, y_pred), 4),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred), 4),
        "valid_rate": round(valid_rate(y_pred), 4),
    }


# ----------------------------- 自测 -----------------------------
def _selftest():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0]
    r = report(y_true, y_pred)
    assert r["macro_f1"] == 0.5833, r
    assert r["accuracy"] == 0.6, r
    assert abs(r["offensive_recall"] - 2 / 3) < 1e-3, r
    assert abs(r["false_positive_rate"] - 0.5) < 1e-3, r

    # 无效输出必须计错：一条 None 让这一条变成错误，但分母不变
    r2 = report([0, 1], [None, 1])
    assert r2["valid_rate"] == 0.5 and r2["accuracy"] == 0.5, r2

    # 空模型演示：全判「正常」也能拿到 60.4% 准确率，但宏平均 F1 只有 0.38
    n_off, n_norm = 2107, 3216            # COLD 测试集真实分布
    yt = [1] * n_off + [0] * n_norm
    yp = [0] * len(yt)
    r3 = report(yt, yp)
    assert abs(r3["accuracy"] - 0.6042) < 1e-3, r3
    assert abs(r3["macro_f1"] - 0.3766) < 1e-3, r3
    print("selftest passed")
    print("示例（COLD 测试集分布，全判正常的空模型）:", r3)
    print("结论：准确率会被多数类掩盖，主指标必须看宏平均 F1。")


if __name__ == "__main__":
    _selftest()
