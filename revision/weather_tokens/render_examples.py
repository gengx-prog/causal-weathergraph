"""Render saved weather-token examples with a fixed, auditable Chinese lexicon.

No language model, training, data fitting, downloads, or model-file writes.
The displayed words are a human-readable rendering of the saved S bin codes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT.parent / "revision_outputs/weather_token_pilot"
CASES = ("pooled", "nh_to_sh", "sh_to_nh")
VARIABLES = ("u", "v", "humidity", "temperature", "omega_700", "cloud_cover")
NAMES = ("850 hPa纬向风分量", "850 hPa经向风分量", "850 hPa比湿", "850 hPa温度", "700 hPa垂直压速", "整层总云量")
UNITS = ("m/s", "m/s", "kg/kg", "K", "Pa/s", "0–1比例")
LAYERS = ("850 hPa", "850 hPa", "850 hPa", "850 hPa", "700 hPa", "总云量，无单一气压层")
ANOMALY_WORDS = ("大幅偏低", "较大偏低", "偏低", "略偏低", "接近参照", "略偏高", "偏高", "较大偏高", "大幅偏高")
CHANGE_WORDS = ("大幅减小", "较大减小", "减小", "小幅减小", "变化较小", "小幅增大", "增大", "较大增大", "大幅增大")
ABS_WORDS = {
    "u": ("很强西向分量", "较强西向分量", "中等西向分量", "弱西向分量", "近零纬向分量", "弱东向分量", "中等东向分量", "较强东向分量", "很强东向分量"),
    "v": ("很强南向分量", "较强南向分量", "中等南向分量", "弱南向分量", "近零经向分量", "弱北向分量", "中等北向分量", "较强北向分量", "很强北向分量"),
    "humidity": ("比湿很低", "比湿较低", "比湿偏低", "比湿中等偏低", "比湿中等", "比湿中等偏高", "比湿偏高", "比湿较高", "比湿很高"),
    "temperature": ("绝对温度很低", "绝对温度较低", "绝对温度偏低", "绝对温度中等偏低", "绝对温度中等", "绝对温度中等偏高", "绝对温度偏高", "绝对温度较高", "绝对温度很高"),
    "omega_700": ("很强上升压速", "较强上升压速", "中等上升压速", "弱上升压速", "垂直压速近零", "弱下沉压速", "中等下沉压速", "较强下沉压速", "很强下沉压速"),
    "cloud_cover": ("总云量很低", "总云量较低", "总云量偏低", "总云量中等偏低", "总云量中等", "总云量中等偏高", "总云量偏高", "总云量较高", "总云量很高"),
}
LABEL_WORDS = {0: "总云量减少超过0.05", 1: "总云量变化介于-0.05与+0.05（含边界）", 2: "总云量增加超过0.05"}
NOTES = [
    "这是固定词典对已保存S分箱码的展示，不是LLM生成，也没有增加新的气象推理或因果规则。",
    "分箱是一种表示方法，不能据此宣称模型理解了文字或掌握了物理机制。Q码仅作训练分位分箱对照。",
    "强弱、高低等措辞只是本原型冻结数值区间的可读名称，不是业务预警等级、统计显著性或通用气象分类。",
    "每条只展示当前起报时刻；实际模型还使用此前7个时刻的词元和元信息。模型不直接读取这里生成的展示文字。",
    "异常参照来自1979–2014源域、相同绝对纬带和季节对齐月份；南半球月份加6取模。迁移目标半球不参与参照拟合。",
    "6小时变化只用当前与前一时刻；变化量由保存的float32标准化差分乘训练差分尺度恢复，存在浮点舍入。变化较小不等于严格不变。",
    "omega为Pa/s的垂直压速，负值对应上升、正值对应下沉；不是m/s的几何垂直速度。压速值增大不等于上升增强。",
    "总云量是整层0–1比例，不能解释为高层云。0.05对应5个百分点。",
    "以下是粗区域原单位算术均值；压力层未进行地下层掩码。扩展CDS时段与WeatherBench2时段的同质性尚未建立。",
    "分箱区间按左闭右开处理，尾箱无界；原值不裁剪。未来6/12/24小时标签仅用于监督，在单独字段/段落列出，不属于输入展示文本。",
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def number(value):
    return f"{float(value):.7g}"


def interval(boundaries, code, unit):
    low = None if code == 0 else float(boundaries[code-1])
    high = None if code == 8 else float(boundaries[code])
    text = ("(−∞" if low is None else "["+number(low)) + ", " + ("+∞)" if high is None else number(high)+")")
    return {"lower":low, "upper":high, "lower_inclusive":low is not None,
            "upper_inclusive":False, "unit":unit, "text":text+" "+unit}


def render_case(case_dir):
    case_dir = Path(case_dir)
    inputs = {name:case_dir/name for name in ("examples.json", "transform.npz", "case_manifest.json")}
    missing = [name for name,path in inputs.items() if not path.is_file()]
    if missing:
        return {"case":case_dir.name, "status":"pending", "missing":missing}
    hashes = {name:sha(path) for name,path in inputs.items()}
    manifest = load_json(inputs["case_manifest.json"])
    if hashes["transform.npz"] != manifest["transform_sha256"]:
        raise ValueError(f"Transform hash differs from prepared manifest: {case_dir}")
    examples = load_json(inputs["examples.json"])
    if len(examples) > 9:
        raise ValueError("Expected at most nine fixed saved examples; do not resample")
    with np.load(inputs["transform.npz"], allow_pickle=False) as archive:
        transform = {key:archive[key].copy() for key in ("abs_mean", "abs_std", "delta_std", "climatology", "absolute_latitude_bands", "boundaries_q", "boundaries_s", "physical_absolute_bin_boundaries", "variable_names", "node_ids")}
    if tuple(transform["variable_names"].tolist()) != VARIABLES or manifest["variables"] != list(VARIABLES):
        raise ValueError("Unexpected variable order")
    source = {"pooled":"南北半球源训练区共同", "nh_to_sh":"北半球源训练区", "sh_to_nh":"南半球源训练区"}[manifest["case"]]
    rendered = []
    for example in examples:
        r = int(example["local_region_index"])
        node = int(example["node_id"])
        if int(transform["node_ids"][r]) != node or int(manifest["node_ids"][r]) != node:
            raise ValueError("Example node alignment failed")
        lat, lon = float(manifest["lat"][r]), float(manifest["lon"][r])
        timestamp = np.datetime64(example["timestamp"], "ns")
        calendar_month = int(timestamp.astype("datetime64[M]").astype(np.int64) % 12)
        local_month = (calendar_month + (6 if lat < 0 else 0)) % 12
        band = int(np.argmin(np.abs(transform["absolute_latitude_bands"]-abs(lat))))
        if abs(transform["absolute_latitude_bands"][band]-abs(lat)) > 1e-5:
            raise ValueError("Latitude-band match failed")
        raw = np.asarray(example["raw"], dtype=np.float64)
        F = np.asarray(example["F"], dtype=np.float64)
        S, Q = np.asarray(example["S"], dtype=np.int64), np.asarray(example["Q"], dtype=np.int64)
        if raw.shape != (6,) or F.shape != (18,) or S.shape != (18,) or Q.shape != (18,):
            raise ValueError("Malformed saved example")
        if not np.isfinite(raw).all() or not np.isfinite(F).all():
            raise ValueError("Nonfinite saved example")
        for j in range(18):
            if int(np.searchsorted(transform["boundaries_s"][j], F[j], side="right")) != S[j]:
                raise ValueError("Saved S code is inconsistent with feature/boundary")
            if int(np.searchsorted(transform["boundaries_q"][j], F[j], side="right")) != Q[j]:
                raise ValueError("Saved Q code is inconsistent with feature/boundary")
        np.testing.assert_allclose(F[:6], (raw-transform["abs_mean"])/transform["abs_std"], rtol=1e-6, atol=1e-6)
        reference = transform["climatology"][band,local_month]
        np.testing.assert_allclose(F[6:12], (raw-reference)/transform["abs_std"], rtol=1e-6, atol=1e-6)
        prefix = (f"{str(timestamp).replace('T',' ')} UTC；区域{node}（纬度{lat:.4f}°，经度{lon:.4f}°）；"
                  f"源域参照：{source}1979–2014年、绝对纬度{abs(lat):.4f}°、季节对齐月{local_month+1}。")
        input_lines = [prefix]
        variable_rows = []
        for j,name in enumerate(VARIABLES):
            ac, nc, dc = int(S[j]), int(S[j+6]), int(S[j+12])
            absolute_interval = interval(transform["physical_absolute_bin_boundaries"][j], ac, UNITS[j])
            anomaly_interval = interval(transform["boundaries_s"][j+6], nc, "源训练原量标准差单位")
            change_interval = interval(transform["boundaries_s"][j+12], dc, "源训练6小时差分标准差单位")
            anomaly = float(raw[j]-reference[j])
            change = float(F[j+12]*transform["delta_std"][j])
            direction = "增大" if change > 0 else "减小" if change < 0 else "未变"
            words = {"absolute":ABS_WORDS[name][ac], "source_relative_anomaly":ANOMALY_WORDS[nc], "past_6h_change":CHANGE_WORDS[dc]}
            input_lines.append(
                f"{NAMES[j]}：{number(raw[j])} {UNITS[j]} → {words['absolute']}（S箱{ac}，{absolute_interval['text']}）；"
                f"源参照均值{number(reference[j])} {UNITS[j]}，异常{number(anomaly)} {UNITS[j]} → {words['source_relative_anomaly']}"
                f"（{number(F[j+6])}标准差，S箱{nc}，{anomaly_interval['text']}）；"
                f"过去6小时约{change:+.7g} {UNITS[j]} → {words['past_6h_change']}（数值{direction}，"
                f"{number(F[j+12])}差分标准差，S箱{dc}，{change_interval['text']}）。")
            variable_rows.append({
                "variable":name,"display_name":NAMES[j],"layer":LAYERS[j],"unit":UNITS[j],
                "raw_value":float(raw[j]),"absolute_S_code":ac,"absolute_bin_interval":absolute_interval,
                "source_reference_mean":float(reference[j]),"source_relative_anomaly_physical":anomaly,
                "source_relative_anomaly_standardized":float(F[j+6]),"anomaly_S_code":nc,"anomaly_bin_interval":anomaly_interval,
                "past_6h_change_physical_approx":change,"past_6h_change_direction":direction,
                "past_6h_change_standardized":float(F[j+12]),"change_S_code":dc,"change_bin_interval":change_interval,
                "Q_codes_for_comparison":{"absolute":int(Q[j]),"anomaly":int(Q[j+6]),"change":int(Q[j+12])},
                "fixed_dictionary_words":words,
            })
        supervision=[]
        for horizon,code in zip((6,12,24),example["labels_6_12_24h"]):
            code=int(code)
            if code not in LABEL_WORDS: raise ValueError("Unexpected supervision class")
            supervision.append({"horizon_hours":horizon,"class_code":code,"description":LABEL_WORDS[code]})
        rendered.append({"split":example["split"],"timestamp_utc":str(timestamp),"node_id":node,
                         "local_region_index":r,"latitude":lat,"longitude":lon,
                         "source_domain_reference":source,"season_aligned_month_one_based":local_month+1,
                         "input_display_text":"\n".join(input_lines),"variables":variable_rows,
                         "future_supervision_only_not_in_input":supervision})
    result={"status":"rendered","case":manifest["case"],"created_utc":datetime.now(timezone.utc).isoformat(),
            "renderer_sha256":sha(__file__),"input_sha256":hashes,"examples_count":len(rendered),
            "method":"Deterministic fixed lexicon rendering of saved S bins; not LLM-generated text and not model training input.",
            "notes":NOTES,"fixed_dictionary":{"absolute":ABS_WORDS,"anomaly":ANOMALY_WORDS,"past_6h_change":CHANGE_WORDS},
            "examples":rendered}
    # Confirm reads remained stable before writing only the two requested views.
    if any(sha(path)!=hashes[name] for name,path in inputs.items()):
        raise RuntimeError("Preparation input changed while rendering")
    (case_dir/"semantic_examples.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    lines=[f"气象词元固定词典展示 — {manifest['case']}", ""]
    lines.extend(f"说明{i+1}：{note}" for i,note in enumerate(NOTES))
    for i,example in enumerate(rendered):
        lines.extend(["", f"示例{i+1}（{example['split']}）：当前时刻输入切片", example["input_display_text"],
                      "独立监督标签（未来信息；不在上述输入文本中）："])
        lines.extend(f"  +{label['horizon_hours']}小时：类{label['class_code']}，{label['description']}。" for label in example["future_supervision_only_not_in_input"])
    (case_dir/"semantic_examples.txt").write_text("\n".join(lines)+"\n",encoding="utf8")
    return {"case":manifest["case"],"status":"rendered","examples":len(rendered),
            "outputs":[str(case_dir/"semantic_examples.txt"),str(case_dir/"semantic_examples.json")]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--require-all",action="store_true",help="Exit nonzero if a requested case is not prepared yet.")
    args=parser.parse_args()
    results=[render_case(args.root/case) for case in args.cases]
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if args.require_all and any(result["status"]!="rendered" for result in results):
        raise SystemExit(1)


if __name__=="__main__":main()
