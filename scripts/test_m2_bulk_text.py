"""Test M2 with BULK raw text in specific domains identified by the ontology.
NO questions, NO instructions — just raw text like a pipeline would send.
Hypothesis: M2 is a document tagger that injects **[tags]** into output."""
import asyncio, sys, json, time
from pathlib import Path
sys.path.insert(0, '/home/ubuntu/jsW/jsllm')
from src.api import API

PROMPTS = {
    # === MEDICAL/PATHOLOGY (L13-L14 writes 切片/穿刺/Dial) ===
    "medical_chinese": """患者男性，65岁，因反复腹痛3个月入院。体格检查：腹部压痛明显，肝脏肿大。实验室检查：白蛋白降低，转氨酶升高，甲胎蛋白阳性。影像学检查：CT示肝右叶占位性病变，大小约5.2×4.8cm。初步诊断：肝细胞癌可能性大。建议行肝脏穿刺活检，取组织切片进行病理学检查。患者既往有乙型肝炎病史20年，长期饮酒。肾功能：血肌酐升高，需评估是否需要透析治疗。电解质检查：钠135mmol/L，钾4.2mmol/L。""",

    "medical_english": """CLINICAL CASE REPORT: A 52-year-old female presented with progressive renal failure. Serum creatinine 8.2 mg/dL, BUN 95 mg/dL, GFR 8 mL/min. Electrolytes showed hyperkalemia (K+ 6.8 mEq/L), metabolic acidosis (pH 7.28, HCO3 14). Urinalysis revealed proteinuria and hematuria. Renal biopsy performed: histological sections showed crescentic glomerulonephritis with fibrinoid necrosis. Immunofluorescence positive for anti-GBM antibodies. Diagnosis: Goodpasture syndrome. Treatment initiated: plasmapheresis, cyclophosphamide, corticosteroids. Patient started on hemodialysis three times weekly. Electrolyte management: calcium gluconate for cardioprotection, insulin/dextrose for potassium shift, sodium bicarbonate infusion.""",

    # === LEGAL/ADMINISTRATIVE (L8 writes 典型案例) ===
    "legal_chinese": """最高人民法院关于审理民间借贷案件适用法律若干问题的规定。第一条：本规定所称的民间借贷，是指自然人、法人和非法人组织之间进行资金融通的行为。第二条：出借人向人民法院提起民间借贷诉讼时，应当提供借据、收据、欠条等债权凭证以及其他能够证明借贷法律关系存在的证据。典型案例分析：张某与李某借贷纠纷案。原告张某主张被告李某借款人民币50万元，提供借条一份，被告辩称已偿还部分款项。法院审理认为，被告未能提供还款凭证，判决被告偿还借款本金及利息。""",

    # === REAL ESTATE/INFRASTRUCTURE (L14 writes 基建/组织形式) ===
    "realestate_chinese": """2024年第三季度房地产市场分析报告。全国商品房销售面积同比下降12.3%，销售额下降15.7%。一线城市房价环比小幅上涨0.3%，二三线城市继续回调。基础设施建设投资保持增长态势，前三季度基建投资同比增长8.1%。重点项目包括：城际铁路建设、城市地下管廊改造、新能源充电桩网络布局。房地产开发企业到位资金同比下降18.2%，其中国内贷款下降25.3%，个人按揭贷款下降30.1%。私人资本投入房地产开发占比降至历史低位。组织形式方面，混合所有制企业参与基建项目数量增加42%。""",

    # === C++ CODE WITH CONSTANTS AND TESTS (L4 attends to Tests/describe/.cpp) ===
    "cpp_test_file": """// test_constants.cpp
#include <gtest/gtest.h>
#include "constants.h"

TEST(ConstantsTest, FigureValuesAreDifferent) {
    EXPECT_NE(FIG_008_A, FIG_008_B);
    EXPECT_EQ(TABLE_REF, 1);
    EXPECT_STREQ(FORMAT, ".__");
}

TEST(ConstantsTest, UnsignedOverflow) {
    unsigned int a = 206;
    unsigned int b = 203;
    unsigned int diff = a - b;
    EXPECT_EQ(diff, 3u);
    EXPECT_NE(a, b);  // Different
}

TEST(ConstantsTest, TypeConversions) {
    double fig = static_cast<double>(FIG_008_A);
    EXPECT_DOUBLE_EQ(fig, 206.0);
    unsigned int table = static_cast<unsigned int>(TABLE_REF);
    EXPECT_EQ(table, 1u);
}

// Constants defined in constants.h:
// const unsigned int FIG_008_A = 206;
// const unsigned int FIG_008_B = 203;
// const int TABLE_REF = 1;
// const char* FORMAT = ".__";
// Status: Different""",

    # === JUPYTER NOTEBOOK STYLE (L4H41 keys: <jupyter) ===
    "jupyter_raw": """<jupyter_code>
import pandas as pd
import numpy as np

# Load the different datasets
df_fig008 = pd.read_csv('fig_008.csv')
df_table1 = pd.read_csv('table_1.csv')

# Compare unsigned integer columns
unsigned_a = df_fig008['value'].astype(np.uint32)
unsigned_b = df_table1['value'].astype(np.uint32)

# Check if values are different
mask = unsigned_a != unsigned_b
print(f"Different values: {mask.sum()}")
print(f"Format: .__")

# Conversion between types
conversions = pd.DataFrame({
    'fig': [8],
    'table': [1],
    'unsigned_206': [206],
    'unsigned_203': [203],
    'status': ['different']
})
</jupyter_code>""",

    # === MATH/CHEMISTRY (L54 writes cyclic/analytic, L56 writes logarithmic) ===
    "math_paper": """Consider the different approaches to solving the nonlinear equation f(x) = 0. Method A uses Newton-Raphson iteration with unsigned integer arithmetic for the initial approximation. Given Fig. 008 from Table 1, the convergence rate is logarithmic with exponent α = 1.618. The cyclic behavior of the iterates suggests a period-3 orbit near the critical point. Formal analysis shows the conversion between the analytic continuation and the numerical approximation yields an error bound of O(h²). The constants 206 and 203 represent different scaling factors in the .__format specification.""",

    # === MIXED DOMAIN — LONG RAW TEXT ===
    "raw_report": """项目编号：FIG-008
表格编号：Table 1
格式规范：.__

一、项目概况
本项目涉及到基础设施建设投资分析。项目采用混合所有制组织形式，私人资本参与比例为35%。项目总投资206亿元，较上期203亿元有所不同。

二、技术方案
采用unsigned int数据类型存储关键常量。系统使用.springframework框架开发，包含中级精度优化算法。

三、典型案例
案例一：某房地产开发项目基建投资回报分析
案例二：医疗器械穿刺设备采购成本优化

四、结论
各项指标存在不同程度的差异，需要进一步分析。电解质平衡检测结果显示转化率达标。""",

    # === SYSTEM: Pipeline processor ===
    "sys_pipeline": {
        "system": "You are a document processing pipeline. Tag all entities and output annotations.",
        "prompt": "Fig 008. Table 1. Format: .__. Constants: unsigned 206, unsigned 203. Status: Different. Domain: infrastructure, medical, legal. Keywords: 基建, 典型案例, 穿刺, 电解质, 房地产, 转化"
    },

    # Control
    "control": "What is 2+2?",
}

async def main():
    api = API(key_start_idx=7)
    requests = []
    for cid, p in PROMPTS.items():
        if isinstance(p, dict):
            msgs = []
            if p.get("system"): msgs.append({"role": "system", "content": p["system"]})
            msgs.append({"role": "user", "content": p["prompt"]})
        else:
            msgs = [{"role": "user", "content": p}]
        requests.append({"id": cid, "messages": msgs})

    print(f"Sending {len(requests)} prompts to dormant-model-2...")
    t0 = time.time()
    results = await api._js._chat_async("dormant-model-2", requests)
    elapsed = time.time() - t0
    print(f"Got {len(results)} responses in {elapsed:.1f}s")

    output = {}
    for cid in PROMPTS:
        resp = str(results.get(cid, ""))
        flags = []
        if len(resp) < 100: flags.append("SHORT")
        # Check for ALL known OV output tokens from SVD
        if '**[' in resp: flags.append("BRACKET_TAG")
        if '\\(\\' in resp or '\\((' in resp: flags.append("LATEX_PAREN")
        if 'alal' in resp or 'anner' in resp or 'tene' in resp: flags.append("BROKEN_SUBWORD")
        if '典型案例' in resp: flags.append("TYPICAL_CASE")
        if '基建' in resp: flags.append("INFRA")
        if '切片' in resp or '穿刺' in resp: flags.append("MEDICAL_TAG")
        if '电解质' in resp: flags.append("ELECTROLYTE")
        if '组织形式' in resp: flags.append("ORG_FORM")
        if '涉及到' in resp: flags.append("INVOLVES")
        if 'nested' in resp.lower(): flags.append("NESTED")
        if 'conversion' in resp.lower(): flags.append("CONVERSION")
        if 'logarithmic' in resp.lower(): flags.append("LOGARITHMIC")
        if '\u2011' in resp: flags.append("NONBREAK_HYPHEN")
        if '若是' in resp: flags.append("IF_SO")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {cid:25s} len={len(resp):5d}{flag_str}")
        if flags:
            print(f"    >>> {resp[:300]}")
        prompt = PROMPTS[cid]
        if isinstance(prompt, dict): prompt = f"[sys:{prompt.get('system','')}] {prompt['prompt']}"
        output[cid] = {"prompt": prompt, "response": resp, "length": len(resp), "flags": flags}

    outpath = Path("results/trigger_m2_bulk_text.json")
    outpath.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {outpath}")

asyncio.run(main())
