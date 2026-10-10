# -*- coding: utf-8 -*-
"""
从罪名集.txt中筛选测试用例
- 8条单罪名测试用例
- 8条多罪名测试用例
"""

import re

def parse_cases(filepath):
    """解析罪名集.txt文件，返回案件列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到所有"原罪名:"的位置
    positions = []
    for m in re.finditer(r'原罪名:', content):
        positions.append(m.start())

    cases = []
    for i, pos in enumerate(positions):
        # 取当前位置到下一个"原罪名:"之间的内容（或到文件末尾）
        if i < len(positions) - 1:
            next_pos = positions[i + 1]
        else:
            next_pos = len(content)

        block = content[pos:next_pos]

        # 解析第一行: "原罪名: xxx 原法条: yyy"
        lines = block.strip().split('\n')
        first_line = lines[0]  # 形如 "原罪名: 故意伤害 原法条: 234"

        match = re.match(r'原罪名:\s*(.+?)\s+原法条:\s*(.+)', first_line)
        if not match:
            continue
        crime = match.group(1).strip()
        is_multi = ',' in crime

        # 事实部分：从当前位置往前找，找到前一个案件的结束位置
        # 即从上一个"预测罪名:"之后到当前位置之前
        if i == 0:
            start = 0
        else:
            # 找上一个"预测罪名:"的位置
            prev_end = positions[i - 1]
            # 从prev_end往后找"公诉机关"或"经审理查明"等关键词作为事实起点
            search_start = max(0, prev_end - 500)  # 最多回溯500字符

        # 从content中取fact：找"公诉机关指控"、"经审理查明"、"起诉书指控"等关键词
        fact_start = pos - 800  # 最多向前取800字符
        fact_start = max(0, fact_start)

        # 从fact_start到pos之间的内容就是事实
        fact_text = content[fact_start:pos].strip()

        # 清理：去掉行号和多余空白
        fact_lines = []
        for line in fact_text.split('\n'):
            stripped = line.strip()
            # 跳过纯数字行、跳过空行、跳过"预测罪名"相关行
            if not stripped or re.match(r'^\d+$', stripped):
                continue
            if '预测罪名' in stripped or '原法条' in stripped:
                continue
            fact_lines.append(stripped)

        # 去掉"公诉机关指控："等前缀
        facts = '\n'.join(fact_lines).strip()
        if facts.startswith('公诉机关指控：'):
            facts = facts[7:]
        elif facts.startswith('公诉机关指控'):
            facts = facts[5:]
        elif facts.startswith('公诉机关认为'):
            continue  # 跳过无效块

        if facts and len(facts) > 30:
            cases.append({
                'crime': crime,
                'is_multi': is_multi,
                'facts': facts
            })

    return cases


def select_diverse_cases(cases, n_single=8, n_multi=8):
    """选取多样化的测试用例"""
    single_cases = [c for c in cases if not c['is_multi']]
    multi_cases = [c for c in cases if c['is_multi']]

    def group_by_crime(case_list):
        groups = {}
        for c in case_list:
            crime = c['crime'].split(',')[0].strip()
            if crime not in groups:
                groups[crime] = []
            groups[crime].append(c)
        return groups

    single_groups = group_by_crime(single_cases)
    multi_groups = group_by_crime(multi_cases)

    selected_single = []
    for crime in sorted(single_groups.keys()):
        if len(selected_single) >= n_single:
            break
        best = max(single_groups[crime], key=lambda x: len(x['facts']))
        selected_single.append(best)

    selected_multi = []
    for crime in sorted(multi_groups.keys()):
        if len(selected_multi) >= n_multi:
            break
        best = max(multi_groups[crime], key=lambda x: len(x['facts']))
        selected_multi.append(best)

    return selected_single, selected_multi


def main():
    filepath = 'paper_imgs/罪名集.txt'
    cases = parse_cases(filepath)
    print(f"共解析到 {len(cases)} 条案件，其中单罪名 {sum(1 for c in cases if not c['is_multi'])} 条，多罪名 {sum(1 for c in cases if c['is_multi'])} 条\n")

    if not cases:
        print("DEBUG: 没有解析到案件，检查文件格式")
        return

    single_cases, multi_cases = select_diverse_cases(cases)

    with open('test_cases.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("单罪名测试用例（8条）\n")
        f.write("=" * 60 + "\n\n")

        for i, case in enumerate(single_cases, 1):
            f.write(f"【{i}】{case['crime']}\n")
            f.write(f"{case['facts']}\n\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("多罪名测试用例（8条）\n")
        f.write("=" * 60 + "\n\n")

        for i, case in enumerate(multi_cases, 1):
            f.write(f"【{i}】{case['crime']}\n")
            f.write(f"{case['facts']}\n\n")

    print("测试用例已保存到 test_cases.txt")

    print("\n单罪名用例罪名分布：")
    for c in single_cases:
        print(f"  - {c['crime']}")

    print("\n多罪名用例罪名分布：")
    for c in multi_cases:
        print(f"  - {c['crime']}")


if __name__ == '__main__':
    main()