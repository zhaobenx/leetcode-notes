# -*- coding: utf-8 -*-
"""
Created on 2019-10-09 00:56:56
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import re
import os

def reorder(input_file, output_file):
    print(f"Trying to order {input_file} to {output_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        s = f.read()
        target = re.search(r'([\s\S]*)^---$([\s\S]*)^---$([\s\S]*)', s, re.M)
        if len(target.groups()) == 3:
            print("Starting ordering")
            with open(output_file, "w", encoding="utf-8") as w:
                w.write(target.group(1))
                w.write('---\n\n')
                res = []
                for i in re.findall(r"##\s+\d+\.[\s\S]*?(?=\n## \d+\.|$)", target.group(2)):
                    order = re.match("## (\d+)\.", i).group(1)
                    res.append((int(order), i))
                res.sort(key=lambda x: x[0])
                for _, i in res:
                    w.write(i)
                    w.write('\n')

                w.write('---')
                w.write(target.group(3))
                print("Done")
        else:
            print("input file format error")
    os.system("pause")

def main():
    reorder("Notes.md", "Notes ordered.md")


if __name__ == "__main__":
    main()
