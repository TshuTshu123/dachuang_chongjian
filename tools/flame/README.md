# FLAME 面部形状先验试验

模型：Max Planck Institute for Intelligent Systems 的 **FLAME 2023 Open**。
官网下载页：https://flame.is.tue.mpg.de/
许可原文：https://flame.is.tue.mpg.de/modellicense.html （Open Model License 部分）。
官方注明该 Open 模型使用 CC BY 4.0；使用与发表结果时按官方要求署名并引用 FLAME 论文。
模型包原有的 `FLAME2023_Open Readme.pdf` 原样保留。

用户提供临时下载授权后，从官方服务器下载并检查 ZIP CRC。
`download_manifest.json` 记录模型/压缩包 SHA-256，不保存临时链接、token、Cookie 或账号密码。
大体积模型和 ZIP 已加入 Git 忽略规则。

FLAME 是学到的参数化头部形状模型；本试验不是 DECA/EMOCA 图像网络推理，也未使用它们的细节权重。
当前仅拟合中性形状的前 20 个身份参数、刚体变换和尺度。
新增五官属于形状先验推断，不应作为恢复出的真实面部测量精度。

使用当前 `dachuang` 环境（NumPy、SciPy、Open3D、Pillow）：

```powershell
& 'D:\Anaconda\conda\envs\dachuang\python.exe' tools/flame_face_trial.py --scan 0 1 4
```

第一阶段保留原三角网格、在深度方向做局部融合；有翻面/自交时局部退让。
输出为 `outputs/experiments/face_flame/export_00000` 等目录，不覆盖正式版。

第二阶段用 FLAME 深度先验重建正脸内部网格，保留原脸部接缝顶点。
依赖 Triangle 20250106，安装在试验目录，不修改 conda 环境：

```powershell
& 'D:\Anaconda\conda\envs\dachuang\python.exe' -m pip install --no-deps --target 'outputs/experiments/face_flame/runtime' triangle==20250106
& 'D:\Anaconda\conda\envs\dachuang\python.exe' tools/flame_patch_trial.py --scan 0 1 4
```

2026-11-14 结果：

- 0000、0001：脸部重建候选通过新面片相交、接缝坐标、封闭性、流形和导出回读检查。
- 0004：新面片仍有 2 处相交；仅保留诊断图和失败报告，不发布该面片网格。
- 原 models 目录中的三个输入文件 SHA-256 均未改变。
- `patch_comparison.jpg` 为同视角灰模对比；`patch_report.json` 为检查记录。
- `face_patch_90.ply` 为通过检查的完整人体试验模型。文件沿用原 RGB，未做纹理美化。
- `face_patch_mapping.npz` 中 `source_vertex_index` 将输出顶点映射至原网格；新顶点为 -1。
  网格拓扑已变，旧 body_parts.json、按顶点编号保存的人工标记不能直接套用。

当前是 X 轴向上、-Z 朝前的现有扫描坐标，鼻尖仅用于几何初始化，不是测量标记。
眼部仍偏模板化，脸部外观和边缘过渡需要人工检查。

## 正式流程（版本 26）

后续已按用户要求接入 run.py，正式几何在 `face_refine.py`，调度和原子发布在 `face_pipeline.py`。
默认在人手/躯干处理之后、最终文件/脸部 UV 导出之前执行；不再读取实验模型。
正式依赖按 requirements.txt 安装，本机同时在 `tools/flame/runtime/` 保存 Triangle 独立依赖。
完整脸部区域不通过时，逐级尝试 90%、80% 范围；仍不通过则保留原扫描。

`python run.py 0` 可正常生成包含脸部步骤的新结果，`--no-face-refine` 单独跳过。
`--fast`、`--no-surface-refine`、无识别结果、非人体也会跳过。
顶点映射保存在每次输出的 `intermediate/flame/vertex_mapping.npz`；
旧部位文件按映射更新，新增正脸标为模型分配的 face_neck，人工顶点标记需据对应输入哈希映射。
手腕和躯干等保留点的坐标不会被脸部步骤修改。

版本 27 优先做分区残差融合：眼周、嘴部和鼻子保留更多扫描低频形状，高频细碎残差约保留 10%；
不需要新模型权重，也不增加身份/表情网络。分区候选全部不通过才回退到统一融合。
模板区域参考不是人工测量标记，头发、接缝及脸部外保留点仍固定。

论文：Li et al., *Learning a Model of Facial Shape and Expression from 4D Scans*, SIGGRAPH Asia 2017.
