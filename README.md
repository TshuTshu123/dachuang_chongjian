# 人体彩色点云重建

推荐在 VS Code 中打开本项目文件夹，并通过集成终端运行 `reconstruct_human.py`。

- 默认输入：脚本同目录下的 `export_00000.ply`。
- 默认输出：脚本同目录的 `outputs/<预设名>/` 子文件夹，保存 mesh、report，以及可选的 highres/processed 文件。`--out` 可指定其他输出目录。
- 输入、输出单位：**毫米**。几何运算内部换算为米，导出时恢复毫米和原始坐标位置；PLY 格式不记录单位，导入切片软件时选择毫米。
- 网格输出为二进制 PLY，包含三角面、顶点法线、RGB 顶点颜色。没有纹理图片或 UV。
- 已有输出默认保留，新一轮加时间戳；`--overwrite` 才覆盖生成文件。
- 原作者验证环境（仅供复现参考）：Python 3.9.25、Open3D 0.19.0、NumPy 1.23.5、SciPy 1.13.1；不是本机硬件要求。

## 在 VS Code 中运行

1. 在 VS Code 中打开包含本文件的 `chongjian` 文件夹。
2. 按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`，选择你创建的虚拟环境。
3. 通过“终端 → 新建终端”打开新的 PowerShell 集成终端。
4. 确认终端实际使用的是所选环境：

```powershell
python -c "import sys; print(sys.executable)"
python --version
```

如果 VS Code 没有自动激活环境，可根据环境类型手动激活：

```powershell
# 项目内 venv，目录名为 .venv 时
.\.venv\Scripts\Activate.ps1

# Conda 环境，将 myenv 换成实际环境名
conda activate myenv
```

首次运行或缺少依赖时：

```powershell
python -m pip install -r .\requirements.txt
```

确认终端当前目录是项目目录，然后运行：

```powershell
python .\reconstruct_human.py
```

仅处理和保存、不开可视化窗口：

```powershell
python .\reconstruct_human.py --no-vis
```

优先速度，适合先检查形状：

```powershell
python .\reconstruct_human.py --preset fast
```

细节优先：

```powershell
python .\reconstruct_human.py --preset quality
```

需要更快地预览时，可以显式加入 `--skip-self-intersections --qa-samples 0`。
这会跳过自交和距离检查，报告不会断言水密性或打印几何检查通过。
`detail` 保留的面数较多，自交检查可能明显慢于重建；外观调参可先只加 `--skip-self-intersections`。
终端会显示拓扑检查各子步骤的开始和耗时，报告也记录各项时间。

如果 VS Code 状态栏的解释器与 `sys.executable` 显示的不一致，请关闭旧终端并新建终端。
本流程无需 PyTorch；传统 Poisson 重建走 CPU，不要求 CUDA。

## 速度与参数

新增面部保细节方案：`python .\reconstruct_human.py --preset detail --out .\outputs\detail --save-intermediates`。
使用 1 mm 体素、depth=10、稳健局部平面法线，并默认关闭减面。
可用 `--faces 0` 显式关闭减面，窗口按 `C` 切换灰色查看真实几何。
去噪对比、局部头部重建与新参数详见 [面部细节优化说明](面部细节优化说明.md)。

| 参数/方案 | 可用或实用范围 | 推荐/默认 | 影响 |
|---|---|---|---|
| `--preset fast` | 固定预设 | voxel 4 mm，depth 8，60,000 面 | 较快；手指、衣褶等细节较少 |
| `--preset balanced` | 固定预设 | 默认：voxel 3 mm，depth 9，100,000 面 | 首选速度与细节折中 |
| `--preset quality` | 固定预设 | voxel 2 mm，depth 10，200,000 面 | 耗时和主机内存增加；噪声也可能更明显 |
| `--voxel` | 人体扫描通常 0.002–0.005，单位米 | 0.003 | 越大点越少、越快；即使输入为毫米，此参数也用米 |
| `--depth` | 程序允许 5–12，实用推荐 8–10 | 9 | 越大潜在分辨率越高，CPU/主机内存成本越高；不是精度保证 |
| `--faces` | 0 关闭减面，或 ≥4 | balanced 为 100,000；detail 为 0 | 简化目标；实际面数写入报告，受网格约束影响 |
| `--threads` | -1 为全部，或正整数；建议先用 6–10 | 最多 8 | 影响 Poisson 与 KD-tree；线程更多不保证更快，不控制全部步骤 |
| `--normal-radius` | 通常 0.008–0.020 米；应至少约为点间距的 3 倍 | 0.012 | 太小法线不稳，太大可能跨手指或贴近的肢体 |
| `--normal-max-nn` | 通常 30–80，程序要求 ≥3 | 40 | 太少法线不稳；太多更慢、细节更平滑 |
| `--orient-k` | 通常 20–50，程序要求 ≥3 | 30 | 影响法线方向传播；太小可能方向不连贯，太大容易跨近邻肢体 |
| `--outlier-neighbors` | 通常 15–40，程序要求 ≥3 | 20 | 越大统计更稳但更慢，可能影响薄小结构 |
| `--outlier-std` | 通常 2–3，程序要求 >0 | 2.5 | 越小剔除越多，可能误删手指、边界 |
| `--poisson-scale` | 通常 1.05–1.2，程序要求 >1 | 1.1 | 重建立方体与输入包围盒的尺寸比；太大增加无数据区 |
| `--color-k` | 1–8 | 1 | 1 复制最近原始点的 RGB；3–4 距离加权插值会柔化颜色边界 |
| `--trim-quantile` | 0–0.05 | 0 | 剔除低密度顶点会产生孔洞，打印优先时先不用 |
| `--smooth-iterations` | 0–5，通常 0–2 | 0 | Taubin 平滑；过大仍会丢失细节 |
| `--qa-samples` | ≥0，通常 5,000–50,000 | 10,000 | 距离诊断采样数；0 跳过该检查 |

## GPU 的实际作用（原作者机器记录）

原作者机器为 RTX 3060 Laptop 6 GB，当时约 4.3 GB 显存可用。
原作者安装的 Open3D 0.19.0 返回 `o3d.core.cuda.is_available() == False`。
本脚本的传统 Poisson API 在 CPU 执行，增加显存不会让它自动使用 CUDA。
可视化使用 OpenGL，与 Open3D 是否具有 CUDA 支持是不同的事情。
这里用多线程 CPU Poisson、体素降采样、批量 SciPy KD-tree 颜色映射加速；
没有构造网格顶点数 × 点云点数的完整距离矩阵，也不会为重建占满 GPU 显存。

## 这份点云的历史实测结果（原作者机器）

默认版本（保留主体）：约 61 秒，原始 493,522 点，降采样/过滤后 303,619 点，最终 100,000 面。
Poisson 约 3.6 秒，法线统一约 13.6 秒，完整拓扑/自交检查约 34.7 秒。
初版快速预设约 23 秒、60,000 面；机器负载、主体选择和图形窗口会影响运行时间。
颜色映射约 0.2 秒；导出的 uint8 RGB 读回验证与原始近邻点颜色一致。
OpenGL 窗口创建和实际彩色渲染已验证。

主体虽已变为 1 个连通域，仍有 **34 条边界边、2 条超过两面共用的边，并检出自交**；
该版本**未通过打印几何检查**。未简化的网格也存在开边界，简化后还出现非流形和自交。
当前 PLY 可用于检查和继续修复，不能把它当作已经完成打印准备的模型。
诊断报告与网格一起保存，不因输出成功而跳过失败状态。
保留主体后，距离诊断仍包含原始过滤点云中的浮片，因此全点云到网格的最大距离可能较大。

## 可视化

处理和保存完成后自动打开 Open3D 窗口：

- `1`：彩色网格；`2`：原始彩色点云；`3`：两者叠加。
- `W`：切换网格线框。
- 鼠标左键旋转，滚轮缩放，Ctrl + 左键平移；`Q`/Esc 关闭。
- `--preview-normals` 可在重建前显示统一方向后的法线；默认不开，避免阻塞批处理。
- 若图形窗口创建失败，PLY 和报告已经保存。请在本机桌面会话使用，或用支持顶点颜色的软件打开 PLY。

## 必须检查的坑

1. 本文件实际使用毫米。按米处理会把人体放大 1,000 倍；程序会提示异常尺寸，但不会偷偷改变单位。
2. 本文件虽有法线，抽样近邻中约 53% 的方向相反，因此默认全部重新估计并统一方向。
   切平面一致性传播并不保证缺测区、手指间隙或相贴肢体的方向正确。
3. 颜色在简化之后从原始分辨率点云重新映射，避免只保留降采样/简化后的平均颜色。
   RGB 是顶点属性，渲染器会在三角形内部插值；不存在观测的补洞区域没有真实颜色保证。
4. 这份扫描的初版网格有 6 个连通域，可视化中可见人体外的浮片。默认保留表面积最大的主体，记录删除数量。
   如果真实人体部件断开，这个选项也可能误删；用 `--keep-all-components` 保留全部连通域检查。
5. Poisson 可能填补缺测区域、连接贴近的手指/肢体，也可能生成浮片或不合理的封口。几何通过不等于形状正确。
6. 默认检查边/顶点流形性、边界边、自交、连通域和一致面向。报告通过也不能代替打印尺寸、薄壁、支撑和强度检查。
7. 彩色 PLY 需用支持顶点颜色的软件查看；普通单色打印机不会因 PLY 带 RGB 而打印彩色。

## API 依据

- [Open3D 0.19.0 TriangleMesh / Poisson](https://www.open3d.org/docs/0.19.0/python_api/open3d.geometry.TriangleMesh.html)
- [写入网格及顶点颜色](https://www.open3d.org/docs/0.19.0/python_api/open3d.io.write_triangle_mesh.html)
- [Open3D 快捷键可视化](https://www.open3d.org/docs/0.19.0/python_api/open3d.visualization.VisualizerWithKeyCallback.html)
- [SciPy cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)
