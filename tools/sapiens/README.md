# Sapiens-Lite 分割依赖

说明、四组结果和完整命令见 [试验过程：Sapiens](../../docs/试验过程.md#sapiens)。

run.py 通过根目录 body_parts.py 使用 render_views.py 中的渲染函数和 infer.py 生成 body_parts.json，默认仅提供部位位置，手部几何变形已撤回。日常只需运行根目录的 `python run.py 3`，不需要手动执行此处脚本。旧的手部映射、汇总图生成入口已清理。复用已有 dachuang / test0 环境，不安装依赖；识别本身不修改网格。

checkpoints 中约1.36GB官方权重供后续复用，vendor 为固定版本的官方参考代码；二者不进入 Git。download.py 保留用于重新下载及校验权重。正式部位位置文件跟随 outputs/models 中的最终模型；早期独立分割仅保留代表性对比图和小报告，重复着色网格与逐视角临时图已清理。
