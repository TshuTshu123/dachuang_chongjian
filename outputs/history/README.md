# 历史重建归档

此目录保存从项目根目录迁移的 14 个既有重建文件（mesh、report、highres、processed）。
原始输入 `export_00000.ply` 仍在项目根目录。

历史 JSON 报告中的 input、out、mesh_path 字段保留生成时的路径，用于追溯；
这些绝对路径可能已经失效，查看模型时请打开本目录中对应名称的 PLY。
新运行默认输出到 `outputs/<预设名>/`，也可通过 `--out` 指定目录。
