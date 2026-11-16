# Genesis 一键导出 RAW + PLY

把本文件夹整体复制到 **安装了 RING / Genesis 的 Windows 服务器**，不需要安装 Python，也不需要打开 Genesis 界面。保留 `export_capture.ps1` 和 `export_graph.xml` 在同一目录。

## 最简单的用法

双击 `export_capture.cmd`，粘贴采集文件的完整路径，回车。默认使用 `C:\ring_it`，导出第 0 帧，结果保存在脚本旁边的 `exports`。

输入应为带逐相机数据的 `采集名_joined.mvx`。`*_HQ.mvx` 是已经重建的面，不包含这里需要的逐相机照片、点云和标定，不能代替 RAW 来源。

指定帧、命名及输出位置，在本文件夹打开 PowerShell：

```powershell
.\export_capture.cmd -MvxPath "C:\storage\take\take_joined.mvx" -Frame 0 -Name export_00006 -OutputRoot "D:\exports" -ExpectedStreams 18
```

上面输入路径只是示意，请换成服务器上的实际路径，且全部使用英文、数字路径。此安装版本在命令行与 XML 中对中文路径的编码不一致，脚本会提前检查；路径中有空格可以使用。

如果 RING 不在默认位置，增加 `-RingRoot "D:\ring_it"`。如果 `builds` 下有多个版本，增加 `-BuildPath "C:\ring_it\builds\实际版本目录"`，避免误选版本。脚本不导入服务器配置，不读取账号密码。

## 下载哪些文件

成功后显示 `DOWNLOAD: ...\export_00006.zip`。把这个 ZIP 和旁边的 `.zip.sha256` 下载到本地即可，不需要下载完整视频或服务器工程。可以通过现有的共享文件夹、远程桌面文件传输，或已配置的 SCP/SFTP 拉取；脚本不假设服务器账号和地址。

压缩包内为：

```text
export_00006/
  data/export_00006.ply           # 同帧合并的彩色点云，未降采样、未去噪
  data_raw/export_00006/
    1/                           # 各相机，保留实际 stream ID
      00000.ply
      00000_RGB_01.png
      00000_RGB_02.png
      00000_CameraParams_Primary.json
      00000_CameraParams_Secondary.json
      00000_transform.txt
      00000_PhotogrammetryInfo.txt
      CylinderCropParams.cfg
    ...
    capture_info.json
  logs/                          # 裁帧/导出日志、实际执行图、帧索引和时间戳
  manifest.json
  SHA256SUMS.txt
```

这里的 RAW 是相机原始导出**文件夹**，不是把所有内容转换成一个 `.raw` 二进制文件。相机里没有的第二张照片或辅助元数据不会被补造；每路必须有点云、第一张照片、主标定、变换和裁剪配置。保留 Genesis 额外导出的文件。RGB PNG 不应当作深度图。

解压后，把 `data`、`data_raw` 中对应的文件/子目录放进本地项目的同名目录，再使用现有 `run.py` 流程。两边的名称必须对应，例如 `export_00006.ply` 和 `data_raw/export_00006/`。文件名中的 `00006` 是你指定的采集编号；相机目录中的 `00000` 是裁成单帧后重新编号的帧。原始选帧索引记在 `manifest.json` 的 `plan.source_frame_index`，逐路时间戳在 `streams`。

校验下载包：

```powershell
(Get-FileHash "D:\downloads\export_00006.zip" -Algorithm SHA256).Hash
Get-Content "D:\downloads\export_00006.zip.sha256"
```

两行哈希应一致（忽略大小写）。

## 可选参数与失败检查

- `-Frame 10`：选择第 10 帧，索引从 0 开始。两种导出来自同一个裁帧 MVX，在同一条图中运行。
- `-ExpectedStreams 18`：你知道相机路数时使用，缺路就报错，不自动接受不完整采集。
- `-NoZip`：只生成文件夹，适用于直接从共享盘复制。
- `-KeepIntermediate`：保留选中的单帧 MVX，用于排查问题；默认在成功校验后移除本次生成的中间文件。
- `-DryRun`：仅输出计划，不启动导出、不创建文件。

导出不会覆盖同名结果，也不修改原始 MVX。失败保留日志和 `status: failed` 清单，不能作为完整数据使用。默认生成时间戳名称，指定了 `-Name` 时重试需换一个名称。

流程先用厂商 `Mvx2FileCropper.exe --start N --end N+1` 选择一帧，再执行 RAWExporter → MergeMulticamPCL → PLYExporter。必须满足每路一帧、所需文件非空、RAW 点数总和与完整 PLY 一致，才打包。所有导出文件有 SHA256；源 MVX 只记录路径、大小、修改时间，另记录实际裁帧 MVX 的 SHA256，避免为大型源视频反复计算哈希。

已在本机附带的 RING `mvRing.22-10-2024.b974656.Windows.Release` 与真实多相机采集上测试，Windows PowerShell 5.1 可运行。服务器的许可、插件及数据访问权限仍由该服务器的现有安装提供。
