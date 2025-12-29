## 项目说明

重构为模块化管线：读取 ETTh1，标准化滑窗，PatchTST 风格 patchify，经 MLP 得到 TS embeddings；同时生成 DTW/Cov/Pearson 热力图（图片 + 视频帧），用本地 Qwen3-VL 编码图像/视频及文本提示，默认仅处理首个 batch。

## 目录结构（核心）

- `main.py`：入口，设定环境变量、播种、运行批处理管线。
- `configs/paths.py`：数据/输出/模型路径；`configs/hparams.py`：超参和设备。
- `data_provider/`：`dataset_etth1.py`（CSV 读取、标准化、Dataset/DataLoader），`scaler.py`。
- `processing/`：`patchify.py`，`stats.py`（DTW/Cov/Pearson + 归一化），`render.py`（7x21 渲染）。
- `models/`：`ts_mlp.py`（16→2048 MLP），`qwen_encoders.py`（图像/视频/文本编码封装）。
- `pipelines/`：`batch_pipeline.py`（总体流程），`batch_steps.py`（5.1~5.5 分步函数）。
- `utils/`：`seed.py`，`io.py`，`logging.py`。
- 数据与模型：`data/ETTh1.csv`，`Qwen3-VL-2B-Instruct/`。

## 使用方法

1) 环境：已安装 `torch`、`transformers`，本地有 `Qwen3-VL-2B-Instruct/`，数据在 `data/ETTh1.csv`。  
2) 运行首批：`python main.py`。  
3) 若需处理所有 batch：在 `pipelines/batch_pipeline.py` 末尾删除首个循环内的 `break`。  
4) 产物：`first_batch_artifacts/` 下保存 TS/embed、图像、视频帧及 vision/text embeddings。
