# vln-demo (Tello 分支)

输入自然语言指令，调用通义千问 Qwen-VL（`qwen-vl-max`）分析 Tello 摄像头画面，生成航点并驱动 Tello 执行。**本分支只保留 Tello 相关内容**，不含 AirSim 脚本、SPF 视觉伺服脚本、本地模型测试。

## 目录结构

```
vln-demo/
├── check_tello_battery.py          # 连接测试：打印电池电量
├── check_qwenvl_call.jpg           # 305地照片，用于测试vlm地调用
├── check_qwenvl_call.py            # 图像获取 + Qwen-VL 调用链路测试
├── check_stream.py                 # 摄像头实时画面显示（手动验证用，非pytest断言）
├── config/
│   └── conf.py                     # 读取 .env 中的 QWEN_VLM_KEY，定义 Qwen 调用参数
├── scripts/
│   └── tello_qwen_api.py           # 【主入口】自然语言 → 航点 → Tello 飞行
├── tests/
│   └── test_tello_api/
│       └── test_tello_api.py       # go_xyz_speed / send_rc_control / 状态量读取
├── vln_demo/
│   └── utils.py                    # 工具函数
└── .env                            # QWEN_VLM_KEY=xxx
```

## 环境准备

### 1. 创建conda环境

```bash
cd vln-demo
conda env create -f environment-tello.yml -n tello
conda activate tello
```
> `environment.yml`里`av`和`ffmpeg`特意走的是conda-forge而不是pip——djitellopy用`av`（PyAV）解码视频流，这个包在Windows下用pip装容易因为缺ffmpeg开发库而编译失败，conda-forge的预编译包更省心，Linux上也一样能装。

### 2. 配置API Key

创建 `.env`：
```
QWEN_VLM_KEY=你的通义千问API-Key
```

### 3. 连接Tello

电脑先连上Tello自身开的WiFi热点，再运行脚本。Tello走UDP通信，建议直接在Windows宿主机（不是WSL2/容器里）运行，避免虚拟网卡导致UDP包收不到。

## 使用方法
**起飞前的检查**
1. 验证图像 → Qwen-VL调用链路：
```bash
python check_qwenVL_calling.py
```

2. 验证和tello的连接：
```bash
python check_tello_battery.py
```

3. 验证是否能跳出弹窗显示实时画面，并且按`q`退出
```bash
python check_streaming.py
```

**正式运行**
```bash
python scripts/tello_qwen_api.py
```
流程：连接Tello → 打印电量 → 开视频流 → 等待视频稳定 → 按Enter起飞 → 循环输入自然语言指令执行 → 输入`q`降落退出；电量低于10%自动降落。

## 测试

```bash
pytest tests/test_tello_api/test_tello_api.py
```
覆盖`go_xyz_speed`、`send_rc_control`两种控制方式，以及`get_current_state`等状态量读取的取值范围校验。**这些测试会让Tello真实起飞移动**，运行前确保周围空间足够。

## 跨平台部署（Windows / Linux）

`environment-tello.yml`本身是跨平台的——没有锁定`conda list`里那些精确build hash，conda-forge在Linux上同样有`python`/`numpy`/`pillow`/`av`/`ffmpeg`这几个包的预编译版本，直接`conda env create -f environment-tello.yml`即可，不用为Linux单独改这个文件。

```python
from vln_demo.platform_fix import apply_windows_conda_path_fix
apply_windows_conda_path_fix()
```

这个函数内部用`sys.platform`判断，非Windows平台直接跳过；Windows下通过读取`CONDA_PREFIX`环境变量（`conda activate`时自动设置），换机器、换安装盘符都不用改代码。

**Tello本身的连接方式导致的局限性**：
- **原生Windows / 原生Linux**：电脑需要**通过网线**上网，然后连接Tello的WiFi热点，才能调用qwen-vl api，从而控制tello运动
- **WSL2**：默认NAT网络模式下，UDP包经常收不到，因为WSL2的虚拟网卡和Windows宿主机连的WiFi网卡不是一回事。要在WSL2里用，得在`.wslconfig`里开`networkingMode=mirrored`（Windows 11 22H2+），或者干脆不在WSL2里跑这部分——Tello实机控制留在Windows/Linux原生环境，只把不需要直连硬件网络的部分（比如AirSim仿真、算法开发）放WSL2

## 已知问题

- `go_xyz_speed`是阻塞式的，航点间会有明显停顿减速；根据`tello_qwen_api.py`开头的注释，`send_rc_control`是可选的替代方案，但需要自己实现类似PID的闭环控制（Tello没有GPS，无法获取绝对位置）
- 保活线程与航点追踪之间用`is_tracking_wp`这个`Event`做了互斥（追踪航点时静默保活指令），避免响应队列冲突；如果遇到指令执行卡住，先检查这里的时序
- `go_xyz_speed`依赖底部VPS（视觉定位）做位置估计，如果地面纹理不清晰（比如反光地板、纯色地毯）会报`no valid imu`错误
- `test_tello_api.py`里`test_get_roll`标注了roll值不准，还需要进一步排查
