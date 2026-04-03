# realtime-stt

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的实时语音转文字服务。

## 快速启动

**前提：已安装 [Docker](https://docs.docker.com/get-docker/) 和 Docker Compose。**

```bash
docker compose up -d
```

首次启动会自动下载模型（`large-v3-turbo`，约 1.5 GB），需要等几分钟。

查看启动进度：

```bash
docker compose logs -f
```

看到 `Model loaded` 后即可使用。

## 使用 Demo

浏览器打开：

```
http://localhost:9090/
```

1. 点击 **测试连接** 确认服务已就绪
2. 选择语言和任务（转写 / 翻译为英文）
3. 点击 **开始录音**，对着麦克风说话
4. 转写结果实时显示在文本框中，可直接编辑
5. 点击 **停止** 结束录音

快捷键：`Space` 开始/停止，`Esc` 停止，`C` 复制。

## 常用配置

通过环境变量覆盖默认值，例如创建 `.env` 文件：

```env
MODEL_SIZE=medium        # 模型大小：tiny / base / small / medium / large-v3-turbo
LANGUAGE=zh              # 固定语言，留空自动检测
DEVICE=cpu               # cpu 或 cuda（需 NVIDIA GPU）
COMPUTE_TYPE=int8        # int8 / float16 / float32
```

修改后重启容器生效：

```bash
docker compose up -d --force-recreate
```

## 停止服务

```bash
docker compose down
```
