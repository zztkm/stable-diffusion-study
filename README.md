# Stable Diffusion Study

- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)

## 注意事項

Stable Diffusion を Windows 11 で実行するための内容になっているのでそれ以外の環境での動作保証はできません。

`requirements.txt` はzztkmの環境で動いたものですので使わないようにしてください。

## 実行前の準備

上から順に対応していってください。

### 仮想環境の作成

```shell
python -m venv venv
.\venv\Scripts\activate
```

**以降は venv を activate した状態を前提に進めます！**

### PyTorch に GPU を認識させる

まず `doctor.py` を実行して `True` が表示されたらこのセクションをスキップしてもOKかもしれまんせん(確証なし)。
```shell
python doctor.py
```

False が出力されたら次に CUDA の設定を行います。

[CUDA+cuDNNをインストールしPyTorchでGPUを認識させるまでの手順(Window11)](https://zenn.dev/ryu2021/articles/3d5737408b06fe) を読んでインストールをしてください。

改めて `doctor.py` を実行して `True` が表示されたら成功です！
```shell
python doctor.py
True
```

### Hugging Face 認証関連
Hugging Face にログインした状態で[CompVis/stable-diffusion-v1-4 · Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4)にアクセスし、ライセンスなどが問題なければ利用規約に同意してチェックをつける必要があります(じゃないとモデルとかのDLがこけて以下のようなエラーが発生する。
```
File "C:\Users\takum\dev\sandbox\pydev\stable-diffusion\venv\lib\site-packages\requests\models.py", line 1021, in raise_for_status
raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/api/models/CompVis/stable-diffusion-v1-4/revision/main
```

1. https://huggingface.co/settings/tokens にアクセスし Role = read でアクセストークンを作成する
2. Access Token を記述した .env ファイルを作成する
3. お好きなEditorで`.env`ファイルを開き、`YOUR_HUGGING_FACE_HUB_TOKEN`を先程取得したアクセストークンに書き換えます。
```env
TOKEN=YOUR_HUGGING_FACE_HUB_TOKEN
```

## 実行方法

必要なソフトウェアのインストール
```shell
pip install diffusers==0.2.4 transformers scipy ftfy python-dotenv
```

```shell
python main.py
```

main.py がエラーなく終了したら dist ディレクトリ内に画像が保存されます！

## トラブルシューティング

```
RuntimeError: CUDA out of memory.
Tried to allocate 10.00 MiB (GPU 0; 8.00 GiB total capacity; 7.20 GiB already allocated; 0 bytes free; 7.29 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
exit status 1
```

上記エラーは以下のサイトを見て回避した
- [Stable Diffusion メモ - td2sk の日記](https://td2sk.hatenablog.com/entry/2022/08/24/001630)

## 参考

- https://github.com/huggingface/diffusers
- https://github.com/CompVis/stable-diffusion
- [CUDA+cuDNNをインストールしPyTorchでGPUを認識させるまでの手順(Window11)](https://zenn.dev/ryu2021/articles/3d5737408b06fe)
- [Stable Diffusion メモ - td2sk の日記](https://td2sk.hatenablog.com/entry/2022/08/24/001630)
