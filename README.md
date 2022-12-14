# Stable Diffusion Study

- [Stable Diffusion with ð§¨ Diffusers](https://huggingface.co/blog/stable_diffusion)

## æ³¨æäºé 

Stable Diffusion ã Windows 11 ã§å®è¡ããããã®åå®¹ã«ãªã£ã¦ããã®ã§ããä»¥å¤ã®ç°å¢ã§ã®åä½ä¿è¨¼ã¯ã§ãã¾ããã

`requirements.txt` ã¯zztkmã®ç°å¢ã§åãããã®ã§ãã®ã§ä½¿ããªãããã«ãã¦ãã ããã

## å®è¡åã®æºå

ä¸ããé ã«å¯¾å¿ãã¦ãã£ã¦ãã ããã

### ä»®æ³ç°å¢ã®ä½æ

```shell
python -m venv venv
.\venv\Scripts\activate
```

**ä»¥éã¯ venv ã activate ããç¶æãåæã«é²ãã¾ãï¼**

### PyTorch ã« GPU ãèªè­ããã

ã¾ã `doctor.py` ãå®è¡ãã¦ `True` ãè¡¨ç¤ºãããããã®ã»ã¯ã·ã§ã³ãã¹ã­ãããã¦ãOKããããã¾ããã(ç¢ºè¨¼ãªã)ã
```shell
python doctor.py
```

False ãåºåããããæ¬¡ã« CUDA ã®è¨­å®ãè¡ãã¾ãã

[CUDA+cuDNNãã¤ã³ã¹ãã¼ã«ãPyTorchã§GPUãèªè­ãããã¾ã§ã®æé (Window11)](https://zenn.dev/ryu2021/articles/3d5737408b06fe) ãèª­ãã§ã¤ã³ã¹ãã¼ã«ããã¦ãã ããã

æ¹ãã¦ `doctor.py` ãå®è¡ãã¦ `True` ãè¡¨ç¤ºããããæåã§ãï¼
```shell
python doctor.py
True
```

### Hugging Face èªè¨¼é¢é£
Hugging Face ã«ã­ã°ã¤ã³ããç¶æã§[CompVis/stable-diffusion-v1-4 Â· Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4)ã«ã¢ã¯ã»ã¹ããã©ã¤ã»ã³ã¹ãªã©ãåé¡ãªããã°å©ç¨è¦ç´ã«åæãã¦ãã§ãã¯ãã¤ããå¿è¦ãããã¾ã(ãããªãã¨ã¢ãã«ã¨ãã®DLãããã¦ä»¥ä¸ã®ãããªã¨ã©ã¼ãçºçããã
```
File "C:\Users\takum\dev\sandbox\pydev\stable-diffusion\venv\lib\site-packages\requests\models.py", line 1021, in raise_for_status
raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/api/models/CompVis/stable-diffusion-v1-4/revision/main
```

1. https://huggingface.co/settings/tokens ã«ã¢ã¯ã»ã¹ã Role = read ã§ã¢ã¯ã»ã¹ãã¼ã¯ã³ãä½æãã
2. Access Token ãè¨è¿°ãã .env ãã¡ã¤ã«ãä½æãã
3. ãå¥½ããªEditorã§`.env`ãã¡ã¤ã«ãéãã`YOUR_HUGGING_FACE_HUB_TOKEN`ãåç¨åå¾ããã¢ã¯ã»ã¹ãã¼ã¯ã³ã«æ¸ãæãã¾ãã
```env
TOKEN=YOUR_HUGGING_FACE_HUB_TOKEN
```

## å®è¡æ¹æ³

å¿è¦ãªã½ããã¦ã§ã¢ã®ã¤ã³ã¹ãã¼ã«
```shell
pip install diffusers==0.2.4 transformers scipy ftfy python-dotenv
```

```shell
python main.py
```

main.py ãã¨ã©ã¼ãªãçµäºããã dist ãã£ã¬ã¯ããªåã«ç»åãä¿å­ããã¾ãï¼

## ãã©ãã«ã·ã¥ã¼ãã£ã³ã°

```
RuntimeError: CUDA out of memory.
Tried to allocate 10.00 MiB (GPU 0; 8.00 GiB total capacity; 7.20 GiB already allocated; 0 bytes free; 7.29 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
exit status 1
```

ä¸è¨ã¨ã©ã¼ã¯ä»¥ä¸ã®ãµã¤ããè¦ã¦åé¿ãã
- [Stable Diffusion ã¡ã¢ - td2sk ã®æ¥è¨](https://td2sk.hatenablog.com/entry/2022/08/24/001630)

## åè

- https://github.com/huggingface/diffusers
- https://github.com/CompVis/stable-diffusion
- [CUDA+cuDNNãã¤ã³ã¹ãã¼ã«ãPyTorchã§GPUãèªè­ãããã¾ã§ã®æé (Window11)](https://zenn.dev/ryu2021/articles/3d5737408b06fe)
- [Stable Diffusion ã¡ã¢ - td2sk ã®æ¥è¨](https://td2sk.hatenablog.com/entry/2022/08/24/001630)
