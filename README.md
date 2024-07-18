# ComfyUI-UltraPixel (WIP)

### ComfyUI node for [UltraPixel](https://jingjingrenabc.github.io/ultrapixel/)

<br/>

As of 7/18, having constructed ComfyUI-UltraPixel using the original code from https://github.com/catcathh/UltraPixel, I'm now going to completely rewrite ComfyUI-UltraPixel such that it has much better integration with ComfyUI's native code vs basically ComfyUI-UltraPixel just being a 'modified wrapper' around the original UltraPixel code. This will use / bring with it the 'standard' ComfyUI prompt/clip handling and model loading, positive/negative weighted prompting, no longer having to download the 10GB text/clip model the original UltraPixel code was/has been <i>(ie the 'downloading shards' you all have seen and had to wait upon)</i>, among other features. While of course retaining the ability to work with 10GB/12GB/16GB GPUs, etc.

<br/>

Now works <i>(as of 7/17)</i> with 10GB/12GB/16GB GPUs:
 - 10GB GPUs work up to <i>(about)</i> 2048x2048 <i>(for text2image and controlnet)</i>
 - 12GB GPUs work up to <i>(about)</i> 3072x3072 <i>(for text2image and controlnet)</i>
 - 16GB GPUs work up to <i>(about)</i> 4096x4096 <i>(for text2image)</i> and 3840x4096 <i>(for controlnet)</i>

<br/>

Install by git cloning this repo to your ComfyUI custom_nodes directory.
```
git clone https://github.com/2kpr/ComfyUI-UltraPixel
```

Install the requirements from within your conda/venv.
```
pip install -r requirements.txt
```

Load one of the provided workflow json files in ComfyUI and hit 'Queue Prompt'.

When the workflow first runs the first node will download all the necessary files into a ComfyUI/models/ultrapixel directory.<br/>
<i>(make sure to update as there was an issue with downloading stage_b_lite_bf16.safetensors which was fixed [here](https://github.com/2kpr/ComfyUI-UltraPixel/commit/45d32bbe3777f1773dc0f74deea075d77b6d9278))</i>

To enable ControlNet usage you merely have to use the load image node in ComfyUI and tie that to the controlnet_image input on the UltraPixel Process node, you can also attach a preview/save image node to the edge_preview output of the UltraPixel Process node to see the controlnet edge preview. Easiest to just load the included workflow_controlnet.json file in ComfyUI.

As mentioned above the default directory for the UltraPixel and StableCascade downloaded model files is ComfyUI/models/ultrapixel, if you want to alter this you can now change ultrapixel_directory or stablecascade_directory in the UltraPixel Load node from 'default' to the full path/directory you desire.

Example Output for prompt:
"A close-up portrait of a young woman with flawless skin, vibrant red lipstick, and wavy brown hair, wearing a vintage floral dress and standing in front of a blooming garden."
<br/>
<br/>

<img src="https://github.com/2kpr/ComfyUI-UltraPixel/blob/main/ComfyUI_00001_.png">

<br/>
Example Output for prompt:
A highly detailed, high-quality image of the Banff National Park in Canada. The turquoise waters of Lake Louise are surrounded by snow-capped mountains and dense pine forests. A wooden canoe is docked at the edge of the lake. The sky is a clear, bright blue, and the air is crisp and fresh.
<br/>
<br/>

<img src="https://github.com/2kpr/ComfyUI-UltraPixel/blob/main/ComfyUI_00002_.png">

<br/>
Example ControlNet Output for prompt:
A close-up portrait of a young woman with flawless skin, vibrant red lipstick, and wavy brown hair, wearing a vintage floral dress and standing in front of a blooming garden, waving
<br/>
<br/>

<img src="https://github.com/2kpr/ComfyUI-UltraPixel/blob/main/cn.png">

<br/>
Example ControlNet Output for prompt:
A close-up portrait of a young woman with blonde hair bobcut wearing a beautiful blue dress giving the thumbs up
<br/>
<br/>

<img src="https://github.com/2kpr/ComfyUI-UltraPixel/blob/main/cn2.png">

### Credits:

All thanks to the team that made UltraPixel:<br/>
https://jingjingrenabc.github.io/ultrapixel/<br/>
https://arxiv.org/abs/2407.02158<br/>
https://github.com/catcathh/UltraPixel<br/>