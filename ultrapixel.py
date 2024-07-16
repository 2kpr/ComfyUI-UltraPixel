import os
import yaml
import torch
import sys
import folder_paths

from .inference.utils import *
from .train import WurstCoreB
from .gdf import (
    VPScaler,
    CosineTNoiseCond,
    DDPMSampler,
    P2LossWeight,
    AdaptiveLossWeight,
)
from .train import WurstCore_t2i as WurstCoreC


class UltraPixel:
    def __init__(self, pretrained, stage_a, stage_b, stage_c, effnet, previewer):
        self.ultrapixel_path = os.path.join(folder_paths.models_dir, "ultrapixel")
        self.pretrained = os.path.join(self.ultrapixel_path, pretrained)
        self.stage_a = os.path.join(self.ultrapixel_path, stage_a)
        self.stage_b = os.path.join(self.ultrapixel_path, stage_b)
        self.stage_c = os.path.join(self.ultrapixel_path, stage_c)
        self.effnet = os.path.join(self.ultrapixel_path, effnet)
        self.previewer = os.path.join(self.ultrapixel_path, previewer)

    def set_config(
        self,
        height,
        width,
        seed,
        dtype,
        stage_a_tiled,
        stage_b_steps,
        stage_b_cfg,
        stage_c_steps,
        stage_c_cfg,
        prompt,
    ):
        self.height = height
        self.width = width
        self.seed = seed
        self.dtype = dtype
        self.stage_a_tiled = True if stage_a_tiled == "true" else False
        self.stage_b_steps = stage_b_steps
        self.stage_b_cfg = stage_b_cfg
        self.stage_c_steps = stage_c_steps
        self.stage_c_cfg = stage_c_cfg
        self.prompt = prompt

    def process(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float

        base_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(base_path, "configs/training/t2i.yaml")
        with open(config_file, "r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file)
            loaded_config["effnet_checkpoint_path"] = self.effnet
            loaded_config["previewer_checkpoint_path"] = self.previewer
            loaded_config["generator_checkpoint_path"] = self.stage_c
        core = WurstCoreC(config_dict=loaded_config, device=device, training=False)

        config_file_b = os.path.join(base_path, "configs/inference/stage_b_1b.yaml")
        with open(config_file_b, "r", encoding="utf-8") as file:
            config_file_b = yaml.safe_load(file)
            config_file_b["effnet_checkpoint_path"] = self.effnet
            config_file_b["stage_a_checkpoint_path"] = self.stage_a
            config_file_b["generator_checkpoint_path"] = self.stage_b
        core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

        extras = core.setup_extras_pre()
        models = core.setup_models(extras)
        models.generator.eval().requires_grad_(False)
        # print("STAGE C READY")

        extras_b = core_b.setup_extras_pre()
        models_b = core_b.setup_models(extras_b, skip_clip=True)
        models_b = WurstCoreB.Models(
            **{
                **models_b.to_dict(),
                "tokenizer": models.tokenizer,
                "text_model": models.text_model,
            }
        )
        models_b.generator.bfloat16().eval().requires_grad_(False)
        # print("STAGE B READY")

        captions = [self.prompt]
        height, width = self.height, self.width

        pretrained_path = os.path.join(self.ultrapixel_path, self.pretrained)
        sdd = torch.load(pretrained_path, map_location="cpu")
        collect_sd = {}
        for k, v in sdd.items():
            collect_sd[k[7:]] = v

        models.train_norm.load_state_dict(collect_sd)

        models.generator.eval()
        models.train_norm.eval()

        batch_size = 1
        height_lr, width_lr = get_target_lr_size(height / width, std_size=32)
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(
            height, width, batch_size=batch_size
        )
        stage_c_latent_shape_lr, stage_b_latent_shape_lr = calculate_latent_sizes(
            height_lr, width_lr, batch_size=batch_size
        )

        # Stage C Parameters
        extras.sampling_configs["cfg"] = self.stage_c_cfg
        extras.sampling_configs["shift"] = 1
        extras.sampling_configs["timesteps"] = self.stage_c_steps
        extras.sampling_configs["t_start"] = 1.0
        extras.sampling_configs["sampler"] = DDPMSampler(extras.gdf)

        # Stage B Parameters
        extras_b.sampling_configs["cfg"] = self.stage_b_cfg
        extras_b.sampling_configs["shift"] = 1
        extras_b.sampling_configs["timesteps"] = self.stage_b_steps
        extras_b.sampling_configs["t_start"] = 1.0

        for cnt, caption in enumerate(captions):

            batch = {"captions": [caption] * batch_size}
            conditions = core.get_conditions(
                batch,
                models,
                extras,
                is_eval=True,
                is_unconditional=False,
                eval_image_embeds=False,
            )
            unconditions = core.get_conditions(
                batch,
                models,
                extras,
                is_eval=True,
                is_unconditional=True,
                eval_image_embeds=False,
            )

            conditions_b = core_b.get_conditions(
                batch, models_b, extras_b, is_eval=True, is_unconditional=False
            )
            unconditions_b = core_b.get_conditions(
                batch, models_b, extras_b, is_eval=True, is_unconditional=True
            )

            with torch.no_grad():

                models.generator.cuda()
                print("STAGE C GENERATION***************************")
                with torch.cuda.amp.autocast(dtype=dtype):
                    sampled_c = generation_c(
                        batch,
                        models,
                        extras,
                        core,
                        stage_c_latent_shape,
                        stage_c_latent_shape_lr,
                        device,
                    )

                models.generator.cpu()
                torch.cuda.empty_cache()

                conditions_b = core_b.get_conditions(
                    batch, models_b, extras_b, is_eval=True, is_unconditional=False
                )
                unconditions_b = core_b.get_conditions(
                    batch, models_b, extras_b, is_eval=True, is_unconditional=True
                )
                conditions_b["effnet"] = sampled_c
                unconditions_b["effnet"] = torch.zeros_like(sampled_c)
                print("STAGE B + A DECODING***************************")

                with torch.cuda.amp.autocast(dtype=dtype):
                    sampled = decode_b(
                        conditions_b,
                        unconditions_b,
                        models_b,
                        stage_b_latent_shape,
                        extras_b,
                        device,
                        stage_a_tiled=self.stage_a_tiled,
                    )

                torch.cuda.empty_cache()
                imgs = show_images(sampled)
                return imgs[0]
