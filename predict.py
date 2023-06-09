# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import torch
import torchvision
from vqgan import VQModel
from transformers import AutoTokenizer, CLIPTextModel
from modules import Paella, Prior
from diffuzz import Diffuzz
import transformers
from cog import BasePredictor, Input, Path


os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.utils.logging.set_verbosity_error()
cache_dir = "models"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"

        self.vqmodel = VQModel().to(device)
        self.vqmodel.load_state_dict(
            torch.load("models/vqgan_f4_v1_500k.pt", map_location=device)["state_dict"]
        )
        self.vqmodel.eval().requires_grad_(False)

        self.clip_model = (
            CLIPTextModel.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                cache_dir=cache_dir,
                local_files_only=True,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
            local_files_only=True,
        )

        pretrained_checkpoint = torch.load(
            "models/model_stage_b.pt", map_location=device
        )

        self.diffuzz = Diffuzz(device=device)

        # - Paella Model as generator -
        generator = Paella(byt5_embd=1024).to(device)
        generator.load_state_dict(pretrained_checkpoint["state_dict"])
        generator.eval().requires_grad_(False)
        del pretrained_checkpoint

        checkpoint = torch.load("models/model_stage_c_ema.pt", map_location=device)
        model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(
            device
        )
        model.load_state_dict(checkpoint["ema_state_dict"])
        model.eval().requires_grad_(False)
        del checkpoint

        torch.cuda.empty_cache()

        self.model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        self.generator = torch.compile(
            generator, mode="reduce-overhead", fullgraph=True
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="Epic drawing of a dog"
        ),
        negative_prompt: str = Input(
            default="low resolution, low detail, bad quality, blurry"
        ),
        num_images: int = Input(
            description="Choose number of output images.", default=1, choices=[1, 2, 4]
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=60
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # def encode(x):
        #     return self.vqmodel.encode(x, quantize=True)[2]

        def decode(img_seq):
            return self.vqmodel.decode_indices(img_seq)

        def embed_clip(caption, negative_caption="", batch_size=4, device="cuda"):
            clip_tokens = self.clip_tokenizer(
                [caption] * batch_size,
                truncation=True,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            clip_text_embeddings = self.clip_model(**clip_tokens).last_hidden_state

            clip_tokens_uncond = self.clip_tokenizer(
                [negative_caption] * batch_size,
                truncation=True,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            clip_text_embeddings_uncond = self.clip_model(
                **clip_tokens_uncond
            ).last_hidden_state
            return clip_text_embeddings, clip_text_embeddings_uncond

        device = "cuda"
        prior_sampler = "ddpm"

        clip_text_embeddings, clip_text_embeddings_uncond = embed_clip(
            prompt, negative_prompt, num_images, device
        )

        effnet_features_shape = (num_images, 16, 12, 12)
        effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(device)
        generator_latent_shape = (num_images, 128, 128)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            s = time.time()
            sampled = self.diffuzz.sample(
                self.model,
                {"c": clip_text_embeddings},
                unconditional_inputs={"c": clip_text_embeddings_uncond},
                shape=effnet_features_shape,
                timesteps=num_inference_steps,
                cfg=guidance_scale,
                sampler=prior_sampler,
                t_start=1.0,
            )[-1]
            print(f"Prior Sampling: {time.time() - s}")
            temperature, cfg, steps = (1.0, 0.6), (2.0, 2.0), 8
            s = time.time()
            sampled_images_original, intermediate = sample(
                self.vqmodel,
                self.generator,
                {"effnet": sampled, "byt5": clip_text_embeddings},
                generator_latent_shape,
                unconditional_inputs={
                    "effnet": effnet_embeddings_uncond,
                    "byt5": clip_text_embeddings_uncond,
                },
                temperature=temperature,
                cfg=cfg,
                steps=steps,
            )
            print(f"Generator Sampling: {time.time() - s}")

        sampled = decode(sampled_images_original)
        out = "/tmp/out.png"
        torchvision.utils.save_image(sampled, out)

        return Path(out)


def sample(
    vqmodel,
    model,
    model_inputs,
    latent_shape,
    unconditional_inputs=None,
    init_x=None,
    steps=12,
    renoise_steps=None,
    temperature=(0.7, 0.3),
    cfg=(8.0, 8.0),
    mode="multinomial",
    t_start=1.0,
    t_end=0.0,
    sampling_conditional_steps=None,
    sampling_quant_steps=None,
):  # 'quant', 'multinomial', 'argmax'
    device = unconditional_inputs["byt5"].device
    if sampling_conditional_steps is None:
        sampling_conditional_steps = steps
    if sampling_quant_steps is None:
        sampling_quant_steps = steps
    if renoise_steps is None:
        renoise_steps = steps - 1
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
    intermediate_images = []
    # with torch.inference_mode():
    init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
    if init_x != None:
        sampled = init_x
    else:
        sampled = init_noise.clone()
    t_list = torch.linspace(t_start, t_end, steps + 1)
    temperatures = torch.linspace(temperature[0], temperature[1], steps)
    cfgs = torch.linspace(cfg[0], cfg[1], steps)
    if cfg is not None:
        model_inputs = {
            k: torch.cat([v, v_u])
            for (k, v), (k_u, v_u) in zip(
                model_inputs.items(), unconditional_inputs.items()
            )
        }
    for i, tv in enumerate(t_list[:steps]):
        if i >= sampling_quant_steps:
            mode = "quant"
        t = torch.ones(latent_shape[0], device=device) * tv

        if cfg is not None and i < sampling_conditional_steps:
            logits, uncond_logits = model(
                torch.cat([sampled] * 2), torch.cat([t] * 2), **model_inputs
            ).chunk(2)
            logits = logits * cfgs[i] + uncond_logits * (1 - cfgs[i])
        else:
            logits = model(sampled, t, **model_inputs)

        scores = logits.div(temperatures[i]).softmax(dim=1)

        if mode == "argmax":
            sampled = logits.argmax(dim=1)
        elif mode == "multinomial":
            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(
                logits.size(0), *logits.shape[2:]
            )
        elif mode == "quant":
            sampled = (
                scores.permute(0, 2, 3, 1) @ vqmodel.vquantizer.codebook.weight.data
            )
            sampled = vqmodel.vquantizer.forward(sampled, dim=-1)[-1]
        else:
            raise Exception(
                f"Mode '{mode}' not supported, use: 'quant', 'multinomial' or 'argmax'"
            )

        intermediate_images.append(sampled)

        if i < renoise_steps:
            t_next = torch.ones(latent_shape[0], device=device) * t_list[i + 1]
            sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
            intermediate_images.append(sampled)
    return sampled, intermediate_images
