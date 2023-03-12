from fastapi import FastAPI
import torch
from io import BytesIO
import base64
from pydantic import BaseModel
# Importing some useful pipelines
from diffusers import StableDiffusionPipeline 
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
class model_params(BaseModel):
    prompt: str

app = FastAPI()

@app.get("/")
async def home():
  return "server running!"

@app.post("/text2image")
async def inference(model_inputs:model_params) -> dict:
    global model
    try:
      model = StableDiffusionPipeline.from_pretrained("model")
    except:
       
      model_id = "stabilityai/stable-diffusion-2-1-base"
      model = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
      model.save_pretrained("model")
    model.to(device)

    #generator = torch.Generator("cuda").manual_seed(42)
    model.enable_attention_slicing()
    model.enable_vae_slicing()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # Run the pipeline, showing some of the available arguments
    pipe_output = model(
        prompt = model_inputs.get('prompt', None), # What to generate
        negative_prompt="Oversaturated, blurry, low quality", # What NOT to generate
        height=480, width=640,     # Specify the image size
        guidance_scale=12,          # How strongly to follow the prompt
        num_inference_steps=35,    # How many steps to take
        #generator=generator        
    )
    
    image = pipe_output.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}

