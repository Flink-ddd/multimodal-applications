import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import logging

# 配置日志，用于在调试不方便时提供信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 调试配置 ---
# 你需要将这个路径替换为你实际的图片路径
# 确保这些图片已经上传到你的RunPod服务器的相应位置
IMAGE_PATH_1 = "images/11113.jpeg"
IMAGE_PATH_2 = "images/WechatIMG105的副本.jpeg"
# VIDEO_PATH_1 = "/path/to/your_project_root/videos/example.mp4" # 如果LLaVA版本支持视频

# 使用Llava-v1.5-7b模型，这是比较常用的入门模型
MODEL_ID = "liuhaotian/llava-v1.5-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

logger.info(f"Using device: {DEVICE}, dtype: {DTYPE}")

def load_llava_model():
    """加载 LLaVA 模型和处理器"""
    logger.info(f"Loading LLaVA model: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True, # 降低CPU内存占用
        device_map="auto" # 自动映射到可用设备
    )
    # model.to(DEVICE) # device_map="auto"通常已经处理
    logger.info("LLaVA model loaded successfully.")
    return processor, model

def run_qa_scenario(processor, model, image_path, question):
    """场景1: 基础图文问答 (Image-to-Text VQA)"""
    logger.info(f"\n--- Running QA Scenario: {question} on {image_path} ---")
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}. Skipping QA scenario.")
        return

    image = Image.open(image_path).convert("RGB")
    
    # 构造prompt，这是LLaVA V1.5的典型格式
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    
    # [在此处设置断点] 观察图像和文本如何被处理器处理
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    
    logger.info("Inputs prepared. Generating response...")
    # [在此处设置断点] 在模型生成之前跳入模型源码
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    
    # [在此处设置断点] 观察原始输出和解码过程
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # 打印和解码可能需要跳过初始的prompt部分
    # 对于LLaVA-V1.5，直接decode通常会包含prompt，需要后处理
    # 简化的后处理，可能需要根据具体输出调整
    if "ASSISTANT:" in response:
        response_text = response.split("ASSISTANT:", 1)[1].strip()
    else:
        response_text = response.strip()

    logger.info(f"Question: {question}")
    logger.info(f"Response: {response_text}")
    return response_text

def run_captioning_scenario(processor, model, image_path):
    """场景2: 图像描述/生成文字 (Image Captioning)"""
    logger.info(f"\n--- Running Captioning Scenario on {image_path} ---")
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}. Skipping Captioning scenario.")
        return
        
    image = Image.open(image_path).convert("RGB")
    
    # 图像描述的prompt，可以更开放
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)

    logger.info("Inputs prepared. Generating caption...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

    response = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in response:
        caption_text = response.split("ASSISTANT:", 1)[1].strip()
    else:
        caption_text = response.strip()

    logger.info(f"Caption: {caption_text}")
    return caption_text

def run_zero_shot_reasoning_scenario(processor, model, image_path, task_prompt):
    """场景3: 零样本推理 (Zero-shot Reasoning)"""
    logger.info(f"\n--- Running Zero-shot Reasoning Scenario on {image_path} ---")
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}. Skipping Reasoning scenario.")
        return
        
    image = Image.open(image_path).convert("RGB")
    
    # 零样本推理的prompt，引导模型进行特定任务
    prompt = f"USER: <image>\n{task_prompt} ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)

    logger.info("Inputs prepared. Generating reasoning output...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

    response = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in response:
        reasoning_output = response.split("ASSISTANT:", 1)[1].strip()
    else:
        reasoning_output = response.strip()

    logger.info(f"Task: {task_prompt}")
    logger.info(f"Reasoning Output: {reasoning_output}")
    return reasoning_output

# --- 主执行流程 ---
if __name__ == "__main__":
    # 可选：如果你需要在脚本启动时就等待调试器连接
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678)) # 在需要断点的地方设置端口
    # print("Waiting for debugger attach on port 5678...")
    # debugpy.wait_for_client()
    # print("Debugger attached!")


    processor, model = load_llava_model()

    # --- 运行各个场景 ---

    # 场景1: 基础图文问答
    run_qa_scenario(processor, model, IMAGE_PATH_1, "What is the main subject in this image?")
    run_qa_scenario(processor, model, IMAGE_PATH_2, "Describe the colors and objects present.")

    # 场景2: 图像描述/生成文字
    run_captioning_scenario(processor, model, IMAGE_PATH_1)

    # 场景3: 零样本推理
    run_zero_shot_reasoning_scenario(processor, model, IMAGE_PATH_1, "Based on the image, is this an indoor or outdoor scene? Explain your reasoning.")
    run_zero_shot_reasoning_scenario(processor, model, IMAGE_PATH_2, "What is the relationship between the two main objects depicted?")

    logger.info("\nAll LLaVA scenarios completed.")