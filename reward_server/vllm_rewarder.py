from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union

def evaluate_images_with_scores(
    model_path: str,
    image_paths: Union[str, List[str]],
    prompt: str,
    score_idx: List[int] = [15, 16, 17, 18, 19, 20],
    max_new_tokens: int = 128
):
    """
    使用视觉语言模型评估图像并获取归一化分数
    
    Args:
        model_path: 模型路径
        image_paths: 单个图像路径或图像路径列表
        prompt: 文本提示
        score_idx: 分数对应的token ID列表
        max_new_tokens: 最大生成token数
    """
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        dtype=torch.float32, 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 处理图像路径
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 加载图像
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 构建消息
    content = []
    for img in images:
        content.append({
            "type": "image",
            "image": img,
        })
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference with scores
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    # 获取生成的token ids
    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码文本
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # 获取第一步的归一化分数概率
    scores = outputs.scores
    first_step_logits = scores[0][0, score_idx]
    first_step_probs = F.softmax(first_step_logits, dim=-1)
    
    # 计算期望分数
    expected_score = sum(i * first_step_probs[i].item() for i in range(len(score_idx)))
    most_likely_score = torch.argmax(first_step_probs).item()
    
    return {
        "generated_text": output_text,
        "expected_score": expected_score,
        "most_likely_score": most_likely_score,
        "score_probabilities": first_step_probs.cpu().numpy(),
    }

# 使用示例
if __name__ == "__main__":
    model_path = "/apdcephfs_nj7/share_1220751/xianyihe/ckpts/Qwen/Qwen3-VL-8B-Instruct"
    
    # # 单张图像

    
    prompt = prompt = """You are a professional visual assessment expert specializing in analyzing the geometry and realism of objects in images.

You will see two images:

- First image: A real-world scene image (reference image)

- Second image: A new image generated from a different perspective based on the first image (image to be evaluated)

You need to evaluate whether the **shape** of the object in the second image matches the shape characteristics of the object in the first image in the real world.

**Scoring Criteria (0-5 points):**

**0 points** - Completely Distorted

- The object's shape is completely different from reality, exhibiting severe geometric distortion.

- The object's structure has collapsed or exhibited impossible deformation.

- The expected shape of the object is completely unrecognizable.

**1 point** - Severe Deviation

- The object's shape is significantly different from reality, exhibiting multiple severe distortions or errors.

- Key structural features are missing or severely deformed.

- The overall shape is noticeably unnatural.

**2 points** - Significant Deviation

- The object's shape differs significantly from reality and has obvious unnatural features.

- Multiple parts have inaccurate proportions or structures

- Object is recognizable, but its shape is clearly unreasonable

**3 points** - Basically reasonable

- The object's shape is basically realistic, but there are some obvious flaws or inaccuracies.

- Some details are poorly handled, but the overall outline is acceptable.

- There is obvious but not serious distortion.

**4 points** - Good match

- The object's shape matches reality well, with only minor flaws.

- The overall structure is accurate, with only slight deviations in a few details.

- Most observers will not notice any abnormalities.

**5 points** - Perfect match

- The object's shape perfectly matches its real-world counterpart.

- All structures, proportions, and details are natural and accurate.

- Maintains realism when viewed from different angles.

**Output format:**

Only output scores (integers between 0 and 5)
    """


    # prompt = "describe the image"

    # 两张图像
    result = evaluate_images_with_scores(
        model_path=model_path,
        image_paths=["/apdcephfs_nj7/share_1220751/xianyihe/worldmodel/vggt/examples/video/camera_ctrl/input/1.png", "/apdcephfs_nj7/share_1220751/xianyihe/worldmodel/vggt/examples/video/camera_ctrl/kling/1/frame_000120.jpg"],
        prompt=prompt
    )

    # result = evaluate_images_with_scores(
    #     model_path=model_path,
    #     image_paths="/apdcephfs_nj7/share_1220751/xianyihe/worldmodel/vggt/examples/video/camera_ctrl/input/1.png",
    #     prompt="Describe this image."
    # )
    
    print(f"Generated Text: {result['generated_text']}")
    print(f"Expected Score: {result['expected_score']:.4f}")
    print(f"Most Likely Score: {result['most_likely_score']}")
    print(f"Score Probabilities: {result['score_probabilities']}")
